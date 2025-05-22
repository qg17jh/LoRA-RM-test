import torch
from torch import nn
import torch.nn.functional as F
import random
import copy
import numpy as np
import time
from sklearn.model_selection import train_test_split
import gc
import GPUtil
from torch.cuda.amp import autocast as autocast
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import functools
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import MixedPrecision
import torch.multiprocessing as mp
from datasets import load_dataset, concatenate_datasets
import os
import argparse
import json
from transformers import AutoConfig, AutoTokenizer, PhiForSequenceClassification

from peft import LoraConfig, get_peft_model

# from RM import PhiForSequenceClassification 
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left" # Allow batched inference

BATCH_SIZE = 5
EPOCHS = 1
log_interval = 600
eval_interval = 2400
SEED = 30
T_max = 8000
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodes', default=1,type=int)
parser.add_argument('-g', '--gpus', default=8, type=int)
parser.add_argument('-nr', '--nr', default=0, type=int)
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args = parser.parse_args()
args.world_size = args.gpus * args.nodes
output_dir='./RMsafe'


def main():
    dataset1 = load_dataset("Anthropic/hh-rlhf", data_dir='harmless-base', split="train")
    dataset2 = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").shuffle(seed=30)
    dataset_merge = concatenate_datasets([dataset1, dataset2]).shuffle(seed=30)
    dataset = dataset_merge.train_test_split(0.05)
    
    config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True, num_labels=1, pad_token_id=tokenizer.pad_token_id)
    model = PhiForSequenceClassification(config).from_pretrained("microsoft/phi-2", config=config)
    # ===== LoRA Configuration & Wrapping ===== 
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Phi-2's attention layers
        modules_to_save=["classifier"],       # Train classification head
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Verify only 0.1-1% params are trainable
    # ========================================

    mp.spawn(train, args=(model, dataset['train'], dataset['test'], args.nr, args.gpus, args.world_size), nprocs=args.world_size, join=True, start_method='spawn')

    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RMDataLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __iter__(self):
        for data in self.dataset:
            if data['prompt'] is not None: # process data from PKU-Alignment/PKU-SafeRLHF (safe)
                x = tokenizer.encode("Instruct: " + data['prompt'].strip() + "\nOutput: " + \
                                     data['response_0'].strip() + '\n' + tokenizer.eos_token) 
                y = tokenizer.encode("Instruct: " + data['prompt'].strip() + "\nOutput: " + \
                                     data['response_1'].strip() + '\n' + tokenizer.eos_token)
                if len(x) > 512 or len(y) > 512:
                    continue
                if data['safer_response_id'] == 0:
                    yield x, y
                else:
                    yield y, x
            else: # process data from Anthropic Harmless
                better = tokenizer(data['chosen'].replace('\nHuman:', 'Instruct:').replace("\nAssistant:", "Output:").strip()+'\n'+tokenizer.eos_token).input_ids
                worse = tokenizer(data['rejected'].replace('\nHuman:', 'Instruct:').replace("\nAssistant:", "Output:").strip()+'\n'+tokenizer.eos_token).input_ids
                if len(better) > 512 or len(worse) > 512:
                    continue
                yield better, worse
                            

def collate_fn(batch):
    max_len1 = max([len(x) for x, _ in batch])
    max_len2 = max([len(y) for _, y in batch])
    res0 = []
    res1 = []
    for x, y in batch:
        curr_len1 = len(x)
        curr_len2 = len(y)
        res0.append([tokenizer.pad_token_id] * (max_len1 - curr_len1) + x)
        res1.append([tokenizer.pad_token_id] * (max_len2 - curr_len2) + y)
    return torch.tensor(res0), torch.tensor(res1)

def validate(model, val_loader):
    model.eval()
    losses = []
    for src, tgt in val_loader:
        src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
        am1 = src != tokenizer.pad_token_id
        am2 = tgt != tokenizer.pad_token_id
        r1 = model(src, attention_mask=am1, output_hidden_states=True).logits
        r2 = model(tgt, attention_mask=am2, output_hidden_states=True).logits
        loss = -1 * nn.functional.logsigmoid(r1 - r2)
        losses.append(loss.mean().item())
    model.train()
    return sum(losses) / len(losses)
    
    
def train(gpu, model, train_dataset, val_dataset, nr, gpus, world_size):
    rank = nr * gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=world_size,                              
    	rank=rank                                               
    )
    train = RMDataLoader(train_dataset)
    val = RMDataLoader(val_dataset)
    train_loader = IterableDatasetShard(
        dataset=train,
        batch_size=BATCH_SIZE,
        num_processes=world_size,
        process_index=rank,
    )
    val_loader = IterableDatasetShard(
        dataset=val,
        batch_size=3,
        num_processes=world_size,
        process_index=rank,
    )
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_loader,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
    	dataset=val_loader,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn,
    )
    torch.cuda.set_device(gpu)
    my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e8))
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=False), device_id=gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.95), eps=0.00001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min = 0.000001)
    
    for epoch in range(1, EPOCHS+1): 
        epoch_loss = 0
        step = 0 
        start = time.time()
        model.train()
        for src, tgt in train_loader:
            step += 1
            optimizer.zero_grad()
            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
            am1 = src != tokenizer.pad_token_id
            am2 = tgt != tokenizer.pad_token_id
            r1 = model(src, attention_mask=am1, output_hidden_states=True).logits
            r2 = model(tgt, attention_mask=am2, output_hidden_states=True).logits
            loss = (-1 * nn.functional.logsigmoid(r1 - r2)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            
            if step % log_interval == 0 and gpu == 0:               
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    (
                        "Epoch: %3d |LR: %10.7f | Step: %3d | Epoch Loss: %7.3f "
                        + "| Sec: %5.1f "
                    )
                    % (epoch, lr, step, epoch_loss / log_interval, elapsed)
                )
                epoch_loss = 0
                gc.collect()
                torch.cuda.empty_cache()
                start = time.time()
                
                
            if step % eval_interval == 0:
                val_loss = validate(model, val_loader)
                if gpu == 0:
                    print(f"Step {step}: val loss {val_loss:.4f}")
                               
        if gpu == 0:
            GPUtil.showUtilization()
    #model.save_pretrained(output_dir, state_dict=model.state_dict())
    if gpu == 0:
        model.module.save_pretrained(output_dir)

if __name__=="__main__":
    set_seed(SEED)
    main()

from components_reg.dataset import CLRPDataset_finetune, CLRPDataset_pseudo, CLRPDataset_pseudo_5fold
from components_reg.util import seed_everything, create_folds, generate_config
from components_reg.model import Custom_bert
from components_reg.optimizer import get_optimizer, get_scheduler
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer
from tqdm import tqdm
import numpy as np
import os
import gc
from accelerate import Accelerator
gc.enable()
from accelerate import DistributedDataParallelKwargs
from transformers import DataCollatorWithPadding

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_fold_ft(fold,config,train_data,tokenizer,t_bar):
    # device = "cuda:0"
    #prep train/val datasets
    print(f"Now Run The FOLD {fold}")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    train_dataset = CLRPDataset_finetune(True, fold,train_data,tokenizer, config)
    val_dataset = CLRPDataset_finetune(False, fold,train_data,tokenizer, config)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
    #                                            pin_memory=True, num_workers=6, collate_fn=data_collator)
    data_collator = DataCollatorWithPadding(tokenizer, max_length=32)  # pad_to_multiple_of=8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=6, collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["eval_batch_size"], shuffle=False, pin_memory=True, num_workers=6, collate_fn=data_collator)
    
    total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'] / config['n_gpu'])
    print("Total Update time: ", len(train_loader), config['num_epoch'], config['batch_size'])
    val_step = 2000
    min_valid_acc = 0
    
    #load model
    model = Custom_bert(config['model_dir'])  #.to(device)
    _ = model.eval()

    model.load_state_dict(torch.load(config['pretrained_path'], accelerator.device), strict=False)
    if config["finetune_long"]:
        print("\n FineTune With Long Sequence: !!! Load Pretrained State Dict\n")
        model.load_state_dict(torch.load(f"./Qymodels/roberta_1/roberta_large_{fold}.pt", accelerator.device), strict=True)

    #get optimizer and scheduler
    optimizer = get_optimizer(model,config)
    lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

    step = 0
    min_step = 0
    last_save_step = 0
    last_save_index = 0
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    #seed_everything(seed=config['seed_'] + fold)

    # optimizer.zero_grad()
    for epoch in range(config['num_epoch']):
        model.train()
        count = 0
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'] # to(device)
            attention_mask = batch['attention_mask'] # to(device)
            target = batch['target'] # to(device)

            outputs = model(input_ids, attention_mask)

            cls_loss = nn.BCEWithLogitsLoss()(torch.squeeze(outputs,1),target)

            loss = cls_loss / config['accumulation_steps']

            total_loss+=torch.pow(nn.BCEWithLogitsLoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']

            # loss.backward()
            accelerator.backward(loss)

            if (count+1) % config['accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                count = 0
                total_loss = 0
            else:
                count+=1
            #only save in radius of certain step
            do_val = False
            if step >= (config['save_center']-config['save_radius']): #  and step <= (config['save_center']+config['save_radius'])
                val_step = 1000
                do_val = True
            # if config['only_val_in_radius']:
            #     if step < (config['save_center']-config['save_radius']) or step > (config['save_center']+config['save_radius']):
            #         do_val = False

            if ((step+1) % val_step == 0 and count == 0) and do_val:
                model.eval()
                l_val = nn.BCEWithLogitsLoss(reduction='sum')
                losses = []
                accs = []
                with torch.no_grad():
                    total_loss_val = 0
                    for batch in val_loader:
                        input_ids = batch['input_ids']  # to(device)
                        attention_mask = batch['attention_mask']  # .to(device)
                        outputs = model(input_ids, attention_mask)

                        cls_loss_val = l_val(torch.squeeze(outputs), batch['target'])
                        acc = ((torch.squeeze(outputs)>0) == batch['target'].bool())
                        #  losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size))) 原因是原先transformers模型output.loss返回的是一个平均值!!
                        # losses.append(accelerator.gather(cls_loss_val.repeat(config["eval_batch_size"])))
                        losses.append(accelerator.gather(cls_loss_val.repeat(config["eval_batch_size"])))
                        accs.append(accelerator.gather(acc))

                        # val_loss = cls_loss_val
                    losses = torch.cat(losses)
                    losses = losses[: len(val_dataset)]
                    total_loss_val = torch.sqrt(torch.mean(losses)).item()
                    accs = torch.cat(accs)
                    accs = accs[:len(val_dataset)]
                    total_acc_val = torch.mean(accs.float()).item()

                    #
                    # total_loss_val+=val_loss.item()
                    # total_loss_val/=len(val_dataset)
                    # total_loss_val = total_loss_val**0.5

                    if min_valid_acc < total_acc_val and step >= (config['save_center']-config['save_radius']):# step <= (config['save_center']+config['save_radius']):
                        #saves model with lower loss
                        min_step = step
                        min_valid_acc = total_acc_val
                        print(f"FOLD {fold} min ACC updated to ",min_valid_acc," at step ",min_step)
                        if not os.path.isdir('./Qymodels'):
                            os.mkdir('./Qymodels')
                        if not os.path.isdir(config['save_path']):
                            os.mkdir(config['save_path'])
                        if 'roberta' in config['model_dir']:
                            # torch.save(model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt') accelerator modified
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save(unwrapped_model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt')
                            # unwrapped_model.save_pretrained(config['save_path'], save_function=accelerator.save)
                        else:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save(unwrapped_model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                            # torch.save(model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                model.train()
            t_bar.update(1)
            step += 1
    del model,train_dataset,train_loader,val_dataset,val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return min_valid_acc, min_step

def train_ft(config, resume_from_fold=0):
    seed_everything(config['seed_'])
    
    train_data = pd.read_csv("./QyData/train.csv")
    train_data = create_folds(train_data, num_splits=5)  # 用Sturge's rule并且做了StratifiedKFold
    model_dir = config['model_dir']
    if config["model_type"]!="de":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)

    t_bar = tqdm(total=((533982*0.8//config['batch_size'])+1)*config['num_epoch']*config['n_folds']/config["n_gpu"])
    # t_bar = tqdm(total=((51611*0.8//config['batch_size'])+1)*config['num_epoch']*config['n_folds']/config["n_gpu"])
    if resume_from_fold!=0:
        t_bar.update(((533982*0.8//config['batch_size'])+1)*config['num_epoch']*config['n_folds']/config["n_gpu"]/5*2)
    train_losses = []
    for i in range(resume_from_fold, config['n_folds']):
        loss, m_step = run_fold_ft(i,config,train_data,tokenizer,t_bar)
        train_losses.append(loss)
    return train_losses

def train_ft_longsequence(config):
    seed_everything(config['seed_'])
    train_data = pd.read_csv("./QyData/train.csv")
    train_data = create_folds(train_data, num_splits=5)  # 用Sturge's rule并且做了StratifiedKFold
    model_dir = config['model_dir']
    if config["model_type"] != "de":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=32)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=32)

    t_bar = tqdm(
        total=((533982 * 0.8 // config['batch_size']) + 1) * config['num_epoch'] * config['n_folds'] / config["n_gpu"])
    train_losses = []
    for i in range(config['n_folds']):
        loss, m_step = run_fold_ft(i, config, train_data, tokenizer, t_bar)
        train_losses.append(loss)
    return train_losses



def train_pseudo(config, label_path):
    device = "cuda:0"
    seed_everything(config['seed_'])
    train_data = pd.read_csv("./data/train.csv")
    train_data = create_folds(train_data, num_splits=5)
    
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)
    
    train_dataset = CLRPDataset_pseudo(True,label_path,train_data,tokenizer)
    t_bar = tqdm(total=((len(train_dataset)//config['batch_size'])+1)*config['num_epoch'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    val_dataset = CLRPDataset_pseudo(False,label_path,train_data,tokenizer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'])
    val_step = 100*config['accumulation_steps']
    min_valid_loss = np.inf

    model = Custom_bert(config['model_dir']).to(device)
    _ = model.eval()

    if config['pretrained_path'] not in [None,'None']:
        print(model.load_state_dict(torch.load(config['pretrained_path']), strict=False))

    optimizer = get_optimizer(model,config)
    lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

    step = 0
    min_step = 0
    last_save_step = 0
    last_save_index = 0

    optimizer.zero_grad()
    for epoch in range(config['num_epoch']):
        model.train()
        count = 0
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            outputs = model(input_ids, attention_mask)

            cls_loss = nn.MSELoss()(torch.squeeze(outputs,1),target)

            loss = cls_loss / config['accumulation_steps']

            total_loss+=torch.pow(nn.MSELoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']
            loss.backward()

            if (count+1) % config['accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                count = 0
                total_loss = 0
            else:
                count+=1

            if ((step+1) % val_step == 0):
                l_val = nn.MSELoss(reduction='sum')
                with torch.no_grad():
                    model.eval()
                    total_loss_val = 0
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        outputs = model(input_ids, attention_mask)

                        cls_loss_val = l_val(torch.squeeze(outputs),batch['target'].to(device))

                        val_loss = cls_loss_val

                        total_loss_val+=val_loss.item()
                    total_loss_val/=len(val_dataset)
                    total_loss_val = total_loss_val**0.5

                if min_valid_loss > total_loss_val:
                    min_step = step
                    min_valid_loss = total_loss_val
                    #print("min loss updated to ",min_valid_loss," at step ",min_step)
                    # Saving State Dict
                    if not os.path.isdir(config['save_path']):
                        os.mkdir(config['save_path'])
                    torch.save(model.state_dict(), config['save_path'] + config['pseudo_save_name'])
                model.train()
            step+=1
            t_bar.update(1)
    del model,train_dataset,train_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return min_valid_loss

def train_pseudo_5fold(config, label_path):
    # device = "cuda:0"
    seed_everything(config['seed_'])
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    train_data = pd.read_csv("./data/train.csv")
    train_data = create_folds(train_data, num_splits=5)
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)

    min_val_losses = []
    for fold in range(config['n_folds']):
        train_dataset = CLRPDataset_pseudo_5fold(True,fold,train_data,tokenizer,label_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=6)
        
        val_dataset = CLRPDataset_pseudo_5fold(False,fold,train_data,tokenizer,label_path)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=6)

        if fold == 0:
            t_bar = tqdm(total=((len(train_dataset)*5//config['batch_size'])+1)*config['num_epoch'])
            
        val_step = 100*config['accumulation_steps']
        min_valid_loss = np.inf

        model = Custom_bert(config['model_dir']) # to(device)
        _ = model.eval()

        if config['pretrained_path'] not in [None,'None']:
            model.load_state_dict(torch.load(config['pretrained_path']), strict=False)

        optimizer = get_optimizer(model,config)
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

        total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps']) / config["n_gpu"]
        lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

        step = 0
        min_step = 0
        last_save_step = 0
        last_save_index = 0

        optimizer.zero_grad()
        for epoch in range(config['num_epoch']):
            model.train()
            count = 0
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'] # .to(device)
                attention_mask = batch['attention_mask'] # .to(device)
                target = batch['target'] # .to(device)

                outputs = model(input_ids, attention_mask)

                cls_loss = nn.MSELoss()(torch.squeeze(outputs,1),target)

                loss = cls_loss / config['accumulation_steps']

                total_loss+=torch.pow(nn.MSELoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']
                loss.backward()

                if (count+1) % config['accumulation_steps'] == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    count = 0
                    total_loss = 0
                else:
                    count+=1

                if ((step+1) % val_step == 0):
                    model.eval()
                    l_val = nn.MSELoss(reduction='sum')
                    with torch.no_grad():
                        total_loss_val = 0
                        losses = []
                        for batch in val_loader:
                            input_ids = batch['input_ids'] # to(device)
                            attention_mask = batch['attention_mask'] # .to(device)
                            outputs = model(input_ids, attention_mask)

                            cls_loss_val = l_val(torch.squeeze(outputs),batch['target'])
                            losses.append(accelerator.gather(cls_loss_val))

                        losses = torch.cat(losses)
                        losses = losses[: len(val_dataset)]
                        total_loss_val = torch.sqrt(torch.mean(losses)).item()
                            # val_loss = cls_loss_val

                        #     total_loss_val+=val_loss.item()
                        # total_loss_val/=len(val_dataset)
                        # total_loss_val = total_loss_val**0.5

                    if min_valid_loss > total_loss_val and epoch > 0:
                        min_step = step
                        min_valid_loss = total_loss_val
                        if not os.path.isdir('./models'):
                            os.mkdir('./models')
                        if not os.path.isdir(config['save_path']):
                            os.mkdir(config['save_path'])
                        if 'roberta' in config['model_dir']:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save(unwrapped_model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt')
                            # torch.save(model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt')
                        else:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save(unwrapped_model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                            # torch.save(model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                    model.train()
                step+=1
                t_bar.update(1)
        del model,train_dataset,train_loader,val_dataset,val_loader
        gc.collect()
        torch.cuda.empty_cache()
        min_val_losses.append(min_valid_loss)
    return min_val_losses

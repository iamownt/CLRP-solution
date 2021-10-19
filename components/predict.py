from components.dataset import CLRPDataset_pred
from components.model import Custom_bert
import numpy as np
import torch
import os
import gc
gc.enable()
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
# device = "cuda:0"

def run_fold(fold_num,model_path,data):
    accelerator = Accelerator()
    if 'roberta' in model_path:
        model_dir = './pretrained/roberta-large/'
        model_name = f"roberta_large_{fold_num}.pt"
    elif 'deberta' in model_path:
        model_dir = './pretrained/deberta-large/'
        model_name = f"deberta_large_{fold_num}.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)    
    model = Custom_bert(model_dir)
    # print(os.system("pwd"), (model_path+model_name))
    model.load_state_dict(torch.load(model_path+model_name))


    
    test_ds = CLRPDataset_pred(data,tokenizer)
    eval_bs = 60
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size = eval_bs,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=12)
    model, test_dl = accelerator.prepare(model, test_dl)
    _ = model.eval()
    pred = []
    identify_id = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            idf_ids = batch['ids']
            output = model(input_ids, attention_mask)
            pred.append(accelerator.gather(output).detach().cpu().numpy())
            identify_id.append(accelerator.gather(idf_ids).detach().cpu().numpy())
            # print(identify_id[0])
            # pred.extend(output.detach().cpu().numpy())
    # pred = pred[:len(test_dl)]  #可能会超过一部分 如果用这个，就说明输入是batch_size的整数倍，如果要真实的取出来。801*120=96120！=96006
    # identify_id = identify_id[:len(test_dl)]   #可能会超过一部分 如果用这个，就说明输入是batch_size的整数倍，如果要真实的取出来

    del model, test_dl, test_ds
    gc.collect()
    torch.cuda.empty_cache()
            
    # return np.array(pred)
    return np.concatenate(pred)[:len(test_ds)], np.concatenate(identify_id)[:len(test_ds)]

def get_single_model(pth,data):
    pred0, _ = run_fold(0,pth,data)
    pred1, _ = run_fold(1,pth,data)
    pred2, _ = run_fold(2,pth,data)
    pred3, _ = run_fold(3,pth,data)
    pred4, _ = run_fold(4,pth,data)
    
    return [pred0,pred1,pred2,pred3,pred4]

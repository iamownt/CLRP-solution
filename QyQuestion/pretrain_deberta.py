import sys

sys.path.append("/hhd/wt106/")

import pandas as pd
from transformers import AutoTokenizer, RobertaForMaskedLM, BertTokenizer, BertForMaskedLM, DebertaForMaskedLM
from transformers import Trainer, TrainingArguments
from components.util import seed_everything
from components.dataset import MLMDataset, QyMLMDataset
from components.optimizer import get_optimizer_robertaMLM, get_scheduler, get_optimizer_bertMLM, get_optimizer_debertaMLM
import torch
import os
import sys
# import wandb
os.environ["WANDB_DISABLED"] = "true"

def deal_with_raw_data(input_files=None, input_no_label=None):
    text_a = []
    text_b = []
    for file in input_files:
        with open(file, "r") as f:
            tmp = f.read()
        for mix_text in tmp.split("\n"):
            if mix_text:
                text_1, text_2, _ = mix_text.split("\t")
                if (text_1 and text_2):
                    text_a.append(text_1)
                    text_b.append(text_2)
    test_A = pd.read_csv(input_no_label, sep="\t", header=None)
    text_a.extend(test_A.iloc[:,0].to_list())
    text_b.extend(test_A.iloc[:,1].to_list())
    pd.DataFrame({"text_a": text_a, "text_b": text_b}).to_csv("QyData/mlm.csv", index=False)

def main():
    # wandb.login()
    if not os.path.exists("QyData/mlm.csv"):
        input_files = ["QyData/train.txt", "QyData/dev.txt"]
        input_no_label = "QyData/test_A.tsv"
        deal_with_raw_data(input_files, input_no_label)

    # model_dir = "/hhd/wt930/user_data/official_model/transformers/chinese-roberta-wwm-ext-large/"
    model_dir = "/hhd/wt930/user_data/official_model/transformers/deberta-large/"
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)
    model = DebertaForMaskedLM.from_pretrained(model_dir, local_files_only=True)

    seed_everything(456982)
    texts = pd.read_csv("QyData/mlm.csv")

    train_dataset = QyMLMDataset(True,texts,tokenizer)
    val_dataset = QyMLMDataset(False,texts,tokenizer)
    config = {
        'lr_type':'custom',
        'base_lr':9e-5,
        'head_lr':1.2e-4,
        'min_lr':4e-5,
        'low_lr':2e-5,
        'n_epoch':5,
        'bs':24,
        'ga':1,
        'lr_scheduler_mul_factor':2,
        'weight_decay':0.01,
        'warm_up_ratio':0.2,
        'decline_1': 0.2,
        'decline_2': 0.7,
        'decline_3': 0.8,
        'decline_4': 0.9,
        'layerwise_decay_rate': 0.9**0.5,
        'betas': (0.9,0.993),
    }

    train_len = len(train_dataset)
    total_train_steps = int(train_len * config['n_epoch'] / config['ga'] / config['bs'])
    optimizer = get_optimizer_debertaMLM(model,config)
    lr_scheduler = get_scheduler(optimizer, total_train_steps, config)

    training_args = TrainingArguments(
        output_dir='./',          # output directory
        num_train_epochs=config['n_epoch'],              # total number of training epochs
        overwrite_output_dir=True,
        per_device_train_batch_size=config['bs'],  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        weight_decay=0.01,               # strength of weight decay
        logging_strategy='no',
        gradient_accumulation_steps = config['ga'],
        save_strategy = "no",
        evaluation_strategy= 'epoch',
        prediction_loss_only=True,
        learning_rate = config['base_lr'],
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        optimizers = (optimizer, lr_scheduler)
    )

    trainer.train()
    try:
        trainer.save_model("/hhd/wt106/QyQuestion/Qymodels/trainer/de")
    except:
        print("Wrong save!")
    if not os.path.isdir('./Qymodels'):
        os.mkdir('./Qymodels')
    dict_ = model.state_dict()
    for key in list(dict_.keys()):
        dict_[key.replace('deberta.', 'base.')] = dict_.pop(key)
    local_rank = torch.distributed.get_rank()  # 防止多个进程互相抢占
    if (local_rank == 0):
        torch.save(dict_, f'./Qymodels/deberta_large_pretrain.pt')




if __name__ == "__main__":
    main()

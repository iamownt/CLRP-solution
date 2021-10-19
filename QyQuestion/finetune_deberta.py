#args:
#1. type of model: 'ro' or 'de'
#2. pretrained path
#3. save path
#4. lr type 'custom' or '3stage'
#5. lr config type 1-3
#    1.training from scratch (use lr type 'custom')
#    2.pseudo pretrain (use lr type 'custom')
#    3.pseudo finetune (use lr type '3stage')
import sys

# sys.path.append("/hhd/wt106/")

import os
from components_reg.train import train_ft
from components_reg.util import generate_config
import sys
import numpy as np
import pandas as pd


def deal_with_label_data(input_files=None):
    text_a = []
    text_b = []
    labels = []
    for file in input_files:
        with open(file, "r") as f:
            tmp = f.read()
        for mix_text in tmp.split("\n"):
            if mix_text:
                text_1, text_2, label = mix_text.split("\t")
                if (text_1 and text_2 and label):
                    text_a.append(text_1)
                    text_b.append(text_2)
                    labels.append(label)
    data = pd.DataFrame({"text_a": text_a, "text_b": text_b, "label": labels})
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data.to_csv("QyData/train.csv", index=False)


def main():
    ###
    # training using provided training data
    ###
    if not os.path.exists("QyData/train.csv"):
        input_files = ["QyData/train.txt", "QyData/dev.txt"]
        deal_with_label_data(input_files)


    # config = generate_config(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    config = generate_config('de', './Qymodels/deberta_large_pretrain.pt', './Qymodels/deberta_1/', 'custom', '1', "cls")
    config["finetune_long"] = False
    losses = train_ft(config)
    print(np.mean(losses),'\n',losses)

if __name__ == "__main__":
    main()

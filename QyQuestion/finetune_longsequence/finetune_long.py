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

sys.path.append("/hhd/wt106/")
import os
from components_reg.train import train_ft, train_ft_longsequence
from components_reg.util import generate_config
import sys
import numpy as np
import pandas as pd


def main():
    ###
    # training using provided training data
    ###
    # config = generate_config(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    config = generate_config('ro', './Qymodels/roberta_large_pretrain.pt', './Qymodels/roberta_long/', 'custom', '1', "cls")
    losses = train_ft(config, resume_from_fold=2)

    losses = train_ft_longsequence(config)
    print(np.mean(losses),'\n',losses)

if __name__ == "__main__":
    main()

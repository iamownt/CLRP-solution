#!/bin/sh
python3.9 0.prepare_data.py
python3.9 1.roberta_pretrain.py

python3.9 2.finetune.py ro ./models/roberta_large_pretrain.pt ./models/roberta_1/ custom 1
#accelerate launch 2.finetune.py ro ./models/roberta_large_pretrain.pt ./models/roberta_1/ custom 1  # 微调5个Fold的Roberta-large模型，有warmup

python3.9 4.predict.py ./extra_data/extra_excerpt.csv ./extra_data/pseudo_1/ 1 1 ./models/roberta_1/
#accelerate launch 4.predict.py ./extra_data/extra_excerpt.csv ./extra_data/pseudo_1/ 1 1 ./models/roberta_1/  #  用5个Fold生成5个Fold的Pesudo Label，防止Leak

python3.9 3.pseudo_train.py de ./extra_data/pseudo_1/ ./models/deberta_1/ 1
#accelerate 3.pseudo_train.py de ./extra_data/pseudo_1/ ./models/deberta_1/ 1  # MixData训练5个Fold的Deberta

python3.9 4.predict.py ./extra_data/extra_excerpt.csv ./extra_data/pseudo_2/ 2 1 ./models/roberta_1/ ./models/deberta_1/  #用训练好的Roberta和Deberta生成PesudoLabel (Cycle 2)
#accelerate launch 4.predict.py ./extra_data/extra_excerpt.csv ./extra_data/pseudo_2/ 2 1 ./models/roberta_1/ ./models/deberta_1/

# 后面全换成accelerate launch
python3.9 3.pseudo_train.py de ./extra_data/pseudo_2/ ./models/deberta_2/ 0 #用Deberta训练MixData

python3.9 2.finetune.py de ./models/deberta_2/deberta_large_single.pt ./models/deberta_2/ 3stage 3  # 三段逐渐下降的学习率

python3.9 4.predict.py ./extra_data/extra_excerpt.csv ./extra_data/pseudo_3/ 3 1 ./models/roberta_1/ ./models/deberta_1/ ./models/deberta_2/ #三个模型再生成PesudoData

python3.9 3.pseudo_train.py ro ./extra_data/pseudo_3/ ./models/roberta_2/ 0

python3.9 2.finetune.py ro ./models/roberta_2/roberta_large_single.pt ./models/roberta_2/ 3stage 3

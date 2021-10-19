import sys

sys.path.append("/hhd/wt106/")
import os
from components_reg.train import train_ft
from components_reg.util import generate_config
import sys
import numpy as np
import pandas as pd
from components_reg.dataset import CLRPDataset_finetune, CLRPDataset_pseudo, CLRPDataset_pseudo_5fold, CLRPDataset_pred
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


def eval_ft(config):
    config["eval_batch_size"] = 168
    seed_everything(config['seed_'])

    train_data = pd.read_csv("./QyData/train.csv")
    train_data = create_folds(train_data, num_splits=5)  # 用Sturge's rule并且做了StratifiedKFold
    model_dir = config['model_dir']
    if config["model_type"] != "de":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)

    t_bar = tqdm(
        total=((533982 * 0.2 // config['batch_size']) + 1) * config['n_folds'] / config["n_gpu"])
    train_losses = []
    save_fold_prediction = True
    for i in range(config['n_folds']):
        run_fold_eval(i, config, train_data, tokenizer, t_bar, save_fold_prediction)
    return train_losses


def run_fold_eval(fold, config, train_data, tokenizer, t_bar, save_fold_prediction):
    # device = "cuda:0
    # prep train/val datasets
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # train_dataset = CLRPDataset_finetune(True, fold, train_data, tokenizer)
    val_dataset = CLRPDataset_finetune(False, fold, train_data, tokenizer, config)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
    #                                            pin_memory=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["eval_batch_size"], shuffle=False,
                                             pin_memory=True, num_workers=12)

    # total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'] / config['n_gpu'])
    # print(len(train_loader), config['num_epoch'])
    val_step = 300
    min_valid_acc = 0

    # load model
    model = Custom_bert(config['model_dir'])  # .to(device)
    _ = model.eval()

    model.load_state_dict(torch.load(config['pretrained_path'], accelerator.device), strict=False)
    model.load_state_dict(torch.load(config['save_path'] + f'roberta_large_{fold}.pt', accelerator.device), strict=True)


    # get optimizer and scheduler
    # optimizer = get_optimizer(model, config)
    # lr_scheduler = get_scheduler(optimizer, total_train_steps, config)

    step = 0
    min_step = 0
    last_save_step = 0
    last_save_index = 0
    model, val_loader = accelerator.prepare(model, val_loader)

    # seed_everything(seed=config['seed_'] + fold)

    # optimizer.zero_grad()
    # for epoch in range(config['num_epoch']):
        # model.train()
        # count = 0
        # total_loss = 0
        # for batch in train_loader:
        #     input_ids = batch['input_ids']  # to(device)
        #     attention_mask = batch['attention_mask']  # to(device)
        #     target = batch['target']  # to(device)
        #
        #     outputs = model(input_ids, attention_mask)
        #
        #     cls_loss = nn.BCEWithLogitsLoss()(torch.squeeze(outputs, 1), target)
        #
        #     loss = cls_loss / config['accumulation_steps']
        #
        #     total_loss += torch.pow(nn.BCEWithLogitsLoss()(torch.squeeze(outputs, 1), target), 0.5).item() / config[
        #         'accumulation_steps']
        #
        #     # loss.backward()
        #     accelerator.backward(loss)
        #
        #     if (count + 1) % config['accumulation_steps'] == 0:
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()
        #         count = 0
        #         total_loss = 0
        #     else:
        #         count += 1
        #     # only save in radius of certain step
        #     if step >= (config['save_center'] - config[
        #         'save_radius']):  # and step <= (config['save_center']+config['save_radius'])
        #         val_step = 300
        #     do_val = True
        #     # if config['only_val_in_radius']:
        #     #     if step < (config['save_center']-config['save_radius']) or step > (config['save_center']+config['save_radius']):
        #     #         do_val = False
        #
        #     if ((step + 1) % val_step == 0 and count == 0) and do_val:
    print("Eval Model: ", )
    model.eval()
    # l_val = nn.BCEWithLogitsLoss(reduction='sum')
    # losses = []
    accs = []
    oof_prediction = []
    with torch.no_grad():
        total_loss_val = 0
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids']  # to(device)
            attention_mask = batch['attention_mask']  # .to(device)
            outputs = model(input_ids, attention_mask)

            # cls_loss_val = l_val(torch.squeeze(outputs), batch['target'])
            acc = ((torch.squeeze(outputs) > 0) == batch['target'].bool())
            # oof_prediction.append((torch.squeeze(outputs)).cpu().numpy())
            #  losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size))) 原因是原先transformers模型output.loss返回的是一个平均值!!
            # losses.append(accelerator.gather(cls_loss_val.repeat(config["eval_batch_size"])))
            # losses.append(accelerator.gather(cls_loss_val.repeat(config["eval_batch_size"])))
            accs.append(accelerator.gather(acc))
            oof_prediction.append(accelerator.gather((torch.squeeze(outputs))))

            # val_loss = cls_loss_val
        # losses = torch.cat(losses)
        # losses = losses[: len(val_dataset)]
        print("Total length: ", len(val_dataset))
        # total_loss_val = torch.sqrt(torch.mean(losses)).item()
        total_oof_prediction = torch.cat(oof_prediction)
        total_oof_prediction = total_oof_prediction[:len(val_dataset)]
        print(f"Gather Total {len(total_oof_prediction)} predictions")
        accs = torch.cat(accs)
        accs = accs[:len(val_dataset)]
        total_acc_val = torch.mean(accs.float()).item()
        print(f"THE FOLD {fold} ACCURACY IS: ", total_acc_val)
        if save_fold_prediction:
            print("\nSAVE FOLD PREDICTION: \n")
            np.save(f"/hhd/wt106/QyQuestion/QyData/{fold}_roberta_prediction.npy", total_oof_prediction.cpu().numpy())
            # np.save(f"/hhd/wt106/QyQuestion/QyData/{fold}_acc.npy", accs.cpu().numpy())
            # np.save(f"/hhd/wt106/QyQuestion/QyData/{fold}_loss.npy", losses.cpu().numpy())

                    #
                    # total_loss_val+=val_loss.item()
                    # total_loss_val/=len(val_dataset)
                    # total_loss_val = total_loss_val**0.5

    #                 if min_valid_acc < total_acc_val and step >= (config['save_center'] - config[
    #                     'save_radius']):  # step <= (config['save_center']+config['save_radius']):
    #                     # saves model with lower loss
    #                     min_step = step
    #                     min_valid_acc = total_acc_val
    #                     print("min ACC updated to ", min_valid_acc, " at step ", min_step)
    #                     if not os.path.isdir('./Qymodels'):
    #                         os.mkdir('./Qymodels')
    #                     if not os.path.isdir(config['save_path']):
    #                         os.mkdir(config['save_path'])
    #                     if 'roberta' in config['model_dir']:
    #                         # torch.save(model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt') accelerator modified
    #                         accelerator.wait_for_everyone()
    #                         unwrapped_model = accelerator.unwrap_model(model)
    #                         accelerator.save(unwrapped_model.state_dict(),
    #                                          config['save_path'] + f'roberta_large_{fold}.pt')
    #                         # unwrapped_model.save_pretrained(config['save_path'], save_function=accelerator.save)
    #                     else:
    #                         accelerator.wait_for_everyone()
    #                         unwrapped_model = accelerator.unwrap_model(model)
    #                         accelerator.save(unwrapped_model.state_dict(),
    #                                          config['save_path'] + f'deberta_large_{fold}.pt')
    #                         # torch.save(model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
    #             model.train()
    #         step += 1
    #         t_bar.update(1)
    # del model, train_dataset, train_loader, val_dataset, val_loader
    # gc.collect()
    # torch.cuda.empty_cache()
    # return min_valid_acc, min_step


def generate_fold_prediction(fold, config, test_A, tokenizer, t_bar):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    testA_dataset = CLRPDataset_pred(test_A, tokenizer)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
    #                                            pin_memory=True, num_workers=6)
    testA_loader = torch.utils.data.DataLoader(testA_dataset, batch_size=config["eval_batch_size"], shuffle=False,
                                             pin_memory=True, num_workers=6)
    model = Custom_bert(config['model_dir'])  # .to(device)
    _ = model.eval()

    model.load_state_dict(torch.load(config['pretrained_path'], accelerator.device), strict=False)
    model.load_state_dict(torch.load(config['save_path'] + f'roberta_large_{fold}.pt', accelerator.device), strict=True)
    model, val_loader = accelerator.prepare(model, testA_loader)
    print("Eval Model: ", )
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        total_fold_prediction = []
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids']  # to(device)
            attention_mask = batch['attention_mask']  # .to(device)
            outputs = model(input_ids, attention_mask)
            total_fold_prediction.append(outputs.cpu().numpy())
    return np.concatenate(total_fold_prediction)

def generate_test_prediction(config, output_dir):
    config["eval_batch_size"] = 256
    seed_everything(config['seed_'])
    test_A = pd.read_csv("/hhd/wt106/QyQuestion/QyData/test_A.tsv", sep="\t", header=None)
    test_A.rename(columns={0: "text_a", 1: "text_b"}, inplace=True)
    model_dir = config['model_dir']
    if config["model_type"] != "de":
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=128)


    t_bar = tqdm(total=((len(test_A) // config['batch_size']) + 1) * config['n_folds'])
    total_predictions = []
    for i in range(config['n_folds']):
        print(f"Current Deal with FoLD {i}")
        fold_np = generate_fold_prediction(i, config, test_A, tokenizer, t_bar)
        total_predictions.append(fold_np)
    last_all_prob = np.concatenate(total_predictions, axis=1)
    return last_all_prob


def post_process(last_all_prob):
    print(f"Prediction Size:{last_all_prob.shape[0]} Models: {last_all_prob.shape[1]}")
    last_all_int = (last_all_prob > 0).astype(np.int)
    last_all_mean = np.mean(last_all_int, axis=1)
    all_same = (last_all_mean == 0).sum() + (last_all_mean == 1).sum()
    print(f"ALL OOF Predictions The SAME {all_same}, ToTal {last_all_prob.shape[0]}")
    last_prediction_by_vote = (last_all_mean > 0.5).astype(np.int)
    m = nn.Sigmoid()
    last_prediction_by_prob = (np.mean(m(torch.tensor(last_all_prob)).cpu().numpy(), axis=1)>0.5).astype(np.int)
    print(f"VOTE AND PROB EQUAL {(last_prediction_by_vote==last_prediction_by_prob).sum()} TOTAL {last_all_prob.shape[0]}")
    return last_prediction_by_vote, last_prediction_by_prob

def post_process_v2(last_all_prob_long):
    test_length = pd.read_csv("/hhd/wt106/QyQuestion/QyData/test_length.csv")
    last_prediction_by_vote = pd.read_csv("/hhd/wt106/QyQuestion/Qypredictions/ccf_qianyan_qm_result_A_by_vote.csv", header=None)
    last_prediction_by_vote_long, _ = post_process(last_all_prob_long)
    small_flag = test_length<=32
    print(f"With Less 32 Length THE SAME: {(last_prediction_by_vote[small_flag.values].values.flatten()==last_prediction_by_vote_long[:, np.newaxis][small_flag.values]).sum()} ToTal {small_flag.sum()}")
    print(f"With More Than 32 Length THE SAME: {(last_prediction_by_vote[(~small_flag).values].values.flatten()==last_prediction_by_vote_long[:, np.newaxis][(~small_flag).values]).sum()} ToTal {(~small_flag).sum()}")
    # new_generate = last_prediction_by_vote
    # new_generate[~small_flag.values] = last_prediction_by_vote_long[:, np.newaxis][~small_flag.values][:, np.newaxis]
    return last_prediction_by_vote_long



def get_tokenized_length(config):
    config["eval_batch_size"] = 512
    seed_everything(config['seed_'])

    train_data = pd.read_csv("./QyData/train.csv")
    train_data = create_folds(train_data, num_splits=5)  # 用Sturge's rule并且做了StratifiedKFold
    train_data["text_a"] = train_data["text_a"].apply(lambda x: str(x))  # BAD 465202
    train_data["text_b"] = train_data["text_b"].apply(lambda x: str(x))
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)
    # t_bar = tqdm(
    #     total=((533982 * 0.2 // config['batch_size']) + 1) * config['n_folds'] / config["n_gpu"])
    all_len = []
    for i in tqdm(range(len(train_data))):
        cur_len = len(tokenizer([[train_data["text_a"].iloc[i], train_data["text_b"].iloc[i]]], add_special_tokens=True, is_split_into_words=False)["input_ids"][0])
        all_len.append(cur_len)

    test_A = pd.read_csv("/hhd/wt106/QyQuestion/QyData/test_A.tsv", sep="\t", header=None)
    test_A.rename(columns={0: "text_a", 1: "text_b"}, inplace=True)
    test_A_len = []
    for i in tqdm(range(len(test_A))):
        cur_len = len(tokenizer([[test_A["text_a"].iloc[i], test_A["text_b"].iloc[i]]], add_special_tokens=True, is_split_into_words=False)["input_ids"][0])
        test_A_len.append(cur_len)


    for i in range(config['n_folds']):
        run_fold_eval(i, config, train_data, tokenizer, t_bar)




if __name__ =="__main__":
    infer = False
    if not infer:
        # config = generate_config('ro', './Qymodels/roberta_large_pretrain.pt', './Qymodels/roberta_2/', 'custom', '1', "cls")
        config = generate_config('ro', './Qymodels/deberta_large_pretrain.pt', './Qymodels/deberta_1/', 'custom', '1', "cls")

        config["finetune_long"] = False
        eval_ft(config)
        # output_dir = "/hhd/wt106/QyQuestion/Qypredictions/"
        # last_all_prob = generate_test_prediction(config, output_dir)
        # np.save(os.path.join(output_dir, "last_all_prob_ro5_long.npy"), last_all_prob)
    else:
        output_dir = "/hhd/wt106/QyQuestion/Qypredictions/"
        last_all_prob = np.load(os.path.join(output_dir, "last_all_prob_ro5_long.npy"))
        # last_prediction_by_vote, last_prediction_by_prob = post_process(last_all_prob)
        last_prediction_by_vote_long = post_process_v2(last_all_prob)
        pd.DataFrame(last_prediction_by_vote).to_csv(os.path.join(output_dir,"ccf_qianyan_qm_result_A_by_vote.csv"), index=False, header=None)
        pd.DataFrame(last_prediction_by_prob).to_csv(os.path.join(output_dir, "ccf_qianyan_qm_result_A_by_prob.csv"), index=False, header=None)

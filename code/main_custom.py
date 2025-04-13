import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.custom import CustomDataset
import clip
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class_names = ["T-shirt", "badminton-racket", "baozi", "guitar", "lychee", "others"]

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def eval_threshold(pos_res, neg_res, threshold):
    pos_res = np.array(pos_res)
    neg_res = np.array(neg_res)
    # 计算tp,fp,fn 这里注意后面的分母是不是会出现nan或者0
    tp = sum(pos_res >= threshold)
    fp = sum(neg_res >= threshold)
    fn = sum(pos_res < threshold)

    # precision = TP / TP + FP
    precision = tp / (tp + fp)

    # recall = TP / TP + FN
    recall = tp / (tp + fn)

    # f1-score = 2 * precision * recall / (precision + recall )
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall

def find_thresholds(pos_res, neg_res, target_class, verbose=False):
    min_val = max(min(pos_res), min(neg_res))
    max_val = min(max(pos_res), max(neg_res))
    num_vals = int((max_val-min_val)*10)
    thresholds = np.linspace(min_val, max_val, num_vals)

    best_threshold = 0.
    best_f1_score = 0.
    best_precision = 0.
    best_recall = 0.
    f1_scores = []

    for threshold in thresholds:
        f1_score, precision, recall = eval_threshold(pos_res, neg_res, threshold)
        f1_scores.append(f1_score)

        # 判断最佳f1
        if f1_score > best_f1_score:
            # 更新指标
            best_threshold = threshold
            best_f1_score = f1_score
            best_precision = precision
            best_recall = recall

    if verbose:
        print(f"{target_class}_best_threshold", best_threshold)
        print(f"{target_class}_best_f1_score", best_f1_score)
        print(f"{target_class}_best_precision", best_precision)
        print(f"{target_class}_best_recall", best_recall)

        import matplotlib.pyplot as plt
        # print(thresholds)
        # print(f1_scores)
        # 绘制曲线
        plt.figure(figsize=(9, 9))
        plt.plot(thresholds, f1_scores)
        # 绘制最佳点
        plt.scatter(x=best_threshold, y=best_f1_score)
        plt.annotate(f"threshold:{best_threshold:.5f}/f1:{best_f1_score:.5f}", xy=(best_threshold, best_f1_score))
        # 添加文字信息
        plt.xlabel('threshold')
        plt.ylabel('f1_score')
        plt.title(f'{target_class}_precision:{best_precision:.4f}_recall:{best_recall:.4f}')
        plt.savefig(f'result_{target_class}_all.jpg')

        print('done')
    return best_f1_score

def get_similarity(similarity, targets, label, device="cuda"):
    with torch.no_grad():
        scores = similarity[:, label].cpu().numpy()
        targets = targets.cpu().numpy()
        # 正类掩码
        pos_mask = (targets == label) 
        # 负类掩码 
        neg_mask = (targets != label) 

        pos_res = scores[pos_mask]
        neg_res = scores[neg_mask]
        return pos_res, neg_res
            

def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels, exclude_class=5)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
    f1 = 0
    for i, class_name in enumerate(class_names):
        if class_name != "others":
            pos_res, neg_res = get_similarity(clip_logits, test_labels, i)
            f1 += find_thresholds(pos_res, neg_res, class_name, verbose=True)
    f1 = f1 / 5.
    print("\n**** Zero-shot CLIP's test F1: {:.4f}. ****\n".format(f1))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values * 10
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels, exclude_class=5)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)
    print("best_beta:", best_beta)
    print("best_alpha:", best_alpha)

    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values * 10
    tip_logits = clip_logits + cache_logits * best_alpha

    f1 = 0
    for i, class_name in enumerate(class_names):
        if class_name != "others":
            pos_res, neg_res = get_similarity(tip_logits, test_labels, i)
            f1 += find_thresholds(pos_res, neg_res, class_name, verbose=True)
    f1 = f1 / 5.
    print("**** Tip-Adapter's test F1: {:.4f}. ****\n".format(f1))


def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch']* len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_f1, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values * 10
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            #print(tip_logits)

            loss = F.cross_entropy(tip_logits, target)
            #print(loss)

            f1 = cls_f1(tip_logits, target)
            #correct_samples += acc / 100 * len(tip_logits)
            #all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, F1: {:.4f}, Loss: {:.4f}'.format(current_lr, f1 , sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values * 10
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha 
        f1 = 0
        for i, class_name in enumerate(class_names):
            if class_name != "others":
                pos_res, neg_res = get_similarity(tip_logits, test_labels, i)
                f1 += find_thresholds(pos_res, neg_res, class_name, verbose=False)
        f1 = f1 / 5.

        print("**** Tip-Adapter-F's test F1: {:.4f}. ****\n".format(f1))
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best F1 score: {best_f1:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    best_beta = beta
    best_alpha = alpha
    adapter.eval()
    affinity = adapter(test_features)

    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values * 10
    
    tip_logits = clip_logits + cache_logits * best_alpha
    f1 = 0
    for i, class_name in enumerate(class_names):
        if class_name != "others":
            pos_res, neg_res = get_similarity(tip_logits, test_labels, i)
            f1 += find_thresholds(pos_res, neg_res, class_name, verbose=True)
    f1 = f1 / 5.

    print("**** Tip-Adapter's test F1: {:.4f}. ****\n".format(f1))
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, adapter)
    print("best_beta:", best_beta)
    print("best_alpha:", best_alpha)

    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values * 10
    tip_logits = clip_logits + cache_logits * best_alpha

    f1 = 0
    for i, class_name in enumerate(class_names):
        if class_name != "others":
            pos_res, neg_res = get_similarity(tip_logits, test_labels, i)
            f1 += find_thresholds(pos_res, neg_res, class_name, verbose=True)
    f1 = f1 / 5.
    print("**** Tip-Adapter-F's test F1: {:.4f}. ****\n".format(f1))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # custom dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing Custom dataset.")
    custom = CustomDataset(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(custom.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(custom.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(custom.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")

    clip_weights = clip_classifier(custom.classnames, custom.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    #run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()
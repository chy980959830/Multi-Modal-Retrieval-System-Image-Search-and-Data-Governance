from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
from main_custom import get_similarity, eval_threshold, find_thresholds
from tqdm import tqdm
import itertools

class_names = ["T-shirt", "badminton-racket", "baozi", "guitar", "lychee", "others"]


def cls_acc(output, target, topk=1, exclude_class=None):
    # 获取topk预测结果
    pred = output.topk(topk, 1, True, True)[1].t()  # [topk, batch_size]
    # 生成正确矩阵 (判断topk预测中是否包含正确答案)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [topk, batch_size]
    
    # 生成掩码：排除指定类别
    if exclude_class is not None:
        mask = target.ne(exclude_class)  # [batch_size], True表示需要保留的样本
    else:
        mask = torch.ones_like(target, dtype=torch.bool)  # 保留所有样本
    
    # 排除无效样本
    correct = correct[:, mask]  # 从correct矩阵中过滤列 [topk, valid_samples]
    
    # 计算有效样本数
    valid_num = mask.sum().item()
    if valid_num == 0:  # 避免除以0
        return 0.0
    
    # 计算准确率（判断每个有效样本的topk预测中是否至少有一个正确）
    acc = correct.any(dim=0).float().sum().item()  # 至少有一个正确即视为正确
    acc = 100 * acc / valid_num
    
    return acc

def cls_f1(output, target):
    """
    计算所有类别的 Macro-F1（各类F1的平均值）
    
    参数:
        output: 模型输出logits，形状为 [batch_size, num_classes]
        target: 真实标签，形状为 [batch_size]
    
    返回:
        macro_f1 (float): Macro-F1 值（百分比形式）
    """
    num_classes = output.size(1)
    pred_labels = output.argmax(dim=1)  # 获取预测类别 [batch_size]
    target = target.to(pred_labels.device)  # 确保device一致
    
    # --- 计算混淆矩阵（高效向量化方法） ---
    # 将 target 和 pred_labels 编码为一维索引
    idx = target * num_classes + pred_labels
    # 用 bincount 统计每个组合出现的次数，再reshape为矩阵
    conf_matrix = torch.bincount(idx, minlength=num_classes*num_classes).view(num_classes, num_classes)
    
    # --- 计算 TP, FP, FN ---
    tp = conf_matrix.diag()               # 对角线为各类的TP
    fp = conf_matrix.sum(dim=0) - tp      # 各列和 - TP = FP
    fn = conf_matrix.sum(dim=1) - tp      # 各行和 - TP = FN
    
    # --- 计算逐类F1（避免除零）---
    epsilon = 1e-6  # 小量防止分母为0
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # --- 计算 Macro-F1 ---
    macro_f1 = f1_per_class.mean().item() * 100  # 转为百分比
    
    return macro_f1


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0) / cfg['search_step'][0] + 0.01 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0) / cfg['search_step'][1] + 0.01 for i in range(cfg['search_step'][1])]
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(beta_list, alpha_list))  

        # 单层进度条显示全局进度
        progress_bar = tqdm(param_combinations, desc="超参数搜索")  

        best_f1 = 0
        best_beta, best_alpha = 0, 0

        for beta, alpha in progress_bar:
            progress_bar.set_description(f"beta={beta}, alpha={alpha}")  # 动态更新当前参数
            # 执行你的超参数训练代码

            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values * 10
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            f1 = 0
            for i, class_name in enumerate(class_names):
                if class_name != "others":
                    pos_res, neg_res = get_similarity(tip_logits, labels, i)
                    f1 += find_thresholds(pos_res, neg_res, class_name, verbose=False)
            f1 = f1 / 5.
        
            if f1 > best_f1:
                print("New best setting, beta: {:.4f}, alpha: {:.4f}; F1: {:.4f}".format(beta, alpha, f1))
                best_f1 = f1
                best_beta = beta
                best_alpha = alpha

        print("\nAfter searching, the best F1: {:.4f}.\n".format(best_f1))
        print("\nAfter searching, the best beta: {:.4f}, the best alpha: {:.4f}.\n".format(best_beta, best_alpha))
    else:
        best_beta = cfg["init_beta"]
        best_alpha = cfg["init_alpha"]

    return best_beta, best_alpha
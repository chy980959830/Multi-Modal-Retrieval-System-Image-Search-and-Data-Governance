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
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from collections import Counter

class_names = ["T-shirt", "badminton-racket", "baozi", "guitar", "lychee", "cherry", "tennis-racket", "violin", "mantou", "dress-shirt"]
class_to_idx = {
    "T-shirt":0,
    "badminton-racket":1,
    "baozi":2,
    "guitar":3,
    "lychee":4,
    "cherry":5,
    "tennis-racket":6,
    "violin": 7,
    "mantou": 8,
    "dress-shirt": 9
}
dataset_path = "data/search"

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
    min_val = min(min(pos_res), min(neg_res))
    max_val = max(max(pos_res), max(neg_res))
    thresholds = np.linspace(min_val, max_val, 200)

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

def get_similarity(features, targets, label, ref_feature, device="cuda"):
    with torch.no_grad():
       similarity = 100. * features.cuda() @ ref_feature.t()

       scores = similarity.cpu().numpy()
       # 正类掩码
       pos_mask = (targets == label) 
       # 负类掩码 
       neg_mask = (targets != label) 
       
       pos_res = scores[pos_mask]
       neg_res = scores[neg_mask]
       return pos_res, neg_res

def get_image_text_features(clip_model, preprocess, class_embeddings, sample_images, class_name):
    images = []
    text_features = []
    class_path = os.path.join(dataset_path, class_name)
    for sample_img in sample_images:
        path = os.path.join(class_path, sample_img)
        text_features.append(class_embeddings[class_to_idx[class_name]])
        with open(path, "rb") as f:
            img = preprocess(Image.open(f).convert("RGB"))
            images.append(img)
    text_features = torch.stack(text_features, dim=0).cuda()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    images = torch.stack(images, dim=0).cuda()
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # 平均图文特征
    image_text_features = (image_features + text_features) / 2.
    image_text_features = image_text_features.mean(dim=0)
    image_features = image_features.mean(dim=0)
    # 平均图像特征
    image_features /= image_features.norm()
    return image_features, image_text_features

def build_cache(clip_model, preprocess):
    cache_features = "./caches/search/features.pkl"
    if not os.path.exists(cache_features):
        imgs = []
        targets = []
        for cls_name in class_names:
            image_paths = [cls_name + "/" + path for path in os.listdir(os.path.join(dataset_path, cls_name))]
            imgs.extend(image_paths)
            targets.extend([class_to_idx[cls_name]]*len(image_paths))
        sample_dict = {}
        with torch.no_grad():
            for img in tqdm(imgs):
                with open(os.path.join(dataset_path, img), "rb") as f:
                    loaded_img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0).cuda()
                image_features = clip_model.encode_image(loaded_img)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                sample_dict[img] = image_features.squeeze(0).cpu().numpy()
        with open(cache_features, "wb") as f:
            pickle.dump(sample_dict, f)
    else:
        # 加载现有缓存
        with open(cache_features, "rb") as f:
            sample_dict = pickle.load(f)
    return sample_dict

def construct_dataset(feature_dict, sample_images, class_name):
    test_imgs = []
    targets = []
    for cls_name in class_names:
        if cls_name != class_name:
            image_paths = [cls_name + "/" + path for path in os.listdir(os.path.join(dataset_path, cls_name))]
            test_imgs.extend(image_paths)
            targets.extend([class_to_idx[cls_name]]*len(image_paths))
        else:
            image_paths = [cls_name + "/" + path for path in os.listdir(os.path.join(dataset_path, cls_name)) if path not in sample_images]
            test_imgs.extend(image_paths)
            targets.extend([class_to_idx[cls_name]]*len(image_paths))
    targets = np.array(targets)
    test_features = np.array([feature_dict[test_img] for test_img in test_imgs])
    test_features = torch.tensor(test_features)
    return test_features, targets


def get_cluster_features(clip_model, preprocess, sample_imgs, shots, class_name):
    # 聚类
    images = []
    class_path = os.path.join(dataset_path, class_name)
    for sample_img in sample_imgs:
        path = os.path.join(class_path, sample_img)
        with open(path, "rb") as f:
            img = preprocess(Image.open(f).convert("RGB"))
            images.append(img)
    with torch.no_grad():
        images = torch.stack(images, dim=0).cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
    # 创建 K-means 模型
    kmeans = KMeans(n_clusters=2)
    
    # 训练模型
    kmeans.fit(image_features)

    # 找到样本数量最多的聚类标签
    cluster_labels = kmeans.labels_
    # 使用Counter统计各簇样本数量
    label_counts = Counter(cluster_labels)
    # 获取数量最多的簇标签
    majority_label = max(label_counts, key=label_counts.get)
    print(label_counts)
    if abs(label_counts[0] - label_counts[1]) / len(cluster_labels) < 0.2:
        # 样本分布接近，直接使用全局平均
        distances = np.linalg.norm(image_features - kmeans.cluster_centers_.mean(axis=0), axis=1)
        nearest_indices = np.argsort(distances)[:shots]
    else:
        #筛选属于主簇的样本
        majority_mask = (cluster_labels == majority_label)
        majority_features = image_features[majority_mask]
    
        # 计算主簇中样本到簇中心的距离
        majority_center = kmeans.cluster_centers_[majority_label]
        distances = np.linalg.norm(majority_features - majority_center, axis=1)
    
        # 获取原始索引（考虑被majority_mask过滤前的索引）
        original_indices = np.where(majority_mask)[0]
        # 按距离排序并取最近的shots个样本
        nearest_indices = original_indices[np.argsort(distances)[:shots]]

    cluster_features = np.mean(image_features[nearest_indices], axis=0)
    cluster_features = torch.tensor(cluster_features).cuda()
    return cluster_features

def get_text_cluster_features(clip_model, preprocess, class_embeddings, sample_imgs, shots, class_name):
    # 聚类
    images = []
    class_path = os.path.join(dataset_path, class_name)
    for sample_img in sample_imgs:
        path = os.path.join(class_path, sample_img)
        with open(path, "rb") as f:
            img = preprocess(Image.open(f).convert("RGB"))
            images.append(img)
    with torch.no_grad():
        images = torch.stack(images, dim=0).cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
    # 创建 K-means 模型
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    
    # 训练模型
    #kmeans.fit(image_features)
    from sklearn.metrics import silhouette_score

    # 轮廓系数评估聚类合理性
    for n_clusters in [2,3,4]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(image_features)
        score = silhouette_score(image_features, kmeans.labels_)
        print(f"K={n_clusters}, Silhouette Score: {score:.3f}")
    """
    # 找到样本数量最多的聚类标签
    cluster_labels = kmeans.labels_
    # 使用Counter统计各簇样本数量
    label_counts = Counter(cluster_labels)
    # 获取数量最多的簇标签
    majority_label = max(label_counts, key=label_counts.get)
    print(label_counts)
    if abs(label_counts[0] - label_counts[1]) / len(cluster_labels) < 0.2:
        # 样本分布接近，直接使用全局平均
        distances = np.linalg.norm(image_features - kmeans.cluster_centers_.mean(axis=0), axis=1)
        nearest_indices = np.argsort(distances)[:shots]
    else:
        #筛选属于主簇的样本
        majority_mask = (cluster_labels == majority_label)
        majority_features = image_features[majority_mask]
    
        # 计算主簇中样本到簇中心的距离
        majority_center = kmeans.cluster_centers_[majority_label]
        distances = np.linalg.norm(majority_features - majority_center, axis=1)
    
        # 获取原始索引（考虑被majority_mask过滤前的索引）
        original_indices = np.where(majority_mask)[0]
        # 按距离排序并取最近的shots个样本
        nearest_indices = original_indices[np.argsort(distances)[:shots]]

    nearest_indices = np.argsort(distances)[:shots]
    text_features = class_embeddings[class_to_idx[class_name]]
    text_features /= text_features.norm(dim=-1, keepdim=True)
    cluster_features = np.mean(image_features[nearest_indices], axis=0)
    cluster_features = torch.tensor(cluster_features).cuda()
    cluster_features = (cluster_features + text_features) / 2.
    return cluster_features
    """

def outlier_filter(clip_model, preprocess, sample_imgs, class_name):
    # 聚类
    images = []
    class_path = os.path.join(dataset_path, class_name)
    for sample_img in sample_imgs:
        path = os.path.join(class_path, sample_img)
        with open(path, "rb") as f:
            img = preprocess(Image.open(f).convert("RGB"))
            images.append(img)
    with torch.no_grad():
        images = torch.stack(images, dim=0).cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
     # 直接计算全局均值
    center = np.mean(image_features, axis=0)
    cos_distances = 1 - image_features @ center  # 1 - 余弦相似度
    
    # 保留距离最小的 (1 - trim_ratio) 样本
    keep_mask = cos_distances <= np.percentile(cos_distances, 95)
    robust_features = np.mean(image_features[keep_mask], axis=0)
    robust_features = torch.tensor(robust_features).cuda()

    return robust_features
    
def main():

    # custom dataset
    random.seed(1)
    torch.manual_seed(1)

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()

    img_paths = os.listdir(dataset_path)

    reference_nums = [1, 5, 10, 20]
    # 存储文本特征
    texts = clip.tokenize(class_names).cuda()
    class_embeddings = clip_model.encode_text(texts)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    feature_dict = build_cache(clip_model, preprocess)

    #for num in reference_nums:
        # 针对每个类别计算
        #for class_name in class_names:
            # 取出参考图
            #img_paths = os.listdir(os.path.join(dataset_path, class_name))
            #sample_imgs = random.sample(img_paths, num)
            #test_features, targets = construct_dataset(feature_dict, sample_imgs, class_name)
            #image_features, image_text_features = get_image_text_features(clip_model, preprocess, class_embeddings, sample_imgs, class_name)
            #pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], image_features)
            #f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
            #print("**** {:d} shot image mean features' test F1: {:.2f}. ****\n".format(num, f1*100))
            #pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], image_text_features)
            #f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
            #print("**** {:d} shot image-text mean features' test F1: {:.2f}. ****\n".format(num, f1*100))
    
    #for class_name in class_names:
        #img_paths = os.listdir(os.path.join(dataset_path, class_name))
        #sample_imgs = random.sample(img_paths, 50)
        #test_features, targets = construct_dataset(feature_dict, sample_imgs, class_name)
        #cluster_features = get_cluster_features(clip_model, preprocess, sample_imgs, 20, class_name)
        #pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], cluster_features)
        #f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
        #print("**** 10-shot cluster features' test F1: {:.2f}. ****\n".format(f1*100))

        #img_paths = os.listdir(os.path.join(dataset_path, class_name))
        #sample_imgs = random.sample(img_paths, 50)
        #test_features, targets = construct_dataset(feature_dict, sample_imgs, class_name)
        #cluster_features = get_text_cluster_features(clip_model, preprocess, class_embeddings, sample_imgs, 10, class_name)
        #get_text_cluster_features(clip_model, preprocess, class_embeddings, sample_imgs, 10, class_name)
        #pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], cluster_features)
        #f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
        #print("**** 10-shot cluster features' test F1: {:.2f}. ****\n".format(f1*100))
    """
    for class_name in class_names:
        img_paths = os.listdir(os.path.join(dataset_path, class_name))
        sample_imgs = random.sample(img_paths, 10)
        test_features, targets = construct_dataset(feature_dict, sample_imgs, class_name)
        cluster_features = outlier_filter(clip_model, preprocess, sample_imgs, class_name)
        pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], cluster_features)
        f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
        print("**** 10-shot cluster features' test F1: {:.2f}. ****\n".format(f1*100))
        """
    for class_name in class_names:
        img_paths = os.listdir(os.path.join(dataset_path, class_name))
        sample_imgs = random.sample(img_paths, 10)
        test_features, targets = construct_dataset(feature_dict, sample_imgs, class_name)
        cluster_features = outlier_filter(clip_model, preprocess, sample_imgs, class_name)
        cluster_features = (cluster_features + class_embeddings[class_to_idx[class_name]]) / 2.
        pos_res, neg_res = get_similarity(test_features, targets, class_to_idx[class_name], cluster_features)
        f1 = find_thresholds(pos_res, neg_res, class_name, verbose=True)
        print("**** 10-shot cluster features' test F1: {:.2f}. ****\n".format(f1*100))
    
    

if __name__ == '__main__':
    main()
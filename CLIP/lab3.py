import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import csv
from torch.utils.data import Dataset, DataLoader
import clip
import logging

# 设置日志
logging.basicConfig(filename="image_processing.log", level=logging.INFO)

class ImageDataset(Dataset):
    def __init__(self, image_paths, true_labels, preprocess):
        self.image_paths = image_paths
        self.true_labels = true_labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.true_labels[idx]
        try:
            with Image.open(img_path).convert("RGB") as img:
                processed_img = self.preprocess(img)
            return processed_img, label, img_path
        except Exception as e:
            logging.error(f"加载图片错误 {img_path}: {e}")
            return torch.zeros(3, 224, 224), "error", img_path

def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images), list(labels), list(paths)

def evaluate_thresholds(similarities, thresholds, positive_class, negative_class):
    """评估不同阈值下的分类性能"""
    results = []
    relevant_similarities = [item for item in similarities if item["true_label"] in [positive_class, negative_class]]
    total_pos = sum(1 for item in relevant_similarities if item["true_label"] == positive_class)
    total_neg = sum(1 for item in relevant_similarities if item["true_label"] == negative_class)

    for threshold in thresholds:
        TP = sum(1 for item in relevant_similarities if item["similarity"] >= threshold and item["true_label"] == positive_class)
        FP = sum(1 for item in relevant_similarities if item["similarity"] >= threshold and item["true_label"] == negative_class)
        FN = total_pos - TP
        TN = total_neg - FP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN
        })
    return results

def main():
    # 参数
    dataset_path = r"C:\Users\chy\Desktop\dataset"
    positive_classes = ["dog", "cat", "porcelain", "horse", "ink painting"]
    negative_classes = ["wolf", "lynx", "pottery", "donkey", "oil painting"]
    batch_size = 64
    output_dir = "results_lab3"
    thresholds = np.arange(0.0, 1.001, 0.001).round(3)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 编码所有文本
    text_features = {}
    for cls in positive_classes:
        text_desc = f"a photo of {cls}"
        text_input = clip.tokenize([text_desc]).to(device)
        with torch.no_grad():
            features = model.encode_text(text_input)
            text_features[cls] = features / features.norm(dim=1, keepdim=True)

    # 扫描数据集
    all_files = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    all_files.append((os.path.join(class_path, file), class_name))

    # 数据加载
    file_paths, true_labels = map(list, zip(*all_files))
    dataset = ImageDataset(file_paths, true_labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    # 计算相似度
    all_similarities = {cls: [] for cls in positive_classes}
    for images, labels, paths in tqdm(dataloader, desc="处理图像"):
        images = images.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            for cls in positive_classes:
                sim = (image_features @ text_features[cls].t()).squeeze().cpu().numpy()
                for s, lbl, p in zip(sim, labels, paths):
                    if lbl != "error":
                        all_similarities[cls].append({"similarity": float(s), "true_label": lbl, "file_path": p})

    # 评估与保存
    results_summary = []
    for pos_cls, neg_cls in zip(positive_classes, negative_classes):
        results = evaluate_thresholds(all_similarities[pos_cls], thresholds, pos_cls, neg_cls)
        best_result = max(results, key=lambda x: x["f1"])
        results_summary.append({
            "positive_class": pos_cls,
            "negative_class": neg_cls,
            "best_threshold": best_result["threshold"],
            "f1": best_result["f1"],
            "precision": best_result["precision"],
            "recall": best_result["recall"]
        })

        # 保存阈值扫描结果
        with open(os.path.join(output_dir, f"threshold_{pos_cls}_vs_{neg_cls}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Threshold", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"])
            for res in results:
                writer.writerow([f"{res['threshold']:.3f}", f"{res['precision']:.3f}", f"{res['recall']:.3f}", f"{res['f1']:.3f}", res["TP"], res["FP"], res["TN"], res["FN"]])

    # 计算平均值
    avg_precision = np.mean([res["precision"] for res in results_summary])
    avg_recall = np.mean([res["recall"] for res in results_summary])
    avg_f1 = np.mean([res["f1"] for res in results_summary])

    # 保存汇总结果
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Positive Class", "Negative Class", "Best Threshold", "F1", "Precision", "Recall"])
        for res in results_summary:
            writer.writerow([res["positive_class"], res["negative_class"], f"{res['best_threshold']:.3f}", f"{res['f1']:.3f}", f"{res['precision']:.3f}", f"{res['recall']:.3f}"])
        writer.writerow(["Average", "", "", f"{avg_f1:.3f}", f"{avg_precision:.3f}", f"{avg_recall:.3f}"])

    # 控制台输出
    print("\n== 分析结果汇总 ==")
    for res in results_summary:
        print(f"{res['positive_class']} vs {res['negative_class']}: 最佳阈值={res['best_threshold']:.3f}, F1={res['f1']:.3f}, Precision={res['precision']:.3f}, Recall={res['recall']:.3f}")
    print(f"\n平均 F1: {avg_f1:.3f}, 平均 Precision: {avg_precision:.3f}, 平均 Recall: {avg_recall:.3f}")
    print("分析完成!")

if __name__ == "__main__":
    main()
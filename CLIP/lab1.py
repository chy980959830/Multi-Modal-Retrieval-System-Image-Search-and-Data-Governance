import os
import clip
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import csv
from torch.utils.data import Dataset, DataLoader

class MultiClassImageDataset(Dataset):
    """用于加载和预处理多类图像的数据集类"""
    def __init__(self, image_paths, true_labels, preprocess):
        self.image_paths = image_paths
        self.true_labels = true_labels
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            true_label = self.true_labels[idx]
            with Image.open(img_path).convert("RGB") as img:
                processed_img = self.preprocess(img)
            return processed_img, true_label, img_path
        except Exception as e:
            print(f"错误加载 {img_path}: {e}")
            return torch.zeros(3, 224, 224), "error", img_path

def main():
    """主函数"""
    # 参数（直接在此处修改）
    dataset_path = r"C:\Users\chy\Desktop\dataset"  # 数据集目录路径
    positive_classes = ["dog", "cat", "porcelain", "horse", "ink painting"]  # 正类
    negative_classes = ["wolf", "lynx", "pottery", "donkey", "oil painting"]  # 负类
    batch_size = 32  # 推理批次大小
    output_dir = "results_lab1"  # 输出文件保存目录
    model_name = "ViT-B/32"  # 使用的 CLIP 模型架构
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载 CLIP 模型
    model, preprocess = clip.load(model_name, device=device)
    
    # 所有类别（正类 + "others"）
    all_classes = positive_classes + ["others"]
    
    # 编码文本描述
    text_descriptions = [f"a photo of {cls}" for cls in all_classes]
    text_inputs = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 扫描数据集
    all_files = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    label = class_name.lower() if class_name.lower() in positive_classes else "others"
                    all_files.append((os.path.join(class_path, file), label))
    
    file_paths, true_labels = map(list, zip(*all_files))
    dataset = MultiClassImageDataset(file_paths, true_labels, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 初始化统计数据
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    
    # 批次预测
    for images, labels, paths in tqdm(dataloader):
        valid_indices = [i for i, label in enumerate(labels) if label != "error"]
        if not valid_indices:
            continue
        valid_images = torch.stack([images[i] for i in valid_indices]).to(device)
        valid_labels = [labels[i] for i in valid_indices]
        
        with torch.no_grad():
            image_features = model.encode_image(valid_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred_indices = similarity.argmax(dim=1).cpu().numpy()
        
        for true_label, pred_idx in zip(valid_labels, pred_indices):
            pred_label = all_classes[pred_idx]
            for cls in positive_classes:
                if true_label == cls:
                    if pred_label == cls:
                        stats[cls]["TP"] += 1
                    else:
                        stats[cls]["FN"] += 1
                else:
                    if pred_label == cls:
                        stats[cls]["FP"] += 1
                    else:
                        stats[cls]["TN"] += 1
    
    # 计算指标
    results = []
    for cls in positive_classes:
        tp = stats[cls]["TP"]
        fp = stats[cls]["FP"]
        fn = stats[cls]["FN"]
        tn = stats[cls]["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        results.append({
            "Class": cls,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Accuracy": accuracy
        })
    
    # 计算平均指标
    avg_precision = np.mean([x["Precision"] for x in results])
    avg_recall = np.mean([x["Recall"] for x in results])
    avg_f1 = np.mean([x["F1-Score"] for x in results])
    avg_accuracy = np.mean([x["Accuracy"] for x in results])
    
    # 打印结果
    print("\n=== 评估结果 ===")
    for res in results:
        print(f"{res['Class']}: 精确率={res['Precision']:.3f}, 召回率={res['Recall']:.3f}, F1={res['F1-Score']:.3f}, 准确率={res['Accuracy']:.3f}")
    print(f"\n=== 平均指标 ===")
    print(f"平均精确率: {avg_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f}")
    print(f"平均 F1 分数: {avg_f1:.4f}")
    print(f"平均准确率: {avg_accuracy:.4f}")
    
    # 保存结果到 CSV
    with open(os.path.join(output_dir, "classification_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Accuracy"])
        for res in results:
            writer.writerow([res["Class"], f"{res['Precision']:.3f}", f"{res['Recall']:.3f}", f"{res['F1-Score']:.3f}", f"{res['Accuracy']:.3f}"])
        writer.writerow(["Average", f"{avg_precision:.3f}", f"{avg_recall:.3f}", f"{avg_f1:.3f}", f"{avg_accuracy:.3f}"])

if __name__ == "__main__":
    main()
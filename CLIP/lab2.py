import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    """用于加载和预处理图像的数据集类"""
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
    target_classes = ["dog", "cat", "porcelain", "horse", "ink painting"]  # 目标类别列表
    batch_size = 32  # 推理批次大小
    output_dir = "results_lab2"  # 输出文件保存目录
    model_name = "ViT-B/32"  # 使用的 CLIP 模型架构
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载 CLIP 模型
    model, preprocess = clip.load(model_name, device=device)
    
    # 初始化用于存储所有类别的指标
    all_precision = []
    all_recall = []
    all_f1 = []
    
    # 针对每个目标类别进行评估
    for target_class in target_classes:
        print(f"\n评估目标: {target_class}")
        
        # 编码文本描述
        text_descriptions = [f"a photo of {target_class}", f"a photo that is not {target_class}"]
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
                        true_label = "positive" if class_name.lower() == target_class.lower() else "negative"
                        all_files.append((os.path.join(class_path, file), true_label))
        
        file_paths, true_labels = map(list, zip(*all_files))
        dataset = ImageDataset(file_paths, true_labels, preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 初始化统计数据
        TP, FP, FN, TN = 0, 0, 0, 0
        
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
                pred_label = "positive" if pred_idx == 0 else "negative"
                if true_label == "positive":
                    TP += 1 if pred_label == "positive" else 0
                    FN += 1 if pred_label == "negative" else 0
                else:
                    FP += 1 if pred_label == "positive" else 0
                    TN += 1 if pred_label == "negative" else 0
        
        # 计算指标
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        # 保存结果到 CSV
        with open(os.path.join(output_dir, f"{target_class}_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["指标", "值"])
            writer.writerow(["TP", TP])
            writer.writerow(["FP", FP])
            writer.writerow(["FN", FN])
            writer.writerow(["TN", TN])
            writer.writerow(["精确率", f"{precision:.4f}"])
            writer.writerow(["召回率", f"{recall:.4f}"])
            writer.writerow(["F1 分数", f"{f1:.4f}"])
            writer.writerow(["准确率", f"{accuracy:.4f}"])
        
        print(f"{target_class} 的指标: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, 准确率={accuracy:.4f}")
        
        # 记录指标以计算平均值
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    
    # 计算并输出平均指标
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    print("\n=== 平均指标 ===")
    print(f"平均精确率: {avg_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f}")
    print(f"平均 F1 分数: {avg_f1:.4f}")
    
    # 将平均指标保存到 CSV
    with open(os.path.join(output_dir, "average_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "值"])
        writer.writerow(["平均精确率", f"{avg_precision:.4f}"])
        writer.writerow(["平均召回率", f"{avg_recall:.4f}"])
        writer.writerow(["平均 F1 分数", f"{avg_f1:.4f}"])

if __name__ == "__main__":
    main()
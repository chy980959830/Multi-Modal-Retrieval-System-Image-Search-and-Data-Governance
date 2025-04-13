import os
import csv
import shutil
import logging
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import clip
from transformers import BertTokenizer, BertForSequenceClassification, CLIPProcessor, CLIPModel

logging.basicConfig(filename="combined_image_processing.log", level=logging.INFO)


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.paths = image_paths
        self.labels = labels
        self.processor = processor
        # 当使用CLIPProcessor时，对图片处理方式不同
        self.use_clipproc = hasattr(processor, "image_processor")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        try:
            with Image.open(path).convert("RGB") as img:
                if self.use_clipproc:
                    inputs = self.processor(images=img, return_tensors="pt", padding=True)
                    return inputs["pixel_values"].squeeze(0), label, path
                else:
                    return self.processor(img), label, path
        except Exception as e:
            logging.error(f"Error loading image {path}: {e}")
            return torch.zeros(3, 224, 224), "error", path


def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), list(labels), list(paths)


def evaluate_thresholds(sim_list, thresholds, pos, neg):
    results = []
    sim_data = [item for item in sim_list if item["true_label"] in [pos, neg]]
    tot_pos = sum(1 for item in sim_data if item["true_label"] == pos)
    tot_neg = sum(1 for item in sim_data if item["true_label"] == neg)
    for thresh in thresholds:
        TP = sum(1 for item in sim_data if item["similarity"] >= thresh and item["true_label"] == pos)
        FP = sum(1 for item in sim_data if item["similarity"] >= thresh and item["true_label"] == neg)
        FN = tot_pos - TP
        TN = tot_neg - FP
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results.append({"threshold": thresh, "precision": prec, "recall": rec, "f1": f1,
                        "TP": TP, "FP": FP, "TN": TN, "FN": FN})
    return results


def save_correct_samples(en_sims, cn_sims, en_threshs, cn_threshs, en_pos, cn_pos,
                         en_dataset_path, cn_dataset_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_log.txt")
    saved_counts = {}
    with open(log_path, "w", encoding="utf-8") as log:
        for i, en_cls in enumerate(en_pos):
            cn_cls = cn_pos[i]
            en_thresh, cn_thresh = en_threshs[i], cn_threshs[i]
            log.write(f"\n===== {en_cls} ({cn_cls}) | en_thresh: {en_thresh}, cn_thresh: {cn_thresh}\n")
            class_dir = os.path.join(out_dir, en_cls)
            if os.path.exists(class_dir):
                shutil.rmtree(class_dir)
            os.makedirs(class_dir)
            en_class_samples = [item for item in en_sims[en_cls] if item["true_label"] == en_cls]
            cn_class_samples = [item for item in cn_sims[cn_cls] if item["true_label"] == cn_cls]
            log.write(f"EN samples: {len(en_class_samples)}; CN samples: {len(cn_class_samples)}\n")
            en_correct = {os.path.basename(item["file_path"]) for item in en_class_samples if item["similarity"] >= en_thresh}
            cn_correct = {os.path.basename(item["file_path"]) for item in cn_class_samples if item["similarity"] >= cn_thresh}
            union_files = en_correct.union(cn_correct)
            log.write(f"EN correct: {len(en_correct)}; CN correct: {len(cn_correct)}; Union: {len(union_files)}\n")
            file_map, details = {}, {}
            for item in en_class_samples:
                bn = os.path.basename(item["file_path"])
                if bn in union_files:
                    file_map[bn] = item["file_path"]
                    details.setdefault(bn, {})["en_sim"] = item["similarity"]
            for item in cn_class_samples:
                bn = os.path.basename(item["file_path"])
                if bn in union_files:
                    file_map.setdefault(bn, item["file_path"])
                    details.setdefault(bn, {})["cn_sim"] = item["similarity"]
            copied = []
            with open(os.path.join(class_dir, "_file_details.csv"), "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["文件名", "英文相似度", "中文相似度", "英文通过", "中文通过", "源路径"])
                for bn, path in file_map.items():
                    try:
                        shutil.copy(path, os.path.join(class_dir, bn))
                        copied.append(bn)
                    except Exception as e:
                        log.write(f"Copy error {bn}: {e}\n")
                        continue
                    en_sim = details.get(bn, {}).get("en_sim")
                    cn_sim = details.get(bn, {}).get("cn_sim")
                    en_pass = "是" if en_sim is not None and en_sim >= en_thresh else "否"
                    cn_pass = "是" if cn_sim is not None and cn_sim >= cn_thresh else "否"
                    writer.writerow([bn,
                                     f"{en_sim:.4f}" if en_sim is not None else "N/A",
                                     f"{cn_sim:.4f}" if cn_sim is not None else "N/A",
                                     en_pass, cn_pass, path])
            log.write(f"Copied {len(copied)} files\n")
            saved_counts[en_cls] = {
                "english_class": en_cls,
                "chinese_class": cn_cls,
                "en_total_samples": len(en_class_samples),
                "cn_total_samples": len(cn_class_samples),
                "en_correct": len(en_correct),
                "cn_correct": len(cn_correct),
                "both": len(en_correct.intersection(cn_correct)),
                "union_total": len(union_files),
                "copied": len(copied)
            }
            with open(os.path.join(class_dir, "_copied_files.txt"), "w", encoding="utf-8") as f:
                f.write(f"{en_cls} ({cn_cls})\nUnion: {len(union_files)}; Copied: {len(copied)}\n")
                f.write("\n".join(sorted(copied)))
    return saved_counts


def calc_combined_metrics(en_sims, cn_sims, en_threshs, cn_threshs, en_pos, en_neg, cn_pos, cn_neg):
    results = []
    
    for i, en_pos_cls in enumerate(en_pos):
        cn_pos_cls = cn_pos[i]
        en_neg_cls = en_neg[i]
        cn_neg_cls = cn_neg[i]
        en_thresh, cn_thresh = en_threshs[i], cn_threshs[i]
        
        # 清晰地区分正样本和负样本的记录
        unique_pos_samples = {}  # basename -> {en_detected_as_pos: bool, cn_detected_as_pos: bool}
        unique_neg_samples = {}  # basename -> {en_misclassified_as_pos: bool, cn_misclassified_as_pos: bool}
        
        # 打印调试信息
        print(f"\nDebug for {en_pos_cls} vs {en_neg_cls}:")
        print(f"En samples in {en_pos_cls}: {len([x for x in en_sims[en_pos_cls] if x['true_label'] == en_pos_cls])}")
        print(f"En samples in {en_neg_cls}: {len([x for x in en_sims[en_pos_cls] if x['true_label'] == en_neg_cls])}")
        
        # 处理英文正样本
        for item in en_sims[en_pos_cls]:
            if item["true_label"] == en_pos_cls:
                basename = os.path.basename(item["file_path"])
                unique_pos_samples[basename] = {
                    "en_detected_as_pos": item["similarity"] >= en_thresh,
                    "cn_detected_as_pos": False  # 默认值
                }
        
        # 处理英文负样本
        for item in en_sims[en_pos_cls]:
            if item["true_label"] == en_neg_cls:
                basename = os.path.basename(item["file_path"])
                unique_neg_samples[basename] = {
                    "en_misclassified_as_pos": item["similarity"] >= en_thresh,  # 假阳性
                    "cn_misclassified_as_pos": False  # 默认值
                }
        
        # 处理中文正样本
        for item in cn_sims[cn_pos_cls]:
            basename = os.path.basename(item["file_path"])
            if item["true_label"] == cn_pos_cls:
                if basename in unique_pos_samples:
                    unique_pos_samples[basename]["cn_detected_as_pos"] = item["similarity"] >= cn_thresh
                else:
                    unique_pos_samples[basename] = {
                        "en_detected_as_pos": False,
                        "cn_detected_as_pos": item["similarity"] >= cn_thresh
                    }
        
        # 处理中文负样本
        for item in cn_sims[cn_pos_cls]:
            basename = os.path.basename(item["file_path"])
            if item["true_label"] == cn_neg_cls:
                if basename in unique_neg_samples:
                    unique_neg_samples[basename]["cn_misclassified_as_pos"] = item["similarity"] >= cn_thresh
                else:
                    unique_neg_samples[basename] = {
                        "en_misclassified_as_pos": False,
                        "cn_misclassified_as_pos": item["similarity"] >= cn_thresh  # 假阳性
                    }
        
        # 计算联合指标
        tp = sum(1 for data in unique_pos_samples.values() if data["en_detected_as_pos"] or data["cn_detected_as_pos"])
        fp = sum(1 for data in unique_neg_samples.values() if data["en_misclassified_as_pos"] or data["cn_misclassified_as_pos"])
        
        total_pos = len(unique_pos_samples)
        total_neg = len(unique_neg_samples)
        fn = total_pos - tp
        
        # 计算指标
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        # 打印调试信息
        print(f"Unique positive samples: {total_pos}")
        print(f"Unique negative samples: {total_neg}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        
        results.append({
            "en_positive_class": en_pos_cls,
            "en_negative_class": en_neg_cls,
            "cn_positive_class": cn_pos_cls, 
            "cn_negative_class": cn_neg_cls,
            "en_threshold": en_thresh,
            "cn_threshold": cn_thresh,
            "combined_f1": f1,
            "combined_precision": prec,
            "combined_recall": rec,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "total_unique_pos": total_pos,
            "total_unique_neg": total_neg
        })
    
    return results


def scan_dataset(dataset_path):
    files = []
    for cls in os.listdir(dataset_path):
        cls_path = os.path.join(dataset_path, cls)
        if os.path.isdir(cls_path):
            for f in os.listdir(cls_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    files.append((os.path.join(cls_path, f), cls))
    if files:
        return list(zip(*files))
    return ([], [])


def process_images(dataloader, model, text_features, pos_classes, device, model_type="en"):
    all_sims = {cls: [] for cls in pos_classes}
    for imgs, labels, paths in tqdm(dataloader, desc=f"Processing {model_type} images"):
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = (model.encode_image(imgs) if model_type == "en"
                     else model.get_image_features(pixel_values=imgs))
            feats = feats / feats.norm(dim=1, keepdim=True)
            for cls in pos_classes:
                sims = (feats @ text_features[cls].t()).squeeze().cpu().numpy()
                for s, lab, p in zip(sims, labels, paths):
                    if lab != "error":
                        all_sims[cls].append({"similarity": float(s), "true_label": lab, "file_path": p})
    return all_sims


def main():
    # 参数配置
    en_dataset_path = r"C:\Users\chy\Desktop\dataset"
    cn_dataset_path = r"C:\Users\chy\Desktop\dataset Chinese"
    en_pos = ["dog", "cat", "porcelain", "horse", "ink painting"]
    en_neg = ["wolf", "lynx", "pottery", "donkey", "oil painting"]
    cn_pos = ["狗", "猫", "瓷器", "马", "水墨画"]
    cn_neg = ["狼", "猞猁", "陶器", "驴", "油画"]
    batch_size = 32
    en_out, cn_out, comb_out, corr_dir = "results_lab3", "results_CN", "results_combined", "union_samples"
    thresholds = np.around(np.arange(0, 1.001, 0.001), 3)

    # 清理/创建输出目录
    for d in [en_out, cn_out, comb_out, corr_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载英文模型及文本特征
    print("Loading English CLIP model...")
    en_model, en_preprocess = clip.load("ViT-B/32", device=device)
    en_text_features = {}
    for cls in en_pos:
        text = f"a photo of {cls}"
        text_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            feat = en_model.encode_text(text_input)
            en_text_features[cls] = feat / feat.norm(dim=1, keepdim=True)

    # 英文数据集
    en_file_paths, en_labels = scan_dataset(en_dataset_path)
    en_dataset = ImageDataset(en_file_paths, en_labels, en_preprocess)
    en_loader = DataLoader(en_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)
    en_sims = process_images(en_loader, en_model, en_text_features, en_pos, device, model_type="en")

    # 加载中文模型及文本特征
    print("Loading Chinese models...")
    cn_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    cn_text_encoder = BertForSequenceClassification.from_pretrained(
        "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().to(device)
    cn_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    cn_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cn_text_features = {}
    for cls in cn_pos:
        text = f"一张{cls}的图片"
        inputs = cn_tokenizer([text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = cn_text_encoder(**inputs).logits
            cn_text_features[cls] = feat / feat.norm(dim=1, keepdim=True)

    cn_file_paths, cn_labels = scan_dataset(cn_dataset_path)
    cn_dataset = ImageDataset(cn_file_paths, cn_labels, cn_processor)
    cn_loader = DataLoader(cn_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)
    cn_sims = process_images(cn_loader, cn_clip_model, cn_text_features, cn_pos, device, model_type="cn")

    # 评估英文模型阈值
    print("Evaluating thresholds for English model...")
    en_res_summary, en_best_thresholds = [], []
    for pos, neg in zip(en_pos, en_neg):
        res = evaluate_thresholds(en_sims[pos], thresholds, pos, neg)
        best = max(res, key=lambda x: x["f1"])
        en_res_summary.append({"positive_class": pos, "negative_class": neg,
                               "best_threshold": best["threshold"],
                               "f1": best["f1"],
                               "precision": best["precision"],
                               "recall": best["recall"]})
        en_best_thresholds.append(best["threshold"])
        with open(os.path.join(en_out, f"threshold_{pos}_vs_{neg}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Threshold", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"])
            for r in res:
                writer.writerow([f"{r['threshold']:.3f}", f"{r['precision']:.3f}",
                                 f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                                 r["TP"], r["FP"], r["TN"], r["FN"]])

    # 评估中文模型阈值
    print("Evaluating thresholds for Chinese model...")
    cn_res_summary, cn_best_thresholds = [], []
    for pos, neg in zip(cn_pos, cn_neg):
        res = evaluate_thresholds(cn_sims[pos], thresholds, pos, neg)
        best = max(res, key=lambda x: x["f1"])
        cn_res_summary.append({"positive_class": pos, "negative_class": neg,
                               "best_threshold": best["threshold"],
                               "f1": best["f1"],
                               "precision": best["precision"],
                               "recall": best["recall"]})
        cn_best_thresholds.append(best["threshold"])
        with open(os.path.join(cn_out, f"threshold_{pos}_vs_{neg}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Threshold", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"])
            for r in res:
                writer.writerow([f"{r['threshold']:.3f}", f"{r['precision']:.3f}",
                                 f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                                 r["TP"], r["FP"], r["TN"], r["FN"]])

    # 保存两个模型下正确预测的样本
    print("Saving correctly predicted samples...")
    saved_counts = save_correct_samples(en_sims, cn_sims, en_best_thresholds, cn_best_thresholds,
                                        en_pos, cn_pos, en_dataset_path, cn_dataset_path, corr_dir)
    with open(os.path.join(corr_dir, "sample_counts.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["English Class", "Chinese Class", "EN Total Samples", "CN Total Samples",
                         "EN Correct", "CN Correct", "Both", "Union Total", "Copied"])
        for cls, count in saved_counts.items():
            writer.writerow([cls, count["chinese_class"], count["en_total_samples"], count["cn_total_samples"],
                             count["en_correct"], count["cn_correct"], count["both"], count["union_total"], count["copied"]])

    # 计算联合指标
    print("Calculating combined metrics...")
    combined_results = calc_combined_metrics(en_sims, cn_sims, en_best_thresholds, cn_best_thresholds,
                                             en_pos, en_neg, cn_pos, cn_neg)
    avg_prec = np.mean([r["combined_precision"] for r in combined_results])
    avg_rec = np.mean([r["combined_recall"] for r in combined_results])
    avg_f1 = np.mean([r["combined_f1"] for r in combined_results])
    with open(os.path.join(comb_out, "combined_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EN Positive", "EN Negative", "CN Positive", "CN Negative",
                         "EN Threshold", "CN Threshold", "F1", "Precision", "Recall", "TP", "FP", "FN"])
        for r in combined_results:
            writer.writerow([r["en_positive_class"], r["en_negative_class"],
                             r["cn_positive_class"], r["cn_negative_class"],
                             f"{r['en_threshold']:.3f}", f"{r['cn_threshold']:.3f}",
                             f"{r['combined_f1']:.3f}", f"{r['combined_precision']:.3f}",
                             f"{r['combined_recall']:.3f}", r["TP"], r["FP"], r["FN"]])
        writer.writerow(["Average", "", "", "", "", "",
                         f"{avg_f1:.3f}", f"{avg_prec:.3f}", f"{avg_rec:.3f}", "", "", ""])

    # 输出英文模型结果及平均指标
    print("\n== English Model Results ==")
    for res in en_res_summary:
        print(f"{res['positive_class']} vs {res['negative_class']}: "
              f"Threshold={res['best_threshold']:.3f}, F1={res['f1']:.3f}, "
              f"Precision={res['precision']:.3f}, Recall={res['recall']:.3f}")
    avg_en_prec = np.mean([res["precision"] for res in en_res_summary])
    avg_en_rec = np.mean([res["recall"] for res in en_res_summary])
    avg_en_f1 = np.mean([res["f1"] for res in en_res_summary])
    print(f"English Model Averages: Precision={avg_en_prec:.3f}, Recall={avg_en_rec:.3f}, F1={avg_en_f1:.3f}")

    # 输出中文模型结果及平均指标
    print("\n== Chinese Model Results ==")
    for res in cn_res_summary:
        print(f"{res['positive_class']} vs {res['negative_class']}: "
              f"Threshold={res['best_threshold']:.3f}, F1={res['f1']:.3f}, "
              f"Precision={res['precision']:.3f}, Recall={res['recall']:.3f}")
    avg_cn_prec = np.mean([res["precision"] for res in cn_res_summary])
    avg_cn_rec = np.mean([res["recall"] for res in cn_res_summary])
    avg_cn_f1 = np.mean([res["f1"] for res in cn_res_summary])
    print(f"Chinese Model Averages: Precision={avg_cn_prec:.3f}, Recall={avg_cn_rec:.3f}, F1={avg_cn_f1:.3f}")

    # 输出联合模型结果及平均指标
    print("\n== Combined Model Results ==")
    for r in combined_results:
        print(f"{r['en_positive_class']} vs {r['en_negative_class']}: "
              f"F1={r['combined_f1']:.3f}, Precision={r['combined_precision']:.3f}, "
              f"Recall={r['combined_recall']:.3f}")
    print(f"\nAverage Combined: F1={avg_f1:.3f}, Precision={avg_prec:.3f}, Recall={avg_rec:.3f}")
    print(f"Correct predicted samples saved in {corr_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
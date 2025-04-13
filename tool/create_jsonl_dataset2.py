import os
import json
import uuid
import random
from pathlib import Path

def create_llava_dataset(root_folder):
    # 设置随机种子以确保结果可重复
    random.seed(42)
    
    # 定义类别
    categories = ["cat", "dog", "horse", "ink painting", "porcelain"]
    
    # 初始化数据集列表
    dataset = []
    
    # 按类别存储图片
    category_images = {}
    pos_samples_count = 0
    
    # 首先收集所有类别的图片
    print("收集所有类别文件夹中的图片...")
    for category in categories:
        category_path = Path(root_folder) / category
        
        # 确保文件夹存在
        if not category_path.exists() or not category_path.is_dir():
            print(f"警告: 类别文件夹 {category} 未找到")
            continue
        
        # 获取类别中的所有图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(category_path.glob(f"*{ext}")))
            image_files.extend(list(category_path.glob(f"*{ext.upper()}")))
        
        # 移除重复项
        unique_files = set()
        for img in image_files:
            unique_files.add(str(img))
        
        image_files = [Path(p) for p in unique_files]
        category_images[category] = image_files
        
        print(f"在类别 {category} 中找到 {len(image_files)} 张图片")
        pos_samples_count += len(image_files)
    
    print(f"所有类别共找到 {pos_samples_count} 张图片")
    
    # 步骤1：创建所有正样本
    print("\n创建正样本...")
    for category, images in category_images.items():
        for img_path in images:
            # 获取相对路径
            relative_path = img_path.relative_to(root_folder)
            img_rel_path = str(relative_path).replace("\\", "/")
            
            # 创建正样本
            positive_sample = {
                "id": str(uuid.uuid4()),
                "image": img_rel_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Does this image contain a {category}?"
                    },
                    {
                        "from": "gpt",
                        "value": "Yes"
                    }
                ]
            }
            
            dataset.append(positive_sample)
    
    # 步骤2：创建负样本，确保每个类别有相同数量的负样本，且平均来自其他类别
    print("\n创建负样本...")
    neg_samples_created = 0
    
    # 跟踪所有已使用的图片-类别对
    used_combinations = set()
    
    # 为每个类别创建负样本
    for target_category in categories:
        target_neg_count = len(category_images[target_category])  # 这个类别需要的负样本数量
        neg_per_source = target_neg_count // (len(categories) - 1)  # 每个源类别需要提供的图片数量
        extra_negs = target_neg_count % (len(categories) - 1)  # 额外需要的图片数量
        
        print(f"\n为类别 {target_category} 创建 {target_neg_count} 个负样本:")
        
        # 从每个其他类别中选择图片
        source_index = 0
        for source_category in categories:
            if source_category == target_category:
                continue
            
            # 计算这个源类别需要提供的图片数量
            source_neg_count = neg_per_source
            if source_index < extra_negs:
                source_neg_count += 1
            
            print(f"  从 {source_category} 选择 {source_neg_count} 张图片")
            
            # 获取源类别的所有图片
            source_images = category_images[source_category]
            
            # 随机打乱图片顺序
            shuffled_images = source_images.copy()
            random.shuffle(shuffled_images)
            
            # 跟踪这个源类别已经使用的图片
            used_from_source = 0
            
            # 尝试不重复地使用图片
            for img_path in shuffled_images:
                # 检查这个图片-类别对是否已经使用过
                combination = (str(img_path), target_category)
                if combination in used_combinations:
                    continue
                
                # 标记这个组合已使用
                used_combinations.add(combination)
                
                # 获取相对路径
                relative_path = img_path.relative_to(root_folder)
                img_rel_path = str(relative_path).replace("\\", "/")
                
                # 创建负样本
                negative_sample = {
                    "id": str(uuid.uuid4()),
                    "image": img_rel_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Does this image contain a {target_category}?"
                        },
                        {
                            "from": "gpt",
                            "value": "No"
                        }
                    ]
                }
                
                dataset.append(negative_sample)
                used_from_source += 1
                neg_samples_created += 1
                
                # 如果已经从这个源类别选择了足够的图片，退出循环
                if used_from_source >= source_neg_count:
                    break
            
            # 如果图片不足，可能需要重复使用图片
            if used_from_source < source_neg_count:
                print(f"    警告: {source_category} 类别的图片不足，只能提供 {used_from_source}/{source_neg_count} 张不重复图片")
                
                # 重新遍历所有图片，允许重复
                remaining_needed = source_neg_count - used_from_source
                for _ in range(remaining_needed):
                    # 随机选择一张图片
                    if not source_images:
                        break
                        
                    img_path = random.choice(source_images)
                    
                    # 获取相对路径
                    relative_path = img_path.relative_to(root_folder)
                    img_rel_path = str(relative_path).replace("\\", "/")
                    
                    # 创建负样本
                    negative_sample = {
                        "id": str(uuid.uuid4()),
                        "image": img_rel_path,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"Does this image contain a {target_category}?"
                            },
                            {
                                "from": "gpt",
                                "value": "No"
                            }
                        ]
                    }
                    
                    dataset.append(negative_sample)
                    neg_samples_created += 1
            
            source_index += 1
    
    # 保存JSON文件
    output_path = Path(root_folder) / "llava_dataset2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据集已保存至: {output_path}")
    print(f"总生成记录数: {len(dataset)}")
    print(f"正样本数: {pos_samples_count}")
    print(f"负样本数: {neg_samples_created}")
    
    # 计算重复使用的图片数量
    used_image_counts = {}
    for sample in dataset:
        if sample["conversations"][1]["value"] == "No":
            img_path = sample["image"]
            target_cat = sample["conversations"][0]["value"].split("contain a ")[1].rstrip("?")
            key = (img_path, target_cat)
            used_image_counts[key] = used_image_counts.get(key, 0) + 1
    
    repeated_combinations = [k for k, v in used_image_counts.items() if v > 1]
    print(f"重复使用的图片-类别组合数: {len(repeated_combinations)}")
    
    # 统计每个类别的负样本分布
    neg_by_target = {}
    neg_by_source_target = {}
    
    for sample in dataset:
        if sample["conversations"][1]["value"] == "No":
            target_cat = sample["conversations"][0]["value"].split("contain a ")[1].rstrip("?")
            img_path = sample["image"]
            source_cat = None
            
            # 判断图片所属的源类别
            for cat, imgs in category_images.items():
                if any(str(img_path).endswith(str(img.relative_to(root_folder)).replace("\\", "/")) for img in imgs):
                    source_cat = cat
                    break
            
            if source_cat:
                # 更新目标类别统计
                neg_by_target[target_cat] = neg_by_target.get(target_cat, 0) + 1
                
                # 更新源-目标类别对统计
                pair = (source_cat, target_cat)
                neg_by_source_target[pair] = neg_by_source_target.get(pair, 0) + 1
    
    print("\n每个类别的负样本数量:")
    for cat, count in neg_by_target.items():
        print(f"  {cat}: {count}")
    
    print("\n每个源-目标类别对的负样本数量:")
    for (source, target), count in sorted(neg_by_source_target.items()):
        print(f"  {source} -> {target}: {count}")
    
    return dataset

if __name__ == "__main__":
    folder_path = input("输入数据集根目录: ")
    if not folder_path:
        folder_path = r"C:\Users\chy\Desktop\llava_dataset2"
    create_llava_dataset(folder_path)
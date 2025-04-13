import os
import json
import uuid
from pathlib import Path

def create_llava_dataset(root_folder):
    # 定义正样本类别和负样本映射关系
    positive_categories = ["cat", "dog", "horse", "ink painting", "porcelain"]
    negative_to_positive_map = {
        "lynx": "cat",
        "wolf": "dog",
        "donkey": "horse",
        "oil painting": "ink painting",
        "pottery": "porcelain"
    }
    
    # 初始化数据集列表
    dataset = []
    
    # 获取所有类别文件夹
    root_path = Path(root_folder)
    all_categories = [folder.name for folder in root_path.iterdir() 
                     if folder.is_dir() and not folder.name.startswith('.')]
    
    print(f"使用的根目录: {root_folder}")
    print(f"发现以下类别: {all_categories}")
    total_image_count = 0
    
    # 处理每个类别文件夹
    for category in all_categories:
        category_path = root_path / category
        
        # 获取该类别中的所有图片（确保不重复计算）
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = set()  # 使用集合避免重复
        
        for ext in image_extensions:
            # 确保大小写扩展名不被重复计算
            files_lower = {str(f).lower() for f in category_path.glob(f"*{ext}")}
            files_upper = {str(f).lower() for f in category_path.glob(f"*{ext.upper()}")}
            image_files.update(files_lower)
            image_files.update(files_upper)
        
        # 将集合转换回文件路径列表
        image_files = [Path(f) for f in image_files]
        
        actual_count = len(image_files)
        total_image_count += actual_count
        print(f"在类别 {category} 中实际发现 {actual_count} 张图片")
        
        # 为每张图片创建样本
        for img_path in image_files:
            # 获取相对路径
            try:
                relative_path = img_path.relative_to(root_folder)
            except ValueError:
                # 如果路径转换有问题，尝试不同的方法
                relative_path = Path(str(img_path).replace(str(root_folder) + os.sep, ""))
            
            img_path_str = str(relative_path).replace("\\", "/")
            
            # 根据类别创建问答对
            if category in positive_categories:
                # 正样本：询问正确的类别，回答"Yes"
                sample = {
                    "id": str(uuid.uuid4()),
                    "image": img_path_str,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Is this image of {category}? Answer with ONLY a single word: 'yes' or 'no'.?"
                        },
                        {
                            "from": "gpt",
                            "value": "Yes"
                        }
                    ]
                }
                dataset.append(sample)
            else:
                # 负样本：询问对应的正样本类别，回答"No"
                if category in negative_to_positive_map:
                    corresponding_positive = negative_to_positive_map[category]
                    sample = {
                        "id": str(uuid.uuid4()),
                        "image": img_path_str,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"Is this image of {corresponding_positive}? Answer with ONLY a single word: 'yes' or 'no'.?"
                            },
                            {
                                "from": "gpt",
                                "value": "No"
                            }
                        ]
                    }
                    dataset.append(sample)
                else:
                    print(f"警告：找不到类别 {category} 对应的正样本类别")
    
    # 保存JSON数组格式
    output_path = Path(root_folder) / "llava_dataset5.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存至: {output_path}")
    print(f"实际图片总数: {total_image_count}")
    print(f"共生成记录数: {len(dataset)}")
    
    return dataset

# 使用示例
if __name__ == "__main__":
    folder_path = r"C:\Users\chy\Desktop\dataset"  # 使用默认路径
    create_llava_dataset(folder_path)
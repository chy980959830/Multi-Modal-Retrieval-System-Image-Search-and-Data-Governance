import os
import hashlib
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict

def calculate_perceptual_hash(image_path, hash_size=8):
    """
    计算图像的感知哈希，对轻微修改更加鲁棒
    """
    try:
        with Image.open(image_path) as img:
            # 计算几种不同的感知哈希
            phash = imagehash.phash(img, hash_size=hash_size)
            dhash = imagehash.dhash(img, hash_size=hash_size)
            whash = imagehash.whash(img, hash_size=hash_size)
            # 返回三种哈希的组合
            return (str(phash), str(dhash), str(whash))
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return None

def get_all_images(folder_path):
    """
    递归查找文件夹及其子文件夹中的所有图像文件
    """
    image_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, filename))
    
    return image_files

def compare_hashes(hash1, hash2, threshold=5):
    """
    比较两个感知哈希的相似度
    返回True如果它们足够相似
    """
    if not hash1 or not hash2:
        return False
    
    # 计算每种哈希的汉明距离
    phash_distance = imagehash.hex_to_hash(hash1[0]) - imagehash.hex_to_hash(hash2[0])
    dhash_distance = imagehash.hex_to_hash(hash1[1]) - imagehash.hex_to_hash(hash2[1])
    whash_distance = imagehash.hex_to_hash(hash1[2]) - imagehash.hex_to_hash(hash2[2])
    
    # 如果任何两种哈希足够接近，认为是相似图像
    return (phash_distance <= threshold or 
            dhash_distance <= threshold or 
            whash_distance <= threshold)

def find_and_remove_duplicate_images(folder_path, similarity_threshold=5):
    """
    在同一个文件夹中查找并删除重复图像
    保留每个图像的第一次出现并删除重复项
    使用感知哈希来识别相似图像
    """
    # 获取文件夹中的所有图像
    all_images = get_all_images(folder_path)
    
    print(f"在文件夹中找到 {len(all_images)} 个图像")
    
    # 存储图像哈希
    reference_hashes = []
    reference_images = []
    duplicate_images = []
    
    # 按文件大小排序，通常保留最大的文件（可能质量更好）
    all_images.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    # 第一步：计算哈希并识别重复项
    for img_path in all_images:
        img_hash = calculate_perceptual_hash(img_path)
        if img_hash:
            # 检查是否与已知图像相似
            is_duplicate = False
            duplicate_of = None
            
            for i, ref_hash in enumerate(reference_hashes):
                if compare_hashes(img_hash, ref_hash, similarity_threshold):
                    is_duplicate = True
                    duplicate_of = reference_images[i]
                    break
            
            if not is_duplicate:
                # 这是这种图像的第一次出现
                reference_hashes.append(img_hash)
                reference_images.append(img_path)
            else:
                # 这是重复的
                duplicate_images.append((img_path, duplicate_of))
    
    # 第二步：删除重复项
    deleted_files = []
    for duplicate_path, original_path in duplicate_images:
        try:
            os.remove(duplicate_path)
            deleted_files.append((duplicate_path, original_path))
        except Exception as e:
            print(f"删除 {duplicate_path} 失败: {e}")
    
    return deleted_files, reference_images, len(all_images)

def main():
    # 设置你的文件夹
    folder_path = r"C:\Users\chy\Desktop\llava_dataset4\dog_negative"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return
    
    print(f"扫描重复图像...")
    print(f"文件夹: {folder_path}")
    
    # 设置相似度阈值 (0-10, 较低的值表示更严格的匹配)
    similarity_threshold = 5
    print(f"使用相似度阈值: {similarity_threshold}")
    
    deleted_files, unique_files, total_images = find_and_remove_duplicate_images(
        folder_path, similarity_threshold
    )
    
    # 打印摘要
    print(f"\n分析和清理完成!")
    print(f"文件夹中的图像总数: {total_images}")
    print(f"删除的重复图像: {len(deleted_files)}")
    print(f"保留的唯一图像: {len(unique_files)}")
    
    # 打印有关删除的内容的详细信息
    if deleted_files:
        print("\n删除了以下重复文件:")
        for delete_file, original_file in deleted_files:
            print(f"  - {delete_file} (重复于 {original_file})")
    else:
        print("\n未找到重复图像。没有删除任何内容。")

if __name__ == "__main__":
    main()
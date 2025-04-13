import os
import sys
from PIL import Image
import imagehash
from collections import defaultdict
import warnings

# 忽略PIL库的警告信息
warnings.filterwarnings("ignore", category=UserWarning)

def detect_and_remove_cross_set_duplicates(test_dir, train_dir, hash_size=8, similarity_threshold=0):
    """
    检测训练集中是否包含测试集中的图片，并删除这些重复图片
    
    参数:
        test_dir (str): 测试集目录
        train_dir (str): 训练集目录
        hash_size (int): 哈希大小，越大越精确但计算越慢
        similarity_threshold (int): 哈希差异阈值，小于此值视为相似图片
    """
    # 检查目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误: 测试集目录 '{test_dir}' 不存在。")
        return
    
    if not os.path.exists(train_dir):
        print(f"错误: 训练集目录 '{train_dir}' 不存在。")
        return
    
    print(f"开始比较:")
    print(f"测试集: {test_dir}")
    print(f"训练集: {train_dir}")
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # 存储测试集图片的哈希值和路径
    test_hashes = {}
    error_files = []
    
    # 第一步：计算测试集所有图片的哈希值
    print("\n计算测试集图片哈希值...")
    test_image_count = 0
    
    for root, _, files in os.walk(test_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # 跳过非图片文件
            if file_ext not in image_extensions:
                continue
            
            test_image_count += 1
            
            # 计算图片的感知哈希值
            try:
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    print(f"警告: 测试集文件为空 {file_path}")
                    error_files.append(file_path)
                    continue
                
                with Image.open(file_path) as img:
                    try:
                        # 将图片转换为RGB模式以处理各种图片格式
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 尝试加载图片数据，检测是否损坏
                        img.load()
                        
                        # 计算差异哈希值（对细微差异更敏感）
                        hash_value = imagehash.dhash(img, hash_size=hash_size)
                        
                        # 存储哈希值和文件路径
                        test_hashes[hash_value] = file_path
                    except Exception as e:
                        print(f"警告: 测试集图片文件损坏 {file_path}: {e}")
                        error_files.append(file_path)
            except Exception as e:
                print(f"无法处理测试集文件 {file_path}: {e}")
                error_files.append(file_path)
    
    print(f"测试集处理完成，共 {test_image_count} 个图片，{len(error_files)} 个错误文件")
    
    # 第二步：遍历训练集图片，查找与测试集相同的图片
    print("\n开始在训练集中查找重复图片...")
    duplicates_found = 0
    deleted_files = 0
    train_image_count = 0
    
    for root, _, files in os.walk(train_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # 跳过非图片文件
            if file_ext not in image_extensions:
                continue
            
            train_image_count += 1
            
            # 计算图片的感知哈希值
            try:
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    continue
                
                with Image.open(file_path) as img:
                    try:
                        # 将图片转换为RGB模式以处理各种图片格式
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 尝试加载图片数据，检测是否损坏
                        img.load()
                        
                        # 计算训练集图片的哈希值
                        train_hash = imagehash.dhash(img, hash_size=hash_size)
                        
                        # 检查是否与测试集中的任何图片相似
                        is_duplicate = False
                        matching_test_file = None
                        
                        # 遍历所有测试集哈希值
                        for test_hash, test_file in test_hashes.items():
                            # 计算哈希差异
                            hash_diff = train_hash - test_hash
                            
                            # 如果差异小于阈值，认为是相同图片
                            if hash_diff <= similarity_threshold:
                                is_duplicate = True
                                matching_test_file = test_file
                                break
                        
                        # 如果找到重复，删除训练集中的图片
                        if is_duplicate:
                            duplicates_found += 1
                            print(f"\n发现重复图片 #{duplicates_found}:")
                            print(f"测试集: {matching_test_file}")
                            print(f"训练集: {file_path}")
                            
                            try:
                                print(f"删除训练集中的重复图片: {file_path}")
                                os.remove(file_path)
                                deleted_files += 1
                            except Exception as e:
                                print(f"无法删除文件 {file_path}: {e}")
                    
                    except Exception as e:
                        print(f"警告: 训练集图片文件损坏 {file_path}: {e}")
            except Exception as e:
                print(f"无法处理训练集文件 {file_path}: {e}")
    
    # 打印摘要
    print("\n====== 操作摘要 ======")
    print(f"测试集图片数: {test_image_count}")
    print(f"训练集图片数: {train_image_count}")
    print(f"发现的重复图片数: {duplicates_found}")
    print(f"已删除的训练集重复图片: {deleted_files}")
    print("=====================")

def main():
    # 指定测试集和训练集目录
    test_dir = r"C:\Users\chy\Desktop\新建文件夹"
    train_dir = r"C:\Users\chy\Desktop\image_downloader_gui_v1.1.1\download_images"
    
    # 首先安装必要的库
    try:
        import imagehash
    except ImportError:
        print("缺少必要的库。请先安装以下依赖:")
        print("pip install Pillow imagehash")
        return
    
    # 执行重复检测和删除
    detect_and_remove_cross_set_duplicates(test_dir, train_dir)

if __name__ == "__main__":
    main()
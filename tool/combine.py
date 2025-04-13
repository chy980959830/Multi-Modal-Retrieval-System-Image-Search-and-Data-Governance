import os
import shutil
import re

def merge_and_rename_images(base_dir):
    """
    将中文文件夹中的图片合并到对应的英文文件夹，并重命名
    
    参数:
        base_dir (str): 包含所有图片文件夹的根目录
    """
    # 定义文件夹映射关系
    folder_mapping = {
        '猫': 'cat',
        '狗': 'dog',
        '马': 'horse',
        '水墨画': 'ink_painting',
        '瓷器': 'porcelain'
    }
    
    # 图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # 处理每个文件夹映射
    for cn_folder, en_folder in folder_mapping.items():
        cn_path = os.path.join(base_dir, cn_folder)
        en_path = os.path.join(base_dir, en_folder)
        
        # 检查文件夹是否存在
        if not os.path.exists(cn_path):
            print(f"警告: 中文文件夹 '{cn_path}' 不存在，跳过处理。")
            continue
        
        if not os.path.exists(en_path):
            print(f"警告: 英文文件夹 '{en_path}' 不存在，跳过处理。")
            continue
        
        print(f"\n处理映射: {cn_folder} -> {en_folder}")
        
        # 获取英文文件夹中已有的图片文件
        en_images = []
        for file in os.listdir(en_path):
            file_path = os.path.join(en_path, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                en_images.append(file)
        
        # 计算下一个序号
        # 首先查找当前文件夹中符合命名模式的最大序号
        max_index = 0
        pattern = re.compile(rf'^{en_folder}(\d+)\.(jpg|jpeg)$', re.IGNORECASE)
        for file in en_images:
            match = pattern.match(file)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
        
        # 下一个序号
        next_index = max_index + 1
        
        # 获取中文文件夹中的图片文件
        cn_images = []
        for file in os.listdir(cn_path):
            file_path = os.path.join(cn_path, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                cn_images.append(file)
        
        print(f"找到 {len(cn_images)} 个图片文件需要合并")
        
        # 复制并重命名中文文件夹中的图片到英文文件夹
        for file in cn_images:
            old_path = os.path.join(cn_path, file)
            
            # 确定新文件名和路径
            new_filename = f"{en_folder}{next_index}.jpg"
            new_path = os.path.join(en_path, new_filename)
            
            # 复制文件
            try:
                shutil.copy2(old_path, new_path)
                print(f"已复制: {old_path} -> {new_path}")
                next_index += 1
            except Exception as e:
                print(f"复制 {old_path} 到 {new_path} 时出错: {e}")
    
    print("\n所有文件处理完成！")
    
    # 重命名英文文件夹中的所有图片文件
    print("\n开始重命名英文文件夹中的所有图片...")
    for folder in folder_mapping.values():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        
        # 获取所有图片文件
        images = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(file)
        
        # 排序文件名
        images.sort()
        
        # 创建临时文件夹用于重命名操作
        temp_dir = os.path.join(base_dir, f"temp_{folder}")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 将文件移动到临时文件夹并重命名
        for i, file in enumerate(images, 1):
            old_path = os.path.join(folder_path, file)
            temp_filename = f"{folder}{i}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                shutil.copy2(old_path, temp_path)
            except Exception as e:
                print(f"重命名 {old_path} 时出错: {e}")
        
        # 删除原文件
        for file in images:
            try:
                os.remove(os.path.join(folder_path, file))
            except Exception as e:
                print(f"删除 {os.path.join(folder_path, file)} 时出错: {e}")
        
        # 将临时文件夹中的文件移回原文件夹
        for file in os.listdir(temp_dir):
            try:
                shutil.move(os.path.join(temp_dir, file), os.path.join(folder_path, file))
            except Exception as e:
                print(f"移动 {os.path.join(temp_dir, file)} 时出错: {e}")
        
        # 删除临时文件夹
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"删除临时文件夹 {temp_dir} 时出错: {e}")
        
        print(f"重命名完成: {folder} 文件夹中的 {len(images)} 个文件已重命名为 {folder}1.jpg 到 {folder}{len(images)}.jpg")
    
    print("\n所有文件处理完成！")

def main():
    # 指定根目录
    base_dir = r"C:\Users\chy\Desktop\image_downloader_gui_v1.1.1\download_images"
    
    # 执行合并和重命名操作
    merge_and_rename_images(base_dir)

if __name__ == "__main__":
    main()
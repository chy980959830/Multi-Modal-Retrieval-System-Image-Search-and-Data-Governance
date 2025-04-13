import os
import random
import string

def rename_files_in_directory(directory):
    # 获取目录名称（取最后一个文件夹名）
    folder_name = os.path.basename(directory)
    
    # 获取目录下所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print(f"文件夹 {directory} 中没有文件")
        return
    
    # 第一步：将文件名打乱
    print(f"开始打乱 {directory} 中的 {len(files)} 个文件名称...")
    
    # 用于存储原文件名和临时文件名的映射
    temp_names = {}
    
    for filename in files:
        # 获取文件扩展名
        base, extension = os.path.splitext(filename)
        
        # 生成随机字符串作为临时文件名
        random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + extension
        
        # 确保随机名称不重复
        while os.path.exists(os.path.join(directory, random_name)):
            random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + extension
        
        # 存储原文件名和临时文件名的映射
        temp_names[filename] = random_name
        
        # 重命名文件为临时名称
        old_path = os.path.join(directory, filename)
        temp_path = os.path.join(directory, random_name)
        os.rename(old_path, temp_path)
    
    print(f"文件夹 {directory} 中的文件名打乱完成！")
    
    # 第二步：按照指定格式重命名文件
    print(f"开始按照指定格式重命名 {directory} 中的文件...")
    
    # 获取当前临时文件列表
    temp_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # 为每个文件按顺序重命名
    for index, temp_file in enumerate(temp_files, 1):
        # 获取文件扩展名
        _, extension = os.path.splitext(temp_file)
        
        # 生成新的文件名格式："目录名+序号"
        new_filename = f"{folder_name}{index}{extension}"
        
        # 重命名文件
        temp_path = os.path.join(directory, temp_file)
        new_path = os.path.join(directory, new_filename)
        
        # 如果新文件名已存在，则在序号前添加一个"_"以避免冲突
        while os.path.exists(new_path):
            new_filename = f"{folder_name}_{index}{extension}"
            new_path = os.path.join(directory, new_filename)
        
        os.rename(temp_path, new_path)
        
    print(f"文件夹 {directory} 中的文件重命名完成！")

def process_all_subdirectories(main_directory):
    print(f"开始处理主文件夹: {main_directory}")
    
    # 获取所有子文件夹
    subdirectories = []
    for item in os.listdir(main_directory):
        item_path = os.path.join(main_directory, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    
    if not subdirectories:
        print(f"警告: 在 {main_directory} 中没有找到子文件夹")
        return
    
    # 处理每个子文件夹
    for subdir in subdirectories:
        rename_files_in_directory(subdir)
    
    print(f"所有子文件夹处理完成！共处理了 {len(subdirectories)} 个子文件夹。")

def main():
    # 主文件夹路径
    main_directory = r"C:\Users\chy\Desktop\llava_dataset4"
    
    # 处理所有子文件夹
    process_all_subdirectories(main_directory)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        input("按回车键退出...")
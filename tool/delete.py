import os
import sys

def delete_non_jpg_images(directory, test_mode=True):
    """
    删除指定目录下所有非jpg/jpeg格式的图片
    
    参数:
        directory (str): 包含图片的目录路径
        test_mode (bool): 如果为True，仅显示将被删除的文件但不实际删除
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在。")
        return
    
    # 支持的图片扩展名（用于识别图片文件）
    image_extensions = ['.png', '.bmp', '.gif', '.tiff', '.webp', '.jpeg', '.jpg']
    # 要保留的扩展名
    keep_extensions = ['.jpg', '.jpeg']
    
    # 存储要删除的文件
    files_to_delete = []
    
    # 遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 获取文件路径和扩展名
            file_path = os.path.join(root, file)
            _, file_ext = os.path.splitext(file)
            file_ext = file_ext.lower()
            
            # 如果是图片文件且不是要保留的格式
            if file_ext in image_extensions and file_ext not in keep_extensions:
                files_to_delete.append(file_path)
    
    # 如果没有找到要删除的文件
    if not files_to_delete:
        print("没有找到需要删除的非jpg/jpeg图片文件。")
        return
    
    # 显示要删除的文件列表
    print(f"找到 {len(files_to_delete)} 个非jpg/jpeg格式的图片文件:")
    for file in files_to_delete:
        print(f"- {file}")
    
    # 在测试模式下，仅显示文件列表
    if test_mode:
        print("\n这是测试模式，没有文件被删除。")
        print("要实际删除这些文件，请将test_mode参数设置为False。")
        return
    
    # 确认删除
    print("\n准备删除上述文件...")
    
    # 删除文件
    deleted_count = 0
    errors = []
    
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"已删除: {file}")
            deleted_count += 1
        except Exception as e:
            errors.append(f"无法删除 {file}: {e}")
    
    # 显示结果
    print(f"\n删除完成! 已删除 {deleted_count} 个文件。")
    if errors:
        print(f"遇到 {len(errors)} 个错误:")
        for error in errors:
            print(f"- {error}")

def main():
    # 指定目录
    directory = r"C:\Users\chy\Desktop\image_downloader_gui_v1.1.1\download_images"
    
    # 允许通过命令行参数覆盖默认目录
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    # 直接运行删除模式，不再使用测试模式
    delete_non_jpg_images(directory, test_mode=False)

if __name__ == "__main__":
    main()
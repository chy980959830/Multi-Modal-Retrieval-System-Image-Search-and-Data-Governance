from PIL import Image
import os
import sys

def convert_to_jpg(directory):
    """
    将指定目录下的所有图片转换为JPG格式
    
    参数:
        directory (str): 包含图片的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在。")
        return
    
    # 支持的图片格式
    image_extensions = ['.png', '.bmp', '.gif', '.tiff', '.webp']
    
    # 用于统计的计数器
    total_files = 0
    converted_files = 0
    
    # 遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 获取文件扩展名
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            file_ext = file_ext.lower()
            
            # 如果已经是JPG格式或不是支持的图片格式则跳过
            if file_ext in ['.jpg', '.jpeg']:
                continue
            
            if file_ext not in image_extensions:
                continue
            
            total_files += 1
            
            try:
                # 打开图片
                img = Image.open(file_path)
                
                # 转换所有图片模式到RGB
                if img.mode == 'P':
                    # 处理调色板模式
                    img = img.convert('RGB')
                elif img.mode in ('RGBA', 'LA'):
                    # 处理带透明通道的图片
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                    img = bg
                elif img.mode != 'RGB':
                    # 处理其他所有非RGB模式
                    img = img.convert('RGB')
                
                # 带jpg扩展名的新文件名
                new_file_path = os.path.join(root, f"{file_name}.jpg")
                
                # 保存为JPG
                img.save(new_file_path, 'JPEG', quality=95)
                
                print(f"已转换: {file_path} -> {new_file_path}")
                converted_files += 1
                
            except Exception as e:
                print(f"转换 {file_path} 时出错: {e}")
    
    # 打印摘要
    print(f"\n转换完成! 共将 {converted_files} 个文件(共 {total_files} 个)转换为JPG格式。")

def main():
    # 默认使用指定的图片目录
    directory = r"C:\Users\chy\Desktop\image_downloader_gui_v1.1.1\download_images"
    
    # 仍然允许通过命令行参数覆盖默认目录
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        
    convert_to_jpg(directory)

if __name__ == "__main__":
    main()
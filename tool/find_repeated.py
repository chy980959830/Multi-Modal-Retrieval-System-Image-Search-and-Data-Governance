import os
import hashlib
from PIL import Image
from collections import defaultdict

def calculate_image_hash(image_path):
    """
    Calculate a hash of the image content to identify duplicates
    """
    try:
        with Image.open(image_path) as img:
            # Convert to a consistent format and size to handle minor differences
            img = img.convert('RGB')
            # Calculate hash based on the raw pixel data
            img_data = img.tobytes()
            return hashlib.md5(img_data).hexdigest()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_all_images(folder_path):
    """
    Recursively find all image files in the folder and its subfolders
    """
    image_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, filename))
    
    return image_files

def find_and_remove_duplicate_images(reference_folder, delete_folder):
    """
    Find duplicate images between reference_folder and delete_folder
    Only delete duplicates from delete_folder
    """
    # Get all images from both folders
    reference_images = get_all_images(reference_folder)
    delete_images = get_all_images(delete_folder)
    
    print(f"Found {len(reference_images)} images in reference folder")
    print(f"Found {len(delete_images)} images in delete folder")
    
    # Store hashes of reference images
    reference_hashes = {}
    for img_path in reference_images:
        file_hash = calculate_image_hash(img_path)
        if file_hash:
            reference_hashes[file_hash] = img_path
    
    # Find and delete duplicates in delete_folder
    deleted_files = []
    kept_files = []
    
    for img_path in delete_images:
        file_hash = calculate_image_hash(img_path)
        if file_hash and file_hash in reference_hashes:
            # This is a duplicate, delete it
            try:
                os.remove(img_path)
                deleted_files.append((img_path, reference_hashes[file_hash]))
            except Exception as e:
                print(f"Failed to delete {img_path}: {e}")
        else:
            # Not a duplicate, keep it
            kept_files.append(img_path)
    
    return deleted_files, kept_files, len(reference_images), len(delete_images)

def main():
    # Set your reference and delete folders
    reference_folder = r"C:\Users\chy\Desktop\llava_dataset4\dog_negative"
    delete_folder = r"C:\Users\chy\Desktop\image_downloader_gui_v1.1.1\download_images\wolf"
    
    # Check if folders exist
    if not os.path.exists(reference_folder):
        print(f"Error: Reference folder '{reference_folder}' does not exist.")
        return
        
    if not os.path.exists(delete_folder):
        print(f"Error: Delete folder '{delete_folder}' does not exist.")
        return
    
    print(f"Scanning for duplicate images...")
    print(f"Reference folder: {reference_folder}")
    print(f"Delete folder: {delete_folder}")
    
    deleted_files, kept_files, total_reference, total_delete = find_and_remove_duplicate_images(
        reference_folder, delete_folder
    )
    
    # Print summary
    print(f"\nAnalysis and cleanup complete!")
    print(f"Total images in reference folder: {total_reference}")
    print(f"Total images in delete folder: {total_delete}")
    print(f"Duplicate images deleted: {len(deleted_files)}")
    print(f"Images kept in delete folder: {len(kept_files)}")
    
    # Print detailed information on what was deleted
    if deleted_files:
        print("\nDeleted the following duplicate files:")
        for delete_file, reference_file in deleted_files:
            print(f"  - {delete_file} (duplicate of {reference_file})")
    else:
        print("\nNo duplicate images found. Nothing was deleted.")

if __name__ == "__main__":
    main()
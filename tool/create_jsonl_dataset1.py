import os
import json
import uuid
from pathlib import Path

def create_llava_dataset(root_folder):
    # Define categories
    categories = ["cat", "dog", "horse", "ink painting", "porcelain"]
    
    # Initialize dataset list
    dataset = []
    
    # Track total images
    total_images = 0
    
    # Process each category folder
    for category in categories:
        category_path = Path(root_folder) / category
        
        # Ensure folder exists
        if not category_path.exists() or not category_path.is_dir():
            print(f"Warning: Category folder {category} not found")
            continue
        
        # Get all images in the category
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(category_path.glob(f"*{ext}")))
            image_files.extend(list(category_path.glob(f"*{ext.upper()}")))
        
        # Remove duplicates by converting paths to strings and using a set
        unique_paths = set()
        unique_image_files = []
        
        for img in image_files:
            img_str = str(img).lower()  # Convert to lowercase for case-insensitive comparison
            if img_str not in unique_paths:
                unique_paths.add(img_str)
                unique_image_files.append(img)
        
        print(f"Found {len(unique_image_files)} unique images in category {category}")
        total_images += len(unique_image_files)
        
        # Create a record for each image
        for img_path in unique_image_files:
            # Get relative path
            relative_path = img_path.relative_to(root_folder)
            
            # Create sample record with the new format
            sample = {
                "id": str(uuid.uuid4()),
                "image": str(relative_path).replace("\\", "/"),
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
            
            # Add to dataset
            dataset.append(sample)
    
    # Save JSON file
    output_path = Path(root_folder) / "llava_dataset1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Total unique images found: {total_images}")
    print(f"Total records generated: {len(dataset)}")
    
    return dataset

# Usage
if __name__ == "__main__":
    folder_path = r"C:\Users\chy\Desktop\llava_dataset1"
    create_llava_dataset(folder_path)
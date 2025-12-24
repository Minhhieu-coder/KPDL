#!/usr/bin/env python3
"""
Example script for exploring the Bean Leaf Lesions Classification dataset.

This script demonstrates basic operations:
- Listing dataset contents
- Counting images by class
- Displaying basic statistics

Usage:
    python explore_dataset.py
"""

import os
from pathlib import Path
from collections import Counter


def explore_dataset(data_dir='./data'):
    """Explore the downloaded dataset."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Data directory '{data_dir}' not found!")
        print("Please download the dataset first using: python download_dataset.py")
        return False
    
    print("=" * 60)
    print("Bean Leaf Lesions Classification - Dataset Explorer")
    print("=" * 60)
    print()
    
    # Get all files and directories
    all_items = list(data_path.iterdir())
    
    # Filter out README.md
    items = [item for item in all_items if item.name != 'README.md']
    
    if not items:
        print("⚠️  Dataset appears to be empty!")
        print("Please download the dataset using: python download_dataset.py")
        return False
    
    print(f"📁 Dataset location: {data_path.absolute()}")
    print()
    
    # Count directories and files
    directories = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    print(f"📊 Dataset summary:")
    print(f"  • Directories: {len(directories)}")
    print(f"  • Files: {len(files)}")
    print()
    
    # If there are class directories, explore them
    if directories:
        print("📂 Class directories found:")
        class_stats = {}
        
        for class_dir in sorted(directories):
            # Count images in this class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            image_files = []
            # Use iterdir for immediate children, then check subdirectories if needed
            for item in class_dir.rglob('*'):
                if item.is_file() and item.suffix.lower() in image_extensions:
                    image_files.append(item)
            
            class_stats[class_dir.name] = len(image_files)
            print(f"  • {class_dir.name}: {len(image_files)} images")
        
        print()
        print("📈 Statistics:")
        if class_stats:
            total_images = sum(class_stats.values())
            print(f"  • Total images: {total_images}")
            print(f"  • Number of classes: {len(class_stats)}")
            avg_images = total_images / len(class_stats)
            print(f"  • Average images per class: {avg_images:.1f}")
    
    # List non-directory files in the root data folder
    if files:
        print()
        print("📄 Files in data directory:")
        for file in sorted(files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  • {file.name} ({size_mb:.2f} MB)")
    
    print()
    print("=" * 60)
    print("✅ Dataset exploration complete!")
    print()
    print("Next steps:")
    print("  • Use the dataset for training ML models")
    print("  • Create data loaders for your framework (PyTorch, TensorFlow, etc.)")
    print("  • Perform data augmentation and preprocessing")
    print("=" * 60)
    
    return True


def main():
    """Main function."""
    explore_dataset()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to download the Bean Leaf Lesions Classification dataset from Kaggle.

This script downloads the dataset from:
https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification

Requirements:
1. Install dependencies: pip install -r requirements.txt
2. Setup Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download kaggle.json
   - Place kaggle.json in ~/.kaggle/ directory
   - chmod 600 ~/.kaggle/kaggle.json

Usage:
    python download_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path


def setup_kaggle_credentials():
    """Check if Kaggle credentials are properly configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token' to download kaggle.json")
        print(f"4. Move kaggle.json to {kaggle_dir}/")
        print(f"5. Run: chmod 600 {kaggle_json}")
        return False
    
    # Check permissions
    if os.name != 'nt':  # Not Windows
        stat_info = kaggle_json.stat()
        if stat_info.st_mode & 0o077:
            print(f"WARNING: {kaggle_json} has incorrect permissions.")
            print(f"Run: chmod 600 {kaggle_json}")
            return False
    
    return True


def download_dataset():
    """Download the bean leaf lesions dataset from Kaggle."""
    try:
        import kaggle
        
        # Dataset information
        dataset_name = "marquis03/bean-leaf-lesions-classification"
        download_path = "./data"
        
        print(f"Downloading dataset: {dataset_name}")
        print(f"Download location: {download_path}")
        
        # Create data directory if it doesn't exist
        Path(download_path).mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"Dataset location: {os.path.abspath(download_path)}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for item in Path(download_path).rglob('*'):
            if item.is_file():
                size = item.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  - {item.relative_to(download_path)} ({size_mb:.2f} MB)")
        
        return True
        
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Install it with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Bean Leaf Lesions Classification Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    if not download_dataset():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Download completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

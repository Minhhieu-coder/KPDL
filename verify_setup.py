#!/usr/bin/env python3
"""
Setup verification script for the Bean Leaf Lesions Classification project.

This script checks if all requirements are properly configured.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.7+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    try:
        import kaggle
        print(f"  ✓ kaggle package installed (version {kaggle.__version__})")
        return True
    except ImportError:
        print("  ✗ kaggle package not installed")
        print("    Install with: pip install -r requirements.txt")
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    print("\nChecking Kaggle API credentials...")
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print(f"  ✗ Credentials not found at {kaggle_json}")
        print("    Please set up your Kaggle API token:")
        print("    1. Go to https://www.kaggle.com/account")
        print("    2. Click 'Create New API Token'")
        print(f"    3. Move kaggle.json to {kaggle_dir}/")
        print(f"    4. Run: chmod 600 {kaggle_json}")
        return False
    
    print(f"  ✓ Credentials found at {kaggle_json}")
    
    # Check permissions (Unix-like systems only)
    if os.name != 'nt':
        stat_info = kaggle_json.stat()
        if stat_info.st_mode & 0o077:
            print(f"  ⚠ Incorrect permissions on {kaggle_json}")
            print(f"    Run: chmod 600 {kaggle_json}")
            return False
        else:
            print("  ✓ Correct file permissions")
    
    return True


def check_data_directory():
    """Check if data directory exists."""
    print("\nChecking data directory...")
    data_dir = Path('./data')
    if data_dir.exists() and data_dir.is_dir():
        print(f"  ✓ Data directory exists")
        
        # Check if dataset is downloaded
        files = list(data_dir.glob('*'))
        dataset_files = [f for f in files if f.name != 'README.md']
        
        if dataset_files:
            print(f"  ✓ Found {len(dataset_files)} files/directories in data/")
            print("    Dataset appears to be downloaded")
        else:
            print("  ⚠ Data directory is empty (dataset not downloaded)")
            print("    Run: python download_dataset.py")
        
        return True
    else:
        print(f"  ✗ Data directory not found")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("Bean Leaf Lesions Classification - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_kaggle_credentials(),
        check_data_directory(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✓ All checks passed! Setup is complete.")
        print("\nYou can now:")
        print("  - Use the dataset for your ML projects")
        print("  - Start building classification models")
    else:
        print("⚠ Some checks failed. Please review the messages above.")
        print("\nNext steps:")
        if not checks[1]:
            print("  1. Install dependencies: pip install -r requirements.txt")
        if not checks[2]:
            print("  2. Set up Kaggle API credentials")
        if not all(checks):
            print("  3. Download the dataset: python download_dataset.py")
    print("=" * 60)
    
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

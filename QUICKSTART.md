# Quick Start Guide

## Getting Started with the Bean Leaf Lesions Classification Dataset

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up Kaggle API

1. **Get your Kaggle API token:**
   - Log in to [Kaggle](https://www.kaggle.com/)
   - Go to your account settings: https://www.kaggle.com/account
   - Scroll to the **API** section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json` to your computer

2. **Install the token:**
   
   **On Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   **On Windows:**
   ```cmd
   mkdir %USERPROFILE%\.kaggle
   move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

### Step 3: Verify Your Setup

```bash
python verify_setup.py
```

This will check:
- ✓ Python version
- ✓ Required packages
- ✓ Kaggle credentials
- ✓ Data directory

### Step 4: Download the Dataset

```bash
python download_dataset.py
```

The dataset will be downloaded to the `data/` directory.

### Step 5: Start Working!

The dataset is now ready to use for:
- Machine learning model training
- Computer vision experiments
- Plant disease classification research

## Troubleshooting

### "kaggle.json not found"
Make sure you placed the file in the correct location:
- Linux/Mac: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Permission errors (Linux/Mac)
Run: `chmod 600 ~/.kaggle/kaggle.json`

### Import errors
Install dependencies: `pip install -r requirements.txt`

## Additional Resources

- [Dataset on Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

---

**Need help?** Check the main [README.md](README.md) for detailed information.

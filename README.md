# KPDL - Bean Leaf Lesions Classification

This repository contains the Bean Leaf Lesions Classification dataset from Kaggle for machine learning research and development.

## 🚀 Quick Start

**New users:** See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide.

## Dataset Information

**Source:** [Bean Leaf Lesions Classification Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

This dataset contains images of bean leaves with various types of lesions, useful for developing machine learning models for plant disease classification.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API Credentials

To download the dataset, you need to set up Kaggle API credentials:

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to the **API** section
3. Click **"Create New API Token"** to download `kaggle.json`
4. Move the file to the appropriate location:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`
5. Set proper permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Verify Your Setup (Optional)

Check if everything is configured correctly:

```bash
python verify_setup.py
```

This script will verify:
- Python version compatibility
- Required packages installation
- Kaggle API credentials
- Data directory setup

### 4. Download the Dataset

Run the download script:

```bash
python download_dataset.py
```

The dataset will be downloaded and extracted to the `data/` directory.

## Alternative: Manual Download

If you prefer to download the dataset manually:

1. Visit the [dataset page](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
2. Click the **Download** button
3. Extract the downloaded ZIP file into the `data/` directory

## Project Structure

```
KPDL/
├── data/                      # Dataset directory (gitignored)
│   └── [dataset files]        # Bean leaf images and labels
├── download_dataset.py        # Automated dataset download script
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Notes

- The `data/` directory is excluded from version control via `.gitignore` to avoid committing large files to the repository
- Dataset files include images and associated metadata for bean leaf disease classification
- Make sure you have sufficient disk space before downloading (dataset size varies)

## Usage

Once downloaded, you can use the dataset for:
- Training machine learning models for plant disease classification
- Computer vision research
- Agricultural technology development
- Deep learning experiments

## License

Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification) for license information and terms of use.
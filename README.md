# 🌿 KPDL - Phân Loại Vết Bệnh Trên Lá Đậu (Bean Leaf Lesions Classification)

Dự án phân loại bệnh trên lá đậu sử dụng Deep Learning với Transfer Learning. Project này bao gồm notebook Jupyter hoàn chỉnh với EDA, huấn luyện mô hình và đánh giá kết quả.

## 📋 Mô tả dự án

Dự án sử dụng dataset từ Kaggle để phân loại 3 trạng thái của lá đậu:
- 🟢 **Healthy** (Lá khỏe)
- 🟡 **Angular Leaf Spot** (Vết bệnh góc)
- 🔴 **Bean Rust** (Bệnh rỉ sắt)

## 🚀 Quick Start

**Người dùng mới:** Xem [QUICKSTART.md](QUICKSTART.md) để bắt đầu nhanh.

## 📓 Notebook Chính

**[bean_leaf_classification.ipynb](bean_leaf_classification.ipynb)** - Notebook Jupyter hoàn chỉnh bao gồm:
- ✅ Khám phá dữ liệu (EDA) với biểu đồ trực quan
- ✅ Tiền xử lý ảnh và Data Augmentation
- ✅ Huấn luyện 3 mô hình Transfer Learning (ResNet50, MobileNetV2, VGG19)
- ✅ Đánh giá và so sánh mô hình
- ✅ Đề xuất nâng cao (Grad-CAM, EfficientNet, Vision Transformer)
- ✅ 100% chú thích tiếng Việt

## 📊 Dataset

**Nguồn:** [Bean Leaf Lesions Classification Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

Dataset gồm ảnh lá đậu đã được phân loại sẵn, phù hợp cho việc phát triển mô hình học máy phân loại bệnh cây trồng.

## 🔧 Cài đặt

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Cấu hình Kaggle API

Để tải dataset, bạn cần thiết lập Kaggle API credentials:

1. Truy cập [Kaggle Account Settings](https://www.kaggle.com/account)
2. Kéo xuống phần **API**
3. Click **"Create New API Token"** để tải `kaggle.json`
4. Di chuyển file đến vị trí phù hợp:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`
5. Đặt quyền truy cập (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Kiểm tra cài đặt (Tùy chọn)

Kiểm tra xem mọi thứ đã được cấu hình đúng chưa:

```bash
python verify_setup.py
```

Script này sẽ kiểm tra:
- Phiên bản Python
- Các thư viện đã cài đặt
- Kaggle API credentials
- Thư mục data

### 4. Tải Dataset

Chạy script tải dataset:

```bash
python download_dataset.py
```

Dataset sẽ được tải và giải nén vào thư mục `data/`.

### 5. Khám phá Dataset (Tùy chọn)

Sau khi tải xong, bạn có thể khám phá nội dung dataset:

```bash
python explore_dataset.py
```

Hiển thị:
- Vị trí và cấu trúc dataset
- Số lượng lớp và ảnh
- Thống kê về dataset

### 6. Chạy Notebook Phân Loại Bệnh Lá Đậu

Mở và chạy notebook chính:

```bash
jupyter notebook bean_leaf_classification.ipynb
```

Notebook bao gồm:
- 📊 Khám phá dữ liệu (EDA) với biểu đồ
- 🔄 Tiền xử lý và Data Augmentation
- 🧠 Huấn luyện 3 mô hình Deep Learning
- 📈 Đánh giá và so sánh kết quả
- 💡 Đề xuất cải thiện

## 📥 Tải thủ công (Tùy chọn)

Nếu bạn muốn tải dataset thủ công:

1. Truy cập [trang dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
2. Click nút **Download**
3. Giải nén file ZIP vào thư mục `data/`

## 📁 Cấu trúc Project

```
KPDL/
├── data/                           # Thư mục dataset (gitignored)
│   ├── README.md                   # Thông tin thư mục data
│   ├── train/                      # Ảnh huấn luyện (sau khi tải)
│   │   ├── angular_leaf_spot/
│   │   ├── bean_rust/
│   │   └── healthy/
│   └── validation/                 # Ảnh validation/test
│       ├── angular_leaf_spot/
│       ├── bean_rust/
│       └── healthy/
├── bean_leaf_classification.ipynb  # 📓 Notebook chính (EDA, Training, Evaluation)
├── download_dataset.py             # Script tải dataset tự động
├── explore_dataset.py              # Công cụ khám phá dataset
├── verify_setup.py                 # Script kiểm tra cài đặt
├── requirements.txt                # Các thư viện Python cần thiết
├── QUICKSTART.md                   # Hướng dẫn bắt đầu nhanh
├── .gitignore                      # Git ignore rules
└── README.md                       # File này
```

## 📝 Ghi chú

- Thư mục `data/` được loại trừ khỏi version control qua `.gitignore`
- Dataset gồm ảnh và metadata cho phân loại bệnh lá đậu
- Đảm bảo có đủ dung lượng đĩa trước khi tải

## 🎯 Sử dụng

Sau khi tải dataset, bạn có thể sử dụng cho:
- 🧠 Huấn luyện mô hình học máy phân loại bệnh cây
- 🔬 Nghiên cứu Computer Vision
- 🌾 Phát triển công nghệ nông nghiệp
- 🧪 Thử nghiệm Deep Learning

## 🏆 Kết quả dự kiến

Các mô hình trong notebook có thể đạt:
- **ResNet50:** ~85-92% accuracy
- **MobileNetV2:** ~82-90% accuracy
- **VGG19:** ~88-93% accuracy

*Kết quả có thể thay đổi tùy thuộc vào hyperparameters và số epoch huấn luyện.*

## 📜 License

Vui lòng tham khảo [trang dataset Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification) để biết thông tin về license và điều khoản sử dụng.

## 👨‍💻 Tác giả

Dự án này được thực hiện như bài tập lớn môn **Khai Phá Dữ Liệu**.

---

**© 2024 - Bean Leaf Lesions Classification Project**
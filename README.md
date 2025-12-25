# NLP Project: Trích xuất thuật ngữ Công nghệ (Technology Term Extraction)

Dự án môn học Xử lý ngôn ngữ tự nhiên (NLP), xây dựng ứng dụng Web trích xuất các thuật ngữ chuyên ngành công nghệ (ví dụ: "trí tuệ nhân tạo", "mô hình ngôn ngữ", "big data"...) từ văn bản tiếng Việt.

Dự án so sánh hiệu quả giữa các phương pháp **Machine Learning truyền thống** và **Deep Learning**.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## 📂 Cấu trúc dự án

Dự án được tổ chức thành 3 thư mục chính:

```text
NLP-app/
├── database/                # DỮ LIỆU
│   └── (Chứa 2000 câu văn bản tiếng Việt thô/raw data)
│
├── training/                # HUẤN LUYỆN MÔ HÌNH
│   ├── combined_data.json   # File dữ liệu 2099 câu văn bản tiếng việt đã gắn nhãn
│   ├── DATA TEST POS TAGGING.xlsx #File 120 câu văn bản tiếng việt dùng để test mô hình
│   ├── POS_TAGGING.ipynb    # Code chính phụ trách việc huấn luyện và chấm điểm mô hình
│   ├── ket_qua_chi_tiet.txt # Điểm của 4 mô hình ML và 1 mô hình DL sau huấn luyện
│   ├── TEST_POS_SVM_GOLD.json
│   └── deployment_resources/# Chứa các file mô hình đã huấn luyện
│       ├── bilstm.h5        # Model Bi-LSTM
│       ├── svm_final.joblib # Model SVM
│       ├── rf_final.joblib  # Model Random Forest (Cần tải thủ công)
│       ├── label_encoder.joblib 
│       ├── logreg_final.joblib
│       ├── max_len.json
│       ├── nb_final.joblib
│       ├── phrase_vocab.pkl
│       ├── tag2idx.json
│       ├── word2idx.json
│       └── vec_full.joblib
│
├── application/             # ỨNG DỤNG DEMO (STREAMLIT)
│   ├── app.py               # File chính để chạy ứng dụng
│   ├── requirements.txt     # Danh sách thư viện cần thiết
│   └── deployment_resources/# Chứa các file mô hình đã huấn luyện
│       ├── bilstm.h5        # Model Bi-LSTM
│       ├── svm_final.joblib # Model SVM
│       ├── rf_final.joblib  # Model Random Forest (Cần tải thủ công)
│       ├── label_encoder.joblib 
│       ├── logreg_final.joblib
│       ├── max_len.json
│       ├── nb_final.joblib
│       ├── phrase_vocab.pkl
│       ├── tag2idx.json
│       ├── word2idx.json
│       └── vec_full.joblib
│
├── venv/                    # Môi trường ảo (Không lưu trên Git)
└── README.md                # Hướng dẫn sử dụng

```

## ✨ Tính năng & Mô hình

Hệ thống đáp ứng yêu cầu sử dụng **3 mô hình Machine Learning** và **1 mô hình Deep Learning**:

1. **Machine Learning:**
* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest
* Naive Bayes


2. **Deep Learning:**
* Bi-LSTM (Bidirectional Long Short-Term Memory)



## 🛠 Yêu cầu hệ thống (Quan trọng)

Để đảm bảo tương thích với các mô hình đã huấn luyện (đặc biệt là TensorFlow), bắt buộc sử dụng:

* **Python:** Phiên bản **3.10** (Khuyên dùng 3.10.11).
* **Hệ điều hành:** Windows, macOS, hoặc Linux.

## 🚀 Hướng dẫn Cài đặt & Chạy

### Bước 1: Clone dự án

```
git clone https://github.com/Sunphuynx/NLP-app.git
cd NLP-app

```

### Bước 2: Tạo và kích hoạt môi trường ảo

* **Windows:**
```
# Đảm bảo dùng Python 3.10
py -3.10 -m venv venv
.\venv\Scripts\activate

```


* **macOS / Linux:**
```
python3.10 -m venv venv
source venv/bin/activate

```



### Bước 3: Cài đặt thư viện

Lưu ý file `requirements.txt` nằm trong thư mục `application`:

```
pip install -r application/requirements.txt

```

### Bước 4: Tải bổ sung Model nặng (BẮT BUỘC)

Do giới hạn của GitHub, file mô hình **Random Forest (`rf_final.joblib`)** (>100MB) không có sẵn trong mã nguồn này.

1. Liên hệ nhóm phát triển hoặc truy cập [Link Google Drive này](https://drive.google.com/file/d/1qK4AYXL4uhq_oRXQ4QqLuChW7VZRBtzu/view?usp=sharing) để tải file `rf_final.joblib`.
2. Copy file tải về vào thư mục: `application/deployment_resources/`

### Bước 5: Chạy ứng dụng

Do mã nguồn ứng dụng nằm trong thư mục `application`, bạn cần di chuyển vào đó trước khi chạy:

```
cd application
streamlit run app.py

```

Trình duyệt sẽ tự động mở tại `http://localhost:8501`.

---

## ⚠️ Các lưu ý khắc phục lỗi thường gặp

**1. Lỗi `ModuleNotFoundError: No module named 'numpy._core'`**

* **Nguyên nhân:** Xung đột giữa model train bằng Numpy 2.0 và App chạy Numpy 1.x.
* **Khắc phục:** Mã nguồn `app.py` hiện tại đã được xử lý để tương thích. Nếu vẫn bị, hãy đảm bảo bạn đang cài đúng các phiên bản trong `requirements.txt`.

**2. Lỗi không tìm thấy file `app.py`**

* Hãy chắc chắn bạn đã chạy lệnh `cd application` trước khi gõ `streamlit run app.py`.

**3. WordCloud bị lỗi ô vuông (□□□)**

* Máy thiếu font tiếng Việt. Hãy tải file font `Roboto.ttf` và bỏ vào thư mục `application/deployment_resources/`.

---

**Nhóm thực hiện:**

* [Phạm Duy Hoàng]
* [Phùng Chí Tâm]
* [Nguyễn Minh Khoa]
* [Biện Bùi Duy Quang]

```

```

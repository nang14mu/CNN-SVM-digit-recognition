# 🔢 CNN-SVM-digit-recognition

![Demo Application](assets/demo.png)

A hybrid Machine Learning model for handwritten digit recognition that leverages a **Convolutional Neural Network (CNN)** for deep feature extraction and a **Support Vector Machine (SVM)** for high-performance classification. Both models are implemented from scratch using NumPy.

## 🧠 Kiến trúc Mô hình (Architecture)

Mô hình kết hợp hai phương pháp mạnh mẽ để nhận dạng chữ số viết tay:

1. **Feature Extractor (CNN):**
   - Mạng CNN được huấn luyện ban đầu với lớp Softmax ở cuối để học cách trích xuất các đặc trưng (feature extraction) hình ảnh.
   - Sau khi huấn luyện, lớp Softmax được loại bỏ. Phần Convolutional và Pooling layer sẽ đóng vai trò trích xuất vector đặc trưng từ ảnh đầu vào.
   
2. **Classifier (SVM):**
   - Các vector đặc trưng thu được từ CNN được đưa vào một mô hình Multi-class Support Vector Machine (One-vs-Rest).
   - SVM tìm ra các siêu phẳng phân cách tối ưu, giúp phân loại chính xác các chữ số từ 0 đến 9 dựa trên các đặc trưng chất lượng cao do CNN cung cấp.

*Lưu ý về thiết kế:* Mặc dù training loss function của CNN ban đầu được thiết kế để phân loại (Softmax) chứ không hoàn toàn tối ưu hóa thuần túy cho chất lượng feature để đưa vào SVM, việc kết hợp với SVM ở tầng cuối vẫn mang lại sự ổn định và độ chính xác phân lớp cực tốt nhờ khả năng tối đa hóa margin của SVM.

## 📊 Kết quả (Accuracy)

Bằng cách tinh chỉnh thuật toán để tận dụng toàn bộ **42,000 mẫu dữ liệu** trong tập huấn luyện (thay vì chỉ giới hạn ở 5000 mẫu như phiên bản sơ khai), mô hình đã đạt được bước nhảy vọt về khả năng tổng quát hóa (generalization) và độ chính xác tổng thể. Kỹ thuật mini-batch gradient descent đảm bảo tiến trình huấn luyện hiệu quả mà không bị tràn bộ nhớ (RAM overflow).

- **Kiến trúc mạng:** 1 Conv Layer (8 filters, 3x3) + Max Pooling + SVM
- **Training Data:** 42,000 ảnh kích thước 28x28 (Bộ dữ liệu MNIST qua Kaggle Digit Recognizer)
- **Độ chính xác kỳ vọng:** ~97%+ (Phụ thuộc vào số vòng lặp và learning rate)

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Installation & Usage)

### 1. Yêu cầu hệ thống (Prerequisites)
- Python 3.8+
- Node.js & npm (Cho Web Frontend)

### 2. Cài đặt Backend (Mô hình & API)

Clone repository và di chuyển vào thư mục dự án:

```bash
git clone https://github.com/nang14mu/CNN-SVM-digit-recognition.git
cd CNN-SVM-digit-recognition
```

Cài đặt các thư viện Python cơ bản:
```bash
pip install numpy pandas flask flask-cors
```

### 3. Huấn luyện Mô hình (Training)

Để tự huấn luyện mô hình CNN-SVM từ đầu với toàn bộ tập dữ liệu:

```bash
cd model
python train_and_save.py
```

Sau quá trình huấn luyện, file trọng số `saved_model_weights.pkl` sẽ được xuất ra trong thư mục `model`.

### 4. Khởi chạy Backend Server (Flask API)

Vẫn tại thư mục `model` (hoặc thư mục chứa `app.py`), chạy server:

```bash
python app.py
```
*Server API sẽ chạy tại: `http://localhost:5000`*

### 5. Khởi chạy Web Frontend (React/Vite)

Mở một terminal mới, chuyển đến thư mục `frontend` và chạy:

```bash
cd frontend
npm install
npm run dev
```
*Giao diện nhận dạng chữ số sẽ hiển thị trên trình duyệt (thường là tại `http://localhost:5173`). Bạn có thể dùng chuột để vẽ trực tiếp chữ số lên canvas và xem mô hình dự đoán theo thời gian thực!*

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

## 📊 Đánh giá Mô hình (Evaluation & Accuracy)

Thông qua việc cập nhật kiến trúc lên **2 lớp mạng Tích chập (CNN)** sâu hơn và tinh chỉnh hệ thống mini-batch gradient descent trên toàn bộ **42,000 mẫu dữ liệu**, mô hình đã đạt được khả năng nhận diện hình ảnh và trích xuất đặc trưng cực kỳ hiệu quả, đẩy độ chính xác phân loại của SVM lên một mốc mới.

- **Kiến trúc mạng:** 2 Conv Layers (Conv1: 8 filters, Conv2: 16 filters) + Max Pooling + Flatten + SVM Đa lớp
- **Tập dữ liệu:** 42,000 ảnh kích thước 28x28 (Bộ dữ liệu MNIST từ Kaggle Digit Recognizer)
- **Tốc độ:** Tối ưu hóa tính toán ma trận với `NumPy Tensordot`, giúp huấn luyện toàn bộ tập dữ liệu hiệu quả trên CPU mà không bị tràn RAM.

### Kết quả trên tập Validation (1000 mẫu)

- **Accuracy (Độ chính xác chung):** `97.90%`
- **Macro Average F1-Score:** `0.9783`

**Báo cáo chi tiết từng lớp số:**

| Lớp (Số) | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| **0** | 0.9796 | 0.9796 | 0.9796 |
| **1** | 0.9917 | 1.0000 | 0.9959 |
| **2** | 0.9899 | 0.9423 | 0.9655 |
| **3** | 0.9767 | 0.9921 | 0.9844 |
| **4** | 0.9634 | 1.0000 | 0.9814 |
| **5** | 0.9714 | 0.9714 | 0.9714 |
| **6** | 0.9664 | 0.9914 | 0.9787 |
| **7** | 0.9818 | 0.9818 | 0.9818 |
| **8** | 0.9625 | 0.9747 | 0.9686 |
| **9** | 1.0000 | 0.9485 | 0.9735 |

> **Phân tích lỗi (Error Analysis):** Kịch bản `evaluate.py` tích hợp sẵn khả năng vẽ Confusion Matrix và hiển thị trực quan các hình ảnh bị dự đoán sai thông qua `matplotlib`, giúp dễ dàng theo dõi và phân tích điểm yếu của mô hình.

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
python train.py
```

Sau quá trình huấn luyện, file trọng số `cnn_svm_weights.pkl` sẽ được xuất ra trong thư mục `model`.

### 4. Khởi chạy Backend Server (Flask API)

Vẫn tại thư mục `model` (hoặc thư mục chứa `api.py`), chạy server:

```bash
python api.py
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

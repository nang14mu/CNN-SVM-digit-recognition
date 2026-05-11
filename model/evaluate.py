# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import pickle
import os
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# --- 1. TẢI DỮ LIỆU ---
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../data/digit-recognizer/train.csv')
print("Đang đọc dữ liệu tập Validation...")
train_df = pd.read_csv(csv_path)
data = np.array(train_df)
m, n = data.shape
np.random.seed(42)
np.random.shuffle(data)

# Chỉ lấy 1000 mẫu đầu tiên làm Validation set để kiểm thử
data_dev = data[0:1000].T
Y_val = data_dev[0]
X_val = data_dev[1:n] / 255.

# --- 2. TẢI MÔ HÌNH CNN + SVM ---
model_path = os.path.join(base_dir, 'cnn_svm_weights.pkl')
print("Đang tải mô hình từ file cnn_svm_weights.pkl...")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# --- 3. ĐỊNH NGHĨA CÁC HÀM CNN ĐỂ DỰ ĐOÁN (BATCH) ---
def im2col(X, fh, fw, stride=1, padding=0):
    if padding > 0:
        X = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), 'constant')
    m, c, h, w = X.shape
    out_h = (h - fh) // stride + 1
    out_w = (w - fw) // stride + 1
    shape = (m, c, out_h, out_w, fh, fw)
    strides = (X.strides[0], X.strides[1], X.strides[2]*stride, X.strides[3]*stride, X.strides[2], X.strides[3])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

def conv2d_forward(X, W, b, stride=1, padding=0):
    f = W.shape[0]
    X_strided = im2col(X, W.shape[2], W.shape[3], stride, padding)
    Z = np.tensordot(X_strided, W, axes=([1, 4, 5], [1, 2, 3]))
    Z = np.transpose(Z, (0, 3, 1, 2)) + b.reshape((1, f, 1, 1))
    return Z

def maxpool_forward(X, pool_size=2):
    m, c, h, w = X.shape
    out_h, out_w = h // pool_size, w // pool_size
    X_sliced = X[:, :, :out_h*pool_size, :out_w*pool_size]
    X_reshaped = X_sliced.reshape(m, c, out_h, pool_size, out_w, pool_size)
    return np.max(X_reshaped, axis=(3, 5))

def ReLU(Z): return np.maximum(Z, 0)

def predict_batch(X):
    # Trích xuất đặc trưng với CNN
    Wc1 = model_data['cnn']['Wc1']
    bc1 = model_data['cnn']['bc1']
    Wc2 = model_data['cnn']['Wc2']
    bc2 = model_data['cnn']['bc2']
    m_samples = X.shape[1]
    A0 = X.T.reshape(m_samples, 1, 28, 28)
    
    Z1 = conv2d_forward(A0, Wc1, bc1)
    A1 = ReLU(Z1)
    A1_pool = maxpool_forward(A1)
    
    Z2 = conv2d_forward(A1_pool, Wc2, bc2)
    A2 = ReLU(Z2)
    A2_pool = maxpool_forward(A2)
    
    feature_vector = A2_pool.reshape(m_samples, -1) # Mảng có kích thước (m_samples, 400)
    
    # Phân loại với SVM
    scores = np.zeros((m_samples, 10))
    for c in range(10):
        w = model_data['svm'][c]['w']
        b = model_data['svm'][c]['b']
        scores[:, c] = np.dot(feature_vector, w) - float(b)
        
    return np.argmax(scores, axis=1)

print("Đang chạy mô hình dự đoán trên tập Validation...")
predictions = predict_batch(X_val)

acc = np.sum(predictions == Y_val) / Y_val.size
print(f"Accuracy (Độ chính xác chung): {acc * 100:.2f}%\n")

# ==========================================
# PHASE 1: CÁC METRIC BỔ SUNG (TỪ ĐẦU - FROM SCRATCH)
# ==========================================
print("=== PHASE 1: TÍNH TOÁN METRICS BỔ SUNG ===")
num_classes = 10

# 1. Confusion Matrix (Ma trận nhầm lẫn)
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(Y_val, predictions):
    cm[t, p] += 1

print("\nConfusion Matrix (Ma trận nhầm lẫn):")
print("Cột: Dự đoán (0->9) | Hàng: Thực tế (0->9)")
print("-" * 45)
print(cm)
print("-" * 45)

# 2. Precision, Recall, F1 (Tính bằng Macro-average)
precisions = []
recalls = []

print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 45)
for c in range(num_classes):
    TP = cm[c, c] # True Positive
    FP = np.sum(cm[:, c]) - TP # False Positive
    FN = np.sum(cm[c, :]) - TP # False Negative
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    
    print(f"Số {c:<4} | {precision:.4f}     | {recall:.4f}     | {f1_class:.4f}")

macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

print("-" * 45)
print(f"MACRO AVG| {macro_precision:.4f}     | {macro_recall:.4f}     | {macro_f1:.4f}\n")


# ==========================================
# PHASE 2: PHÂN TÍCH LỖI (ERROR ANALYSIS)
# ==========================================
print("=== PHASE 2: PHÂN TÍCH LỖI (HIỂN THỊ ẢNH) ===")
incorrect_indices = np.where(predictions != Y_val)[0]
print(f"Có {len(incorrect_indices)} dự đoán sai trên tổng số {Y_val.size} mẫu.")

if len(incorrect_indices) > 0:
    # Lấy tối đa 10 ảnh đoán sai để trực quan hóa
    display_count = min(10, len(incorrect_indices))
    sample_indices = incorrect_indices[:display_count]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Phân Tích Lỗi: Các Mẫu Mô Hình Dự Đoán Sai', fontsize=16, fontweight='bold', color='darkred')
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i//5, i%5]
        img = X_val[:, idx].reshape(28, 28)
        true_label = Y_val[idx]
        pred_label = predictions[idx]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Thực tế: {true_label}\nDự đoán: {pred_label}", color='red', fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    print("Vui lòng kiểm tra cửa sổ đồ thị (Matplotlib) vừa hiện lên để xem các hình ảnh bị đoán sai!")
    plt.show()

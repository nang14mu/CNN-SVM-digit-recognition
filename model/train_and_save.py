import numpy as np
import pandas as pd
import pickle
import os

# Tính toán đường dẫn tuyệt đối của thư mục chứa script
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../data/digit-recognizer/train.csv')

print("Đọc dữ liệu...")
train_df = pd.read_csv(csv_path)
data = np.array(train_df)
m, n = data.shape
np.random.seed(42)
np.random.shuffle(data)

# Chỉ lấy một phần để train cho nhanh nếu cần, nhưng ở đây dùng đúng như notebook của bạn
data_dev = data[0:1000].T
Y_val = data_dev[0]
X_val = data_dev[1:n] / 255.

# CHỈNH SỬA: Giới hạn lượng dữ liệu huấn luyện để máy yếu có thể chạy được (5000 mẫu)
# Nếu máy bạn chạy được, có thể đổi 6000 thành m để dùng toàn bộ dữ liệu.
train_limit = min(m, 6000)
data_train = data[1000:train_limit].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.

# --- MẠNG NƠ-RON TÍCH CHẬP (CNN) ---
def init_cnn_params():
    np.random.seed(42)
    # Lớp Conv1: 8 bộ lọc kích thước 3x3
    Wc = np.random.randn(8, 1, 3, 3) * np.sqrt(2. / 9)
    bc = np.zeros((8, 1))
    # Lớp Dense (dùng để huấn luyện trích xuất đặc trưng): 8*13*13 = 1352 -> 10
    Wd = np.random.randn(10, 1352) * np.sqrt(2. / 1352)
    bd = np.zeros((10, 1))
    return Wc, bc, Wd, bd

def ReLU(Z): return np.maximum(Z, 0)
def ReLU_deriv(Z): return Z > 0
def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0)
    return np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

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
    return Z, X_strided

def conv2d_backward(dZ, X_strided, W):
    dW = np.tensordot(dZ, X_strided, axes=([0, 2, 3], [0, 2, 3]))
    db = np.sum(dZ, axis=(0, 2, 3)).reshape(-1, 1)
    return dW, db

def maxpool_forward(X, pool_size=2):
    m, c, h, w = X.shape
    out_h, out_w = h // pool_size, w // pool_size
    X_reshaped = X.reshape(m, c, out_h, pool_size, out_w, pool_size)
    return np.max(X_reshaped, axis=(3, 5))

def maxpool_backward(dZ, X, pool_size=2):
    m, c, h, w = X.shape
    out_h, out_w = h // pool_size, w // pool_size
    dZ_expanded = np.repeat(np.repeat(dZ, pool_size, axis=2), pool_size, axis=3)
    X_reshaped = X.reshape(m, c, out_h, pool_size, out_w, pool_size)
    X_max = np.max(X_reshaped, axis=(3, 5), keepdims=True)
    mask = (X_reshaped == X_max).reshape(m, c, h, w)
    return mask * dZ_expanded

def cnn_forward_prop(Wc, bc, Wd, bd, X):
    m = X.shape[1]
    # Reshape input (784, m) -> (m, 1, 28, 28)
    A0 = X.T.reshape(m, 1, 28, 28)
    Z1, X_strided = conv2d_forward(A0, Wc, bc)
    A1 = ReLU(Z1)
    A1_pool = maxpool_forward(A1)
    A1_flat = A1_pool.reshape(m, -1).T # Shape: (1352, m)
    
    # Lớp Softmax chỉ dùng để lấy đạo hàm, sau này bỏ qua
    Z2 = Wd.dot(A1_flat) + bd
    A2 = softmax(Z2)
    return (X_strided, A1, A1_pool, A1_flat), Z2, A2

def cnn_backward_prop(cache, Z2, A2, Wc, Wd, Y):
    m = Y.size
    X_strided, A1, A1_pool, A1_flat = cache
    dZ2 = A2 - one_hot(Y)
    dWd = 1 / m * dZ2.dot(A1_flat.T)
    dbd = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1_flat = Wd.T.dot(dZ2)
    dA1_pool = dA1_flat.T.reshape(A1_pool.shape)
    dA1 = maxpool_backward(dA1_pool, A1)
    dZ1 = dA1 * ReLU_deriv(A1)
    
    dWc, dbc = conv2d_backward(dZ1, X_strided, Wc)
    dWc /= m; dbc /= m
    return dWc, dbc, dWd, dbd

def update_cnn_params(Wc, bc, Wd, bd, dWc, dbc, dWd, dbd, alpha):
    Wc -= alpha * dWc; bc -= alpha * dbc    
    Wd -= alpha * dWd; bd -= alpha * dbd    
    return Wc, bc, Wd, bd

def get_predictions(A2): return np.argmax(A2, 0)
def get_accuracy(predictions, Y): return np.sum(predictions == Y) / Y.size

print("Đang huấn luyện mạng CNN (dùng Softmax) để lấy khả năng trích xuất đặc trưng...")
Wc, bc, Wd, bd = init_cnn_params()
alpha = 0.1
iterations = 20 # Giảm số vòng lặp xuống do dùng mini-batch sẽ hội tụ nhanh hơn
batch_size = 128 # CHỈNH SỬA: Dùng mini-batch để tránh tràn RAM
m_train = X_train.shape[1]

for i in range(iterations):
    # Trộn dữ liệu cho mỗi epoch
    permutation = np.random.permutation(m_train)
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[permutation]
    
    for j in range(0, m_train, batch_size):
        X_batch = X_train_shuffled[:, j:j+batch_size]
        Y_batch = Y_train_shuffled[j:j+batch_size]
        
        cache, Z2, A2 = cnn_forward_prop(Wc, bc, Wd, bd, X_batch)
        dWc, dbc, dWd, dbd = cnn_backward_prop(cache, Z2, A2, Wc, Wd, Y_batch)
        Wc, bc, Wd, bd = update_cnn_params(Wc, bc, Wd, bd, dWc, dbc, dWd, dbd, alpha)
        
    if i % 5 == 0 or i == iterations - 1:
        # In độ chính xác của batch cuối cùng để theo dõi nhanh
        print(f"CNN Vòng lặp {i:4d} | Độ chính xác Batch cuối: {get_accuracy(get_predictions(A2), Y_batch)*100:.2f}%")

print("Đang trích xuất đặc trưng từ CNN (bỏ qua Softmax để đưa vào SVM)...")
# CHỈNH SỬA: Trích xuất đặc trưng theo từng batch để không bị tràn RAM
train_features_list = []
for j in range(0, m_train, batch_size):
    X_batch = X_train[:, j:j+batch_size]
    cache, _, _ = cnn_forward_prop(Wc, bc, Wd, bd, X_batch)
    train_features_list.append(cache[3].T)
train_features = np.vstack(train_features_list)

# --- SVM ---
class LinearSVM_Binary:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            margin = y * (np.dot(X, self.w) - self.b)
            misclassified = margin < 1
            X_mis = X[misclassified]
            y_mis = y[misclassified]
            dw = 2 * self.lambda_param * self.w - np.dot(y_mis, X_mis) / n_samples
            db = np.sum(y_mis) / n_samples
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def get_score(self, X): 
        return np.dot(X, self.w) - self.b

class MultiClassSVM_OVR:
    def __init__(self, n_classes=10, learning_rate=0.1, lambda_param=0.001, n_iters=1000):
        self.n_classes = n_classes
        self.models = [LinearSVM_Binary(learning_rate, lambda_param, n_iters) for _ in range(n_classes)]
        
    def fit(self, X, y):
        for c in range(self.n_classes):
            y_binary = np.where(y == c, 1, -1)
            self.models[c].fit(X, y_binary)

print("Đang huấn luyện SVM Đa Lớp tự code trên vector đặc trưng...")
custom_svm = MultiClassSVM_OVR(n_classes=10, learning_rate=0.5, lambda_param=0.001, n_iters=2000)
custom_svm.fit(train_features, Y_train)

# --- SAVE MODEL ---
print("Đang lưu trọng số của mô hình...")
model_data = {
    'cnn': {
        'Wc': Wc, 'bc': bc, 'Wd': Wd, 'bd': bd
    },
    'svm': [
        {'w': custom_svm.models[i].w, 'b': custom_svm.models[i].b} for i in range(10)
    ]
}

save_path = os.path.join(base_dir, 'saved_model_weights.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)
    
print(f"Lưu mô hình hoàn tất vào file '{save_path}'!")

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

# CHỈNH SỬA: Sử dụng toàn bộ dữ liệu huấn luyện để tăng độ chính xác
train_limit = m
data_train = data[1000:train_limit].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.

# --- MẠNG NƠ-RON TÍCH CHẬP (CNN) ---
def init_cnn_params():
    np.random.seed(42)
    # Lớp Conv1: 8 bộ lọc kích thước 3x3
    Wc1 = np.random.randn(8, 1, 3, 3) * np.sqrt(2. / 9)
    bc1 = np.zeros((8, 1))
    
    # Lớp Conv2: 16 bộ lọc kích thước 3x3, quét trên 8 kênh của lớp trước
    Wc2 = np.random.randn(16, 8, 3, 3) * np.sqrt(2. / (8 * 3 * 3))
    bc2 = np.zeros((16, 1))
    
    # Lớp Dense (dùng để huấn luyện trích xuất đặc trưng): 16*5*5 = 400 -> 10
    Wd = np.random.randn(10, 400) * np.sqrt(2. / 400)
    bd = np.zeros((10, 1))
    return Wc1, bc1, Wc2, bc2, Wd, bd

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

def conv2d_backward(dZ, X_strided, W, X_shape=None):
    dW = np.tensordot(dZ, X_strided, axes=([0, 2, 3], [0, 2, 3]))
    db = np.sum(dZ, axis=(0, 2, 3)).reshape(-1, 1)
    
    if X_shape is None:
        return dW, db
        
    m, c, h, w = X_shape
    f, _, fh, fw = W.shape
    _, _, out_h, out_w = dZ.shape
    
    dX = np.zeros(X_shape)
    for i in range(out_h):
        for j in range(out_w):
            dX[:, :, i:i+fh, j:j+fw] += np.tensordot(dZ[:, :, i, j], W, axes=([1], [0]))
    return dX, dW, db

def maxpool_forward(X, pool_size=2):
    m, c, h, w = X.shape
    out_h, out_w = h // pool_size, w // pool_size
    X_sliced = X[:, :, :out_h*pool_size, :out_w*pool_size]
    X_reshaped = X_sliced.reshape(m, c, out_h, pool_size, out_w, pool_size)
    return np.max(X_reshaped, axis=(3, 5))

def maxpool_backward(dZ, X, pool_size=2):
    m, c, h, w = X.shape
    out_h, out_w = h // pool_size, w // pool_size
    dZ_expanded = np.repeat(np.repeat(dZ, pool_size, axis=2), pool_size, axis=3)
    
    X_sliced = X[:, :, :out_h*pool_size, :out_w*pool_size]
    X_reshaped = X_sliced.reshape(m, c, out_h, pool_size, out_w, pool_size)
    X_max = np.max(X_reshaped, axis=(3, 5), keepdims=True)
    mask_sliced = (X_reshaped == X_max).reshape(m, c, out_h*pool_size, out_w*pool_size)
    
    dX = np.zeros_like(X)
    dX[:, :, :out_h*pool_size, :out_w*pool_size] = mask_sliced * dZ_expanded
    return dX

def cnn_forward_prop(Wc1, bc1, Wc2, bc2, Wd, bd, X):
    m = X.shape[1]
    # Reshape input (784, m) -> (m, 1, 28, 28)
    A0 = X.T.reshape(m, 1, 28, 28)
    
    # Lớp ẩn 1: Conv1 + ReLU + Pool1
    Z1, X1_strided = conv2d_forward(A0, Wc1, bc1)
    A1 = ReLU(Z1)
    A1_pool = maxpool_forward(A1)
    
    # Lớp ẩn 2: Conv2 + ReLU + Pool2
    Z2, X2_strided = conv2d_forward(A1_pool, Wc2, bc2)
    A2 = ReLU(Z2)
    A2_pool = maxpool_forward(A2)
    
    # Làm phẳng (Flatten)
    A2_flat = A2_pool.reshape(m, -1).T # Shape: (400, m)
    
    # Lớp Softmax chỉ dùng để lấy đạo hàm, sau này bỏ qua
    Z3 = Wd.dot(A2_flat) + bd
    A3 = softmax(Z3)
    
    cache = (X1_strided, A1, A1_pool, X2_strided, A2, A2_pool, A2_flat)
    return cache, Z3, A3

def cnn_backward_prop(cache, Z3, A3, Wc1, Wc2, Wd, Y):
    m = Y.size
    X1_strided, A1, A1_pool, X2_strided, A2, A2_pool, A2_flat = cache
    
    # Đạo hàm lớp Softmax & Dense
    dZ3 = A3 - one_hot(Y)
    dWd = 1 / m * dZ3.dot(A2_flat.T)
    dbd = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    
    # Đạo hàm đi qua lớp Pool2 và Conv2
    dA2_flat = Wd.T.dot(dZ3)
    dA2_pool = dA2_flat.T.reshape(A2_pool.shape)
    dA2 = maxpool_backward(dA2_pool, A2)
    dZ2 = dA2 * ReLU_deriv(A2)
    
    # Tính dX2 (chính là dA1_pool), dWc2, dbc2
    dA1_pool, dWc2, dbc2 = conv2d_backward(dZ2, X2_strided, Wc2, X_shape=A1_pool.shape)
    dWc2 /= m; dbc2 /= m
    
    # Đạo hàm đi qua lớp Pool1 và Conv1
    dA1 = maxpool_backward(dA1_pool, A1)
    dZ1 = dA1 * ReLU_deriv(A1)
    
    # Lớp đầu tiên không cần tính dX
    dWc1, dbc1 = conv2d_backward(dZ1, X1_strided, Wc1)
    dWc1 /= m; dbc1 /= m
    
    return dWc1, dbc1, dWc2, dbc2, dWd, dbd

def update_cnn_params(Wc1, bc1, Wc2, bc2, Wd, bd, dWc1, dbc1, dWc2, dbc2, dWd, dbd, alpha):
    Wc1 -= alpha * dWc1; bc1 -= alpha * dbc1    
    Wc2 -= alpha * dWc2; bc2 -= alpha * dbc2    
    Wd -= alpha * dWd; bd -= alpha * dbd    
    return Wc1, bc1, Wc2, bc2, Wd, bd

def get_predictions(A2): return np.argmax(A2, 0)
def get_accuracy(predictions, Y): return np.sum(predictions == Y) / Y.size

print("Đang huấn luyện mạng CNN (dùng Softmax) để lấy khả năng trích xuất đặc trưng...")
Wc1, bc1, Wc2, bc2, Wd, bd = init_cnn_params()
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
        
        cache, Z3, A3 = cnn_forward_prop(Wc1, bc1, Wc2, bc2, Wd, bd, X_batch)
        dWc1, dbc1, dWc2, dbc2, dWd, dbd = cnn_backward_prop(cache, Z3, A3, Wc1, Wc2, Wd, Y_batch)
        Wc1, bc1, Wc2, bc2, Wd, bd = update_cnn_params(Wc1, bc1, Wc2, bc2, Wd, bd, dWc1, dbc1, dWc2, dbc2, dWd, dbd, alpha)
        
    if i % 5 == 0 or i == iterations - 1:
        # In độ chính xác của batch cuối cùng để theo dõi nhanh
        print(f"CNN Vòng lặp {i:4d} | Độ chính xác Batch cuối: {get_accuracy(get_predictions(A3), Y_batch)*100:.2f}%")

print("Đang trích xuất đặc trưng từ CNN (bỏ qua Softmax để đưa vào SVM)...")
# CHỈNH SỬA: Trích xuất đặc trưng theo từng batch để không bị tràn RAM
train_features_list = []
for j in range(0, m_train, batch_size):
    X_batch = X_train[:, j:j+batch_size]
    cache, _, _ = cnn_forward_prop(Wc1, bc1, Wc2, bc2, Wd, bd, X_batch)
    train_features_list.append(cache[6].T) # cache[6] là A2_flat
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
        'Wc1': Wc1, 'bc1': bc1, 'Wc2': Wc2, 'bc2': bc2, 'Wd': Wd, 'bd': bd
    },
    'svm': [
        {'w': custom_svm.models[i].w, 'b': custom_svm.models[i].b} for i in range(10)
    ]
}

save_path = os.path.join(base_dir, 'saved_model_weights.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)
    
print(f"Lưu mô hình hoàn tất vào file '{save_path}'!")

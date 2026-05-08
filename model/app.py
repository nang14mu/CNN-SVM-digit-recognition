from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app)

import os

# Tải mô hình
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'saved_model_weights.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print("Mô hình đã được tải thành công!")
except FileNotFoundError:
    print("Không tìm thấy file saved_model_weights.pkl. Hãy chạy train_and_save.py trước.")
    model_data = None

def ReLU(Z): return np.maximum(Z, 0)

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
    X_reshaped = X.reshape(m, c, out_h, pool_size, out_w, pool_size)
    return np.max(X_reshaped, axis=(3, 5))

def predict_digit(image_array):
    if not model_data:
        return -1
        
    # --- 1. FEATURE EXTRACTION (Mạng Nơ-ron Tích chập - CNN) ---
    Wc = model_data['cnn']['Wc']
    bc = model_data['cnn']['bc']
    
    # image_array có kích thước (28, 28), thêm batch và channel để thành (1, 1, 28, 28)
    A0 = image_array.reshape(1, 1, 28, 28)
    
    Z1 = conv2d_forward(A0, Wc, bc)
    A1 = ReLU(Z1)
    A1_pool = maxpool_forward(A1)
    
    # Flatten thành (1, 1352) để đưa vào SVM
    feature_vector = A1_pool.reshape(1, -1)
    
    # --- 2. CLASSIFICATION (SVM) ---
    scores = np.zeros(10)
    for c in range(10):
        w = model_data['svm'][c]['w']
        b = model_data['svm'][c]['b']
        # Dùng float() để tránh lỗi DeprecationWarning của NumPy khi ép mảng (1,) sang vô hướng
        scores[c] = float(np.dot(feature_vector, w)) - float(b)
        
    return int(np.argmax(scores))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Dữ liệu ảnh dạng base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Đọc ảnh bằng OpenCV
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # 1. Resize ảnh xuống khoảng 500px nếu nó quá to để adaptive threshold chạy tốt
        h, w = img_cv.shape
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
            
        # 2. Làm mờ nhẹ để khử nhiễu hột
        blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
        
        # 3. Adaptive Thresholding (Tìm nét bút, bỏ hoàn toàn bóng râm và màu giấy)
        # THRESH_BINARY_INV tự động đảo màu: chữ thành trắng (255), giấy thành đen (0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 51, 15)
                                       
        # 4. Tìm nét chữ chính xác nhất ở giữa ảnh (tránh viền mép giấy)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        clean_img = np.zeros_like(binary)
        if num_labels > 1:
            max_score = -1
            best_label = 1
            h_b, w_b = binary.shape
            center_x, center_y = w_b / 2, h_b / 2
            
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                if area < 30: continue # Bỏ qua các chấm nhiễu li ti
                
                # Tính khoảng cách từ nét chữ này đến tâm ảnh
                cx, cy = centroids[i]
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                # Chữ số thường to và được chụp ở trung tâm ảnh
                score = area / (dist + 50)
                if score > max_score:
                    max_score = score
                    best_label = i
                    
            if max_score != -1:
                clean_img[labels == best_label] = 255
            else:
                clean_img = binary
        else:
            clean_img = binary
            
        # 5. Tìm Bounding box và Cắt gọt chuẩn MNIST
        rows = np.any(clean_img > 0, axis=1)
        cols = np.any(clean_img > 0, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            cropped = clean_img[rmin:rmax+1, cmin:cmax+1]
            
            # Làm đậm nét chữ (Dilation) trước khi thu nhỏ
            # Tập MNIST có nét chữ khá dày. Chữ viết tay bằng bút bi khi thu nhỏ sẽ rất mờ và mỏng.
            # Ta tự động tính toán độ dày cần bơm thêm dựa trên chiều cao của chữ.
            thickness = max(2, int(cropped.shape[0] / 15))
            kernel = np.ones((thickness, thickness), np.uint8)
            cropped = cv2.dilate(cropped, kernel, iterations=1)
            
            cropped_img = Image.fromarray(cropped)
            
            # Resize fit vào hộp 20x20
            width, height = cropped_img.size
            aspect_ratio = width / height
            
            if width > height:
                new_width = 20
                new_height = max(1, int(20 / aspect_ratio))
            else:
                new_height = 20
                new_width = max(1, int(20 * aspect_ratio))
                
            # Dùng LANCZOS để resize mượt mà
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
                
            cropped_img = cropped_img.resize((new_width, new_height), resample_filter)
            
            # --- CĂN GIỮA BẰNG TRỌNG TÂM (CENTER OF MASS) ---
            # Tập MNIST xịn căn giữa theo trọng tâm nét mực, không phải theo hình học!
            # Điều này là CỰC KỲ QUAN TRỌNG với mạng Nơ-ron thẳng (MLP) vì nó không có tính bất biến tịnh tiến như CNN.
            resized_array = np.array(cropped_img)
            M = cv2.moments(resized_array)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx = new_width / 2.0
                cy = new_height / 2.0
                
            # Tính toán tọa độ paste sao cho (cx, cy) nằm ở điểm (14, 14) của ảnh 28x28
            paste_x = int(round(14.0 - cx))
            paste_y = int(round(14.0 - cy))
            
            # Đặt vào nền đen 28x28
            final_img = Image.new('L', (28, 28), 0)
            final_img.paste(cropped_img, (paste_x, paste_y))
            
            img_array = np.array(final_img, dtype=np.float32)
            max_val = np.max(img_array)
            if max_val > 0:
                img_array = (img_array / max_val) * 255.0
            final_img = Image.fromarray(img_array.astype(np.uint8))
        else:
            # Nếu không tìm thấy nét chữ
            final_img = Image.fromarray(clean_img).resize((28, 28))
            img_array = np.array(final_img, dtype=np.float32)
        
        # --- LƯU ẢNH PROCESSED DƯỚI DẠNG BASE64 ĐỂ HIỂN THỊ LÊN FRONTEND ---
        buffered = BytesIO()
        final_img.save(buffered, format="PNG")
        processed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        processed_image_data = f"data:image/png;base64,{processed_base64}"
        
        # Chuẩn hóa về 0-1
        img_array = img_array / 255.0
        
        # Dự đoán trực tiếp với ma trận 28x28 (Không cần Flatten thành 784 nữa vì CNN dùng ảnh 2D)
        prediction = predict_digit(img_array)
        
        return jsonify({
            'prediction': prediction,
            'processed_image': processed_image_data,
            'success': True
        })
        
    except Exception as e:
        print("Lỗi:", e)
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    print("Backend AI đang chạy tại http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
    

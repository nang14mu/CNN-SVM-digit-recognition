# PHÂN TÍCH CHI TIẾT CÁC HÀM, Ý NGHĨA VÀ HÌNH DẠNG (SHAPE) BIẾN TRONG TRAIN.PY

Tài liệu này phân tích chi tiết từng hàm trong file [train.py](file:///d:/Ki_2_nam_3/H%E1%BB%8Dc%20m%C3%A1y/BTL/model/train.py), giải thích ý nghĩa thuật toán, cơ chế hoạt động chi tiết, định dạng đầu vào và cấu trúc hình dạng (shape) của đầu ra. Định dạng tài liệu được tối ưu hóa để có thể xuất trực tiếp sang file Word (.docx).

---

## I. PHẦN TẢI DỮ LIỆU VÀ TIỀN XỬ LÝ (DATA LOADING)

### 1. Đọc dữ liệu từ file CSV
Dữ liệu được tải trực tiếp từ tệp `train.csv` của MNIST Kaggle.
*   **Ma trận gốc `data`**:
    *   **Shape**: `(42000, 785)` (42,000 ảnh vẽ tay, mỗi hàng gồm 1 cột nhãn số ở đầu và 784 cột điểm ảnh 28 × 28 phía sau).
*   **Tập Validation (`data_dev`)**: Lấy 1,000 mẫu đầu tiên để đánh giá mô hình. Ma trận được chuyển vị để thuận tiện tính toán.
    *   `Y_val`: Nhãn chữ số thực tế. **Shape**: `(1000,)`.
    *   `X_val`: Các ma trận điểm ảnh đã chuẩn hóa về dải [0, 1] (chia 255.0). **Shape**: `(784, 1000)`.
*   **Tập Huấn luyện (`data_train`)**: Lấy 41,000 mẫu còn lại để huấn luyện mô hình.
    *   `Y_train`: Nhãn chữ số thực tế. **Shape**: `(41000,)`.
    *   `X_train`: Ma trận ảnh đã chuẩn hóa về dải [0, 1]. **Shape**: `(784, 41000)`.

---

## II. CHI TIẾT TỪNG HÀM VÀ SHAPE ĐẦU RA TRONG CNN

### 1. Hàm khởi tạo `init_cnn_params()`
*   **Ý nghĩa / Vai trò**: Khởi tạo ngẫu nhiên các giá trị ban đầu cho các tham số (Trọng số và Độ chệch) của mạng tích chập CNN. Hàm này sử dụng phương pháp **He Initialization** (nhân với căn bậc hai của 2 chia cho số lượng kết nối đầu vào). Đây là phương pháp tối ưu giúp tránh hiện tượng triệt tiêu gradient (vanishing gradient) hoặc bùng nổ gradient (exploding gradient) khi huấn luyện các mạng sử dụng hàm kích hoạt ReLU.
*   **Cách hoạt động**:
    *   *Lớp Conv1:* Sử dụng `np.random.randn` để sinh ma trận ngẫu nhiên chuẩn hóa cho bộ lọc tích chập, sau đó nhân với hệ số điều chỉnh tỷ lệ $\sqrt{2 / (1 \times 3 \times 3)}$ để có trọng số `Wc1`. Độ chệch `bc1` được khởi tạo bằng 0.
    *   *Lớp Conv2:* Tương tự như Conv1 nhưng với số kênh đầu vào tăng lên 8, nhân với hệ số tỷ lệ $\sqrt{2 / (8 \times 3 \times 3)}$ để tạo trọng số `Wc2`. Độ chệch `bc2` khởi tạo bằng 0.
    *   *Lớp Dense:* Khởi tạo ma trận liên kết đầy đủ `Wd` kết nối từ 400 nơ-ron phẳng đến 10 nơ-ron đầu ra, nhân với hệ số tỷ lệ $\sqrt{2 / 400}$. Độ chệch `bd` khởi tạo bằng 0.
*   **Đầu vào**: Không có.
*   **Đầu ra (6 tham số)**:
    *   `Wc1` (Trọng số lớp Conv1): **Shape**: `(8, 1, 3, 3)` (8 bộ lọc, 1 kênh đầu vào, kích thước bộ lọc 3 × 3).
    *   `bc1` (Độ chệch lớp Conv1): **Shape**: `(8, 1)`.
    *   `Wc2` (Trọng số lớp Conv2): **Shape**: `(16, 8, 3, 3)` (16 bộ lọc, 8 kênh đầu vào, kích thước bộ lọc 3 × 3).
    *   `bc2` (Độ chệch lớp Conv2): **Shape**: `(16, 1)`.
    *   `Wd` (Trọng số lớp Dense): **Shape**: `(10, 400)` (10 lớp đầu ra, 400 nơ-ron đầu vào từ Flatten).
    *   `bd` (Độ chệch lớp Dense): **Shape**: `(10, 1)`.

### 2. Hàm kích hoạt `ReLU(Z)` & Đạo hàm `ReLU_deriv(Z)`
*   **Ý nghĩa / Vai trò**: Hàm phi tuyến tính giúp mạng học được các quan hệ phi tuyến tính phức tạp của dữ liệu. Nếu không có hàm kích hoạt phi tuyến tính, toàn bộ mạng sâu sẽ chỉ hoạt động tương đương với một bộ phân loại tuyến tính đơn giản.
*   **Cách hoạt động**:
    *   `ReLU(Z)`: Lọc ma trận `Z` đầu vào và gán toàn bộ các giá trị âm bằng 0. Các giá trị dương giữ nguyên. Công thức thực hiện bằng hàm `np.maximum(Z, 0)`.
    *   `ReLU_deriv(Z)`: Tính đạo hàm của ReLU để dùng cho lan truyền ngược. Hàm trả về `True` (tương đương giá trị 1) tại các vị trí có phần tử dương ($Z > 0$) và `False` (tương đương giá trị 0) tại các vị trí $\le 0$.
*   **Đầu vào**: Ma trận giá trị kích hoạt thô `Z` bất kỳ.
*   **Đầu ra**: 
    *   `ReLU(Z)`: Ma trận đã kích hoạt. **Shape**: Giữ nguyên shape của `Z`.
    *   `ReLU_deriv(Z)`: Ma trận Boolean đạo hàm. **Shape**: Giữ nguyên shape của `Z`.

### 3. Hàm phân phối xác suất `softmax(Z)`
*   **Ý nghĩa / Vai trò**: Chuyển đổi các giá trị điểm số thô (logits) ở đầu ra của lớp Dense thành phân phối xác suất của 10 lớp chữ số (từ 0 đến 9) sao cho tổng các xác suất bằng 1.
*   **Cách hoạt động**:
    *   Để tránh hiện tượng tràn số (overflow) khi tính hàm mũ của các số quá lớn (`np.exp`), hàm thực hiện trừ giá trị lớn nhất trong mỗi cột dữ liệu của ma trận `Z` (lưu vào biến `Z_shift`).
    *   Tính giá trị hàm mũ của tất cả các phần tử đã dịch chuyển, sau đó chia cho tổng giá trị hàm mũ của cột đó để thu được xác suất.
*   **Đầu vào**: Ma trận điểm số `Z` lớp Dense. **Shape**: `(10, batch_size)`.
*   **Đầu ra**: Ma trận xác suất dự đoán `A3`. **Shape**: `(10, batch_size)`.

### 4. Hàm mã hóa nhãn `one_hot(Y)`
*   **Ý nghĩa / Vai trò**: Chuyển đổi nhãn lớp thực tế dưới dạng số nguyên (ví dụ: `3`) thành một vector nhị phân 10 chiều (ví dụ: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`) để tính toán sai số Cross-Entropy dễ dàng trong lan truyền ngược.
*   **Cách hoạt động**:
    *   Khởi tạo một ma trận chứa toàn bộ số 0 có kích thước `(batch_size, 10)`.
    *   Sử dụng chỉ mục nâng cao của NumPy (`np.arange(Y.size)`) để điền giá trị 1 vào cột tương ứng với nhãn lớp của từng mẫu.
    *   Thực hiện chuyển vị ma trận đầu ra để khớp với ma trận xác suất `A3` khi thực hiện phép trừ tính sai số.
*   **Đầu vào**: Vector nhãn số nguyên `Y`. **Shape**: `(batch_size,)`.
*   **Đầu ra**: Ma trận nhãn mã hóa chuyển vị. **Shape**: `(10, batch_size)`.

### 5. Hàm biến đổi ma trận `im2col(X, fh, fw, stride=1, padding=0)`
*   **Ý nghĩa / Vai trò**: Đây là hàm bổ trợ tối quan trọng để tăng tốc độ tính toán lớp Tích chập. Phép toán tích chập truyền thống yêu cầu các vòng lặp `for` lồng nhau cực kỳ chậm trên Python. Hàm `im2col` giải quyết bằng cách lấy các vùng cục bộ trên ảnh sẽ được bộ lọc quét qua, sau đó duỗi phẳng và sắp xếp chúng thành các cột trong một ma trận lớn. Nhờ đó, tích chập được tính toán rất nhanh bằng một phép nhân ma trận duy nhất (phương pháp GEMM).
*   **Cách hoạt động**:
    *   Nếu có cấu hình `padding > 0`, hàm tiến hành đắp thêm các hàng/cột chứa giá trị 0 xung quanh ảnh.
    *   Sử dụng hàm hệ thống nâng cao `np.lib.stride_tricks.as_strided` để tạo ra một góc nhìn bộ nhớ (view) có cấu trúc đặc biệt mà không sao chép dữ liệu thật sang vùng nhớ mới, giúp tiết kiệm bộ nhớ tối đa.
*   **Đầu vào**:
    *   `X` (Ma trận ảnh hoặc bản đồ đặc trưng): **Shape**: `(batch_size, c, h, w)`.
    *   `fh`, `fw`: Kích thước bộ lọc quét (trong bài là `3`, `3`).
*   **Đầu ra**: Ma trận chập các vùng ảnh quét.
    *   **Shape**: `(batch_size, c, out_h, out_w, fh, fw)`
    *   *Trong đó*: `out_h = (h - fh + 2*padding) // stride + 1` và `out_w = (w - fw + 2*padding) // stride + 1`.

### 6. Hàm tích chập xuôi `conv2d_forward(X, W, b, stride=1, padding=0)`
*   **Ý nghĩa / Vai trò**: Thực hiện phép nhân chập trên không gian ma trận để trích xuất các đặc trưng hình ảnh cục bộ (nét vẽ, góc cạnh).
*   **Cách hoạt động**:
    *   Gọi hàm `im2col` trên ảnh đầu vào `X` để lấy ma trận các vùng quét `X_strided`.
    *   Sử dụng hàm nhân ma trận `np.tensordot` để nhân các vùng ảnh quét này với bộ lọc trọng số `W` dọc theo các trục đại diện cho số kênh đầu vào, chiều cao bộ lọc và chiều rộng bộ lọc.
    *   Chuyển vị kết quả thu được về định dạng tiêu chuẩn và cộng thêm vector độ chệch `b` (bias) sau khi đã được nắn chỉnh kích thước (reshape).
*   **Đầu vào**:
    *   `X`: **Shape**: `(batch_size, c_in, h_in, w_in)`.
    *   `W`: **Shape**: `(c_out, c_in, fh, fw)`.
    *   `b`: **Shape**: `(c_out, 1)`.
*   **Đầu ra (Tuple 2 phần tử)**:
    *   `Z` (Kết quả tích chập thô): **Shape**: `(batch_size, c_out, out_h, out_w)`.
    *   `X_strided` (Ảnh đã được duỗi để tái sử dụng ở bước tính đạo hàm lan truyền ngược): **Shape**: `(batch_size, c_in, out_h, out_w, fh, fw)`.

### 7. Hàm tích chập ngược `conv2d_backward(dZ, X_strided, W, X_shape=None)`
*   **Ý nghĩa / Vai trò**: Lan truyền đạo hàm lỗi từ các lớp phía sau đi ngược qua lớp tích chập hiện tại. Hàm tính toán gradient (độ dốc) sai số của trọng số `W` và bias `b` để chuẩn bị cập nhật trọng số, đồng thời tính gradient của ảnh đầu vào `dX` để truyền lỗi tiếp về các lớp trước đó.
*   **Cách hoạt động**:
    *   Tính toán gradient của trọng số `dW` bằng cách nhân `np.tensordot` giữa đạo hàm lỗi đầu ra `dZ` với ma trận ảnh đã được duỗi `X_strided`.
    *   Tính toán gradient của bias `db` bằng cách cộng tổng các lỗi `dZ` dọc theo toàn bộ các trục trừ trục kênh đầu ra.
    *   Nếu tham số `X_shape` được truyền vào (chỉ áp dụng từ lớp Conv2 ngược về trước, lớp Conv1 sát ảnh đầu vào không cần truyền tiếp lỗi về trước), hàm tính toán gradient lỗi của ảnh đầu vào `dX` thông qua việc nhân chập ngược giữa lỗi `dZ` và ma trận trọng số bộ lọc `W`.
*   **Đầu vào**:
    *   `dZ` (Đạo hàm lỗi từ lớp sau): **Shape**: `(batch_size, c_out, out_h, out_w)`.
    *   `X_strided`: Ma trận lưu trữ từ bước forward. **Shape**: `(batch_size, c_in, out_h, out_w, fh, fw)`.
    *   `W`: Ma trận trọng số lớp Conv. **Shape**: `(c_out, c_in, fh, fw)`.
    *   `X_shape` (Kích thước ảnh đầu vào của lớp khi forward): `(batch_size, c_in, h_in, w_in)` hoặc `None`.
*   **Đầu ra**:
    *   `dW` (Gradient trọng số): **Shape**: `(c_out, c_in, fh, fw)`.
    *   `db` (Gradient bias): **Shape**: `(c_out, 1)`.
    *   `dX` (Gradient đầu vào - chỉ trả về nếu `X_shape` khác `None`): **Shape**: `(batch_size, c_in, h_in, w_in)`.

### 8. Hàm gom cụm xuôi `maxpool_forward(X, pool_size=2)`
*   **Ý nghĩa / Vai trò**: Thu nhỏ chiều không gian của bản đồ đặc trưng, giúp giảm thiểu số lượng tham số tính toán của lớp tiếp theo, chống overfitting và tạo tính bất biến tịnh tiến cho mô hình.
*   **Cách hoạt động**:
    *   Xác định kích thước chiều cao và chiều rộng đầu ra bằng cách chia nguyên kích thước đầu vào cho `pool_size`.
    *   Cắt bỏ phần rìa thừa của ma trận `X` nếu kích thước không chia hết cho `pool_size`.
    *   Thực hiện reshape ma trận thành một mảng 6 chiều để tách các vùng không gian `2 × 2` riêng biệt.
    *   Sử dụng hàm `np.max` dọc theo trục thứ 3 và thứ 5 để lấy giá trị lớn nhất trong từng khối `2 × 2`.
*   **Đầu vào**: Bản đồ đặc trưng `X`. **Shape**: `(batch_size, c, h, w)`.
*   **Đầu ra**: Bản đồ đặc trưng sau pooling. **Shape**: `(batch_size, c, h // 2, w // 2)`.

### 9. Hàm gom cụm ngược `maxpool_backward(dZ, X, pool_size=2)`
*   **Ý nghĩa / Vai trò**: Lan truyền lỗi qua lớp Max Pooling. Vì trong quá trình lan truyền xuôi, ta chỉ giữ lại giá trị lớn nhất trong ô cửa sổ `2 × 2` và loại bỏ các giá trị khác, nên khi lan truyền ngược, sai số lỗi cũng chỉ được truyền ngược về đúng vị trí đã đóng góp giá trị lớn nhất đó (các vị trí khác nhận gradient lỗi bằng 0).
*   **Cách hoạt động**:
    *   Mở rộng kích thước lỗi `dZ` về kích thước ảnh ban đầu bằng cách nhân bản các phần tử thông qua `np.repeat`.
    *   Reshape ảnh đầu vào `X` thành dạng khối 6 chiều tương ứng để xác định giá trị lớn nhất trong từng ô cửa sổ.
    *   So sánh từng phần tử với giá trị Max để tạo ra mặt nạ Boolean `mask_sliced` (`True` tại vị trí đạt giá trị lớn nhất, `False` tại các vị trí còn lại).
    *   Nhân ma trận lỗi đã được mở rộng với mặt nạ Boolean để tạo ra gradient lỗi đầu ra `dX`.
*   **Đầu vào**:
    *   `dZ` (Đạo hàm lỗi từ lớp sau): **Shape**: `(batch_size, c, out_h, out_w)`.
    *   `X` (Bản đồ đặc trưng đầu vào trước khi pooling ở lượt forward): **Shape**: `(batch_size, c, h, w)`.
*   **Đầu ra**: `dX` (Gradient truyền ngược về trước). **Shape**: `(batch_size, c, h, w)`.

### 10. Hàm lan truyền xuôi toàn mạng CNN `cnn_forward_prop(...)`
*   **Ý nghĩa / Vai trò**: Kết nối tất cả các lớp của mạng CNN theo một đường ống tuần tự để tính toán xác suất dự đoán của ảnh đầu vào.
*   **Cách hoạt động**:
    *   Nhận ma trận ảnh phẳng `X` kích thước `(784, batch_size)`, chuyển vị và reshape về dạng tensor ảnh xám 2D `(batch_size, 1, 28, 28)`.
    *   Truyền dữ liệu qua: Conv1 $\rightarrow$ Kích hoạt ReLU1 $\rightarrow$ Max Pooling 1.
    *   Truyền tiếp kết quả qua: Conv2 $\rightarrow$ Kích hoạt ReLU2 $\rightarrow$ Max Pooling 2.
    *   Trải phẳng (Flatten) bản đồ đặc trưng thu được từ lớp Pooling 2 thành một ma trận đặc trưng 1D có kích thước `(400, batch_size)`.
    *   Đưa qua lớp kết nối đầy đủ (Dense) bằng cách nhân ma trận trọng số `Wd` cộng bias `bd`.
    *   Áp dụng hàm kích hoạt `softmax` để thu được ma trận phân phối xác suất dự đoán `A3`.
*   **Đầu vào**: `X` ảnh batch dạng phẳng. **Shape**: `(784, batch_size)`.
*   **Đầu ra (3 phần tử)**:
    *   `cache`: Tuple chứa các giá trị trung gian phục vụ lan truyền ngược: `(X1_strided, A1, A1_pool, X2_strided, A2, A2_pool, A2_flat)`.
        *   `A2_flat` (Vector trích xuất đặc trưng cuối cùng): **Shape**: `(400, batch_size)`.
    *   `Z3` (Điểm số đầu ra lớp Dense): **Shape**: `(10, batch_size)`.
    *   `A3` (Xác suất đầu ra Softmax): **Shape**: `(10, batch_size)`.

### 11. Hàm lan truyền ngược toàn mạng CNN `cnn_backward_prop(...)`
*   **Ý nghĩa / Vai trò**: Tính toán các gradient sai số cho toàn bộ tham số của mạng tích chập CNN từ cuối mạng về đầu mạng để làm cơ sở cập nhật trọng số.
*   **Cách hoạt động**:
    *   Tính sai số trực tiếp tại đầu ra của lớp Softmax: `dZ3 = A3 - one_hot(Y)`.
    *   Tính gradient sai số của lớp kết nối đầy đủ (Dense): `dWd` và `dbd`.
    *   Truyền lỗi qua lớp Dense ngược về lớp đặc trưng phẳng Flatten: `dA2_flat = Wd.T.dot(dZ3)`.
    *   Reshape ma trận đặc trưng phẳng thành tensor 2D `dA2_pool` để khớp kích thước đầu ra của lớp Pooling 2.
    *   Lan truyền ngược tuần tự qua: Pooling 2 ngược $\rightarrow$ ReLU 2 ngược $\rightarrow$ Conv 2 ngược $\rightarrow$ Pooling 1 ngược $\rightarrow$ ReLU 1 ngược $\rightarrow$ Conv 1 ngược để thu được toàn bộ các gradient trọng số bộ lọc.
*   **Đầu vào**:
    *   `cache`, `Z3`, `A3`: Các giá trị thu được từ lượt forward.
    *   `Y`: Vector nhãn thực tế của batch. **Shape**: `(batch_size,)`.
*   **Đầu ra (6 gradient)**:
    *   `dWc1` (Gradient trọng số Conv1): **Shape**: `(8, 1, 3, 3)`.
    *   `dbc1` (Gradient bias Conv1): **Shape**: `(8, 1)`.
    *   `dWc2` (Gradient trọng số Conv2): **Shape**: `(16, 8, 3, 3)`.
    *   `dbc2` (Gradient bias Conv2): **Shape**: `(16, 1)`.
    *   `dWd` (Gradient trọng số Dense): **Shape**: `(10, 400)`.
    *   `dbd` (Gradient bias Dense): **Shape**: `(10, 1)`.

### 12. Hàm cập nhật tham số `update_cnn_params(...)`
*   **Ý nghĩa / Vai trò**: Thực hiện bước tối ưu hóa trọng số dựa trên các gradient vừa tính được từ lan truyền ngược nhằm giảm thiểu sai số mất mát (Loss) của mạng CNN ở các lượt chạy sau.
*   **Cách hoạt động**: Sử dụng thuật toán Gradient Descent cơ bản. Lấy giá trị tham số cũ trừ đi tích của Tốc độ học (`alpha`) nhân với gradient lỗi tương ứng của tham số đó.
*   **Shape đầu ra**: Các ma trận đầu ra giữ nguyên kích thước như ban đầu.

---

## III. CHI TIẾT TỪNG PHẦN VÀ SHAPE ĐẦU RA TRONG SVM ĐA LỚP

Sau khi CNN huấn luyện xong, lớp Softmax/Dense bị loại bỏ. Ta đưa toàn bộ tập huấn luyện qua phần Conv-Pooling của CNN để trích xuất đặc trưng phẳng `train_features`.
*   `train_features`: **Shape**: `(41000, 400)`.

### 1. Lớp SVM nhị phân `LinearSVM_Binary`
Đại diện cho một bộ phân loại tuyến tính nhị phân dựa trên thuật toán tối đa hóa khoảng cách biên (Margin).
*   **Hàm `fit(X, y)`**:
    *   *Ý nghĩa*: Huấn luyện mô hình tìm kiếm siêu phẳng (định nghĩa bởi trọng số `w` và bias `b`) phân chia tối ưu lớp $+1$ và lớp $-1$.
    *   *Cách hoạt động*: Chạy qua số vòng lặp cấu hình sẵn (`n_iters`). Với mỗi vòng lặp, tính khoảng cách biên (margin) của tất cả dữ liệu mẫu. Phân loại các mẫu thành: mẫu phân loại đúng an toàn ($margin \ge 1$) và mẫu bị phân loại sai hoặc nằm sát biên nguy hiểm ($margin < 1$). Tiến hành tính gradient lỗi (kết hợp đạo hàm hàm lỗi Hinge Loss và thành phần phạt kiểm soát độ phức tạp mô hình L2 Regularization) rồi cập nhật trọng số `w` và bias `b`.
    *   *Shape đầu vào*: `X` shape `(41000, 400)`, `y` shape `(41000,)`.
    *   *Shape đầu ra*: Trọng số `self.w` shape `(400,)` và bias `self.b` (số thực).
*   **Hàm `get_score(X)`**:
    *   *Ý nghĩa*: Tính điểm số quyết định (Decision Score) của dữ liệu đối với siêu phẳng SVM. Điểm số dương lớn có nghĩa là mẫu nằm sâu trong vùng phân loại đúng của lớp $+1$.
    *   *Cách hoạt động*: Nhân vô hướng giữa ma trận đặc trưng `X` và vector trọng số `w`, sau đó trừ đi hằng số bias `b`.
    *   *Shape đầu vào*: `X` shape `(k, 400)`.
    *   *Shape đầu ra*: Vector điểm số. **Shape**: `(k,)`.

### 2. Lớp SVM đa lớp `MultiClassSVM_OVR`
Kết hợp 10 mô hình SVM nhị phân độc lập theo chiến lược Một-đối-Tất cả (One-vs-Rest / OVR) để phục vụ phân loại 10 lớp chữ số.
*   **Hàm `fit(X, y)`**:
    *   *Ý nghĩa*: Huấn luyện đồng thời cả 10 mô hình SVM nhị phân trên tập đặc trưng thu được từ CNN.
    *   *Cách hoạt động*: Duyệt qua từng lớp `c` từ 0 đến 9. Chuyển đổi nhãn lớp thực tế `y` thành nhãn nhị phân `y_binary` (+1 nếu nhãn bằng `c`, ngược lại bằng -1). Sau đó, gọi hàm `fit` của mô hình SVM nhị phân thứ `c` tương ứng để huấn luyện độc lập.
    *   *Shape đầu vào*: `X` shape `(41000, 400)`, `y` shape `(41000,)`.
    *   *Shape đầu ra*: Danh sách 10 mô hình SVM nhị phân đã được huấn luyện hoàn tất.

---

## IV. BẢNG TỔNG HỢP SHAPE DÒNG CHẢY DỮ LIỆU (FLOW SHAPES)

Bảng dưới đây tóm tắt cấu trúc hình dạng (shape) của dữ liệu khi đi qua các hàm chính trong một chu trình xử lý Mini-batch (giả sử kích thước batch học là B):

| Tên Hàm / Thành phần | Dữ liệu đầu vào (Shape) | Dữ liệu đầu ra (Shape) | Công thức / Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **Ảnh thô đầu vào** | `(784, B)` | `(B, 1, 28, 28)` | Reshape để chuẩn bị tích chập |
| **conv2d_forward (Conv1)** | `(B, 1, 28, 28)` | `(B, 8, 26, 26)` | Tích chập không padding với 8 bộ lọc 3 × 3 |
| **ReLU (Conv1)** | `(B, 8, 26, 26)` | `(B, 8, 26, 26)` | Loại bỏ giá trị âm |
| **maxpool_forward (Pool1)**| `(B, 8, 26, 26)` | `(B, 8, 13, 13)` | Gom cụm cửa sổ 2 × 2 |
| **conv2d_forward (Conv2)** | `(B, 8, 13, 13)` | `(B, 16, 11, 11)`| Tích chập không padding với 16 bộ lọc 3 × 3 |
| **ReLU (Conv2)** | `(B, 16, 11, 11)`| `(B, 16, 11, 11)`| Loại bỏ giá trị âm lần 2 |
| **maxpool_forward (Pool2)**| `(B, 16, 11, 11)`| `(B, 16, 5, 5)`  | Gom cụm cửa sổ 2 × 2 lần 2 |
| **Trải phẳng (Flatten)** | `(B, 16, 5, 5)`  | `(400, B)`       | Duỗi không gian 2D thành vector 1D |
| **Lớp Dense (Đầu ra CNN)**| `(400, B)`       | `(10, B)`        | Nhân trọng số Wd và cộng bias bd |
| **softmax (Dự đoán CNN)** | `(10, B)`        | `(10, B)`        | Tính phân phối xác suất |
| **one_hot (Nhãn thực tế)**| `(B,)`           | `(10, B)`        | Mã hóa One-hot phục vụ tính toán sai số |
| **Trích xuất Đặc trưng** | `(41000, 784)`   | `(41000, 400)`   | Toàn bộ tập huấn luyện sau khi bỏ lớp Dense |
| **SVM nhị phân thứ c** | `(41000, 400)`   | Trọng số `(400,)` | Tìm siêu phẳng phân tách lớp c với lớp khác |

---

## V. PHÂN TÍCH CHI TIẾT CÁC THAM SỐ VÀ SIÊU THAM SỐ (PARAMETERS & HYPERPARAMETERS ANALYSIS)

Các tham số trong mô hình được chia làm hai loại chính: **Các tham số học tập (Learnable Parameters)** và **Các siêu tham số (Hyperparameters)**.

### 1. Phân tích các Tham số Học tập (Learnable Parameters)
Đây là các tham số được mô hình tối ưu hóa tự động thông qua quá trình huấn luyện:

*   **Bộ Trích Xuất Đặc Trưng (Mạng CNN)**:
    *   *Lớp Tích chập 1 (Conv1)*:
        *   Trọng số `Wc1`: Kích thước `(8, 1, 3, 3)`. Gồm 8 × 1 × 3 × 3 = 72 tham số.
        *   Độ chệch `bc1`: Kích thước `(8, 1)`. Gồm 8 tham số.
        *   *Tổng số tham số Conv1*: 80 tham số.
    *   *Lớp Tích chập 2 (Conv2)*:
        *   Trọng số `Wc2`: Kích thước `(16, 8, 3, 3)`. Gồm 16 × 8 × 3 × 3 = 1152 tham số.
        *   Độ chệch `bc2`: Kích thước `(16, 1)`. Gồm 16 tham số.
        *   *Tổng số tham số Conv2*: 1168 tham số.
    *   *Lớp Kết Nối Đầy Đủ (Dense)* (chỉ dùng khi train CNN):
        *   Trọng số `Wd`: Kích thước `(10, 400)`. Gồm 10 × 400 = 4000 tham số.
        *   Độ chệch `bd`: Kích thước `(10, 1)`. Gồm 10 tham số.
        *   *Tổng số tham số lớp Dense*: 4010 tham số.

    *Tổng số tham số học tập của CNN trong giai đoạn huấn luyện là 5258 tham số. Khi lưu và chạy mô hình thực tế, lớp Dense được loại bỏ, chỉ giữ lại phần trích xuất đặc trưng tích chập với 1248 tham số.*

*   **Bộ Phân Loại Đa Lớp (SVM - OVR)**:
    *   Gồm 10 bộ phân loại SVM nhị phân độc lập (ứng với các số từ 0 đến 9).
    *   Mỗi bộ SVM nhị phân có:
        *   Trọng số `w`: Kích thước `(400,)`. Gồm 400 tham số.
        *   Độ chệch `b`: 1 số thực vô hướng.
    *   *Tổng số tham số học tập của SVM*: 10 × (400 + 1) = 4010 tham số.

*   **Tổng số tham số học tập của mô hình lai cuối cùng (CNN trích xuất + SVM phân loại)**:
    *   CNN (bỏ lớp Dense) + SVM = 1248 + 4010 = **5258** tham số.

### 2. Phân tích các Siêu tham số (Hyperparameters)
Đây là các tham số cấu hình tĩnh do người lập trình thiết lập để điều phối quá trình huấn luyện:

*   **Quá trình huấn luyện CNN**:
    *   *Tốc độ học (Learning Rate - `alpha = 0.1`)*: Quyết định biên độ thay đổi trọng số ở mỗi bước gradient descent. Giá trị 0.1 được chọn để đảm bảo tốc độ học nhanh nhưng không làm dao động mất ổn định.
    *   *Kích thước lô (Batch Size - `batch_size = 128`)*: Số lượng mẫu đưa vào xử lý đồng thời. Mini-batch 128 tối ưu tốc độ tính toán ma trận song song của CPU mà không gây tràn bộ nhớ RAM.
    *   *Số kỷ nguyên (Epochs/Iterations - `iterations = 20`)*: Số lượt duyệt qua toàn bộ tập dữ liệu. Với kích thước lô 128, 20 vòng lặp là đủ để mạng hội tụ.

*   **Quá trình huấn luyện SVM**:
    *   *Tốc độ học của SVM (`learning_rate = 0.5`)*: Điều chỉnh bước cập nhật biên phân chia cho thuật toán tối ưu SVM. Giá trị 0.5 giúp SVM hội tụ rất nhanh trên không gian đặc trưng phẳng.
    *   *Hằng số phạt Regularization L2 (`lambda_param = 0.001`)*: Đại lượng phạt các trọng số có giá trị quá lớn để kiểm soát độ phức tạp của mô hình, giúp tránh overfitting trên tập dữ liệu validation.
    *   *Số vòng lặp tối ưu SVM (`n_iters = 2000`)*: Số bước cập nhật trọng số trong thuật toán gradient descent của mỗi bộ SVM nhị phân để đảm bảo tìm thấy siêu phẳng phân tách tối ưu.


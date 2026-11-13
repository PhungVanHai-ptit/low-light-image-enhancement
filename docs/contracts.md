# HỢP ĐỒNG KỸ THUẬT VÀ QUY CHUẨN DỮ LIỆU (API CONTRACTS)
## DỰ ÁN LOW-LIGHT IMAGE ENHANCEMENT

> **Mục đích:** Tài liệu này đóng vai trò là "hợp đồng giao tiếp" (API contracts) bắt buộc giữa các thành viên TV1, TV2, và TV3. Mọi cài đặt thuật toán và module phải tuân thủ nghiêm ngặt 100% các quy định về kiểu dữ liệu, tham số và giá trị trả về trong văn bản này để đảm bảo việc ghép nối không xảy ra lỗi.

---

## 1. QUY CHUẨN DỮ LIỆU CHUNG (DATA CONVENTIONS)

1. **Kênh độ sáng $Y$ (Đầu vào / Đầu ra của các thuật toán):**
   * **Kiểu dữ liệu:** `numpy.ndarray`
   * **Số chiều (ndim):** Bắt buộc là mảng 2 chiều (`H x W`).
   * **Kiểu số (dtype):** `np.uint8`.
   * **Miền giá trị:** Nguyên trong đoạn $[0, 255]$.
   * **Tính bất biến (Immutability):** Không sửa đổi trực tiếp dữ liệu mảng đầu vào (`no in-place mutation`). Luôn trả về mảng kết quả mới có cùng kích thước `(H, W)` và kiểu `np.uint8`.

2. **Ảnh màu RGB:**
   * **Kiểu dữ liệu:** `numpy.ndarray`, shape `(H, W, 3)`, `dtype=np.uint8`, dải giá trị $[0, 255]$.
   * Thứ tự kênh màu: **RGB** (Red, Green, Blue) — phù hợp với Pillow và Tkinter.

3. **Kênh sắc độ $C_b, C_r$:**
   * **Kiểu dữ liệu:** `numpy.ndarray`, shape `(H, W)`, `dtype=np.float64`.
   * Được giữ nguyên trong quá trình tăng cường kênh sáng $Y$ nhằm bảo toàn màu sắc và tránh sai lệch màu khi khôi phục lại RGB.

---

## 2. QUY CHUẨN CHỮ KÝ HÀM THUẬT TOÁN (ALGORITHMS API)

Tất cả các thuật toán phải được cài đặt **from-scratch bằng NumPy** trong thư mục `source/algorithms/`, tuyệt đối không dùng các hàm xử lý ảnh có sẵn của OpenCV hay SciPy.

### 2.1. Gamma Correction (`source/algorithms/gamma.py`) — Phụ trách: TV1
* **Chữ ký hàm:**
  ```python
  def gamma_correct(y: np.ndarray, gamma: float = 0.6) -> np.ndarray:
      """
      Hiệu chỉnh gamma phi tuyến trên kênh sáng Y.
      
      Args:
          y: Mảng 2D np.ndarray (H, W), dtype uint8.
          gamma: Hệ số gamma (> 0). Với ảnh thiếu sáng, thường dùng gamma < 1.0 (mặc định 0.6).
          
      Returns:
          np.ndarray 2D (H, W), dtype uint8 sau khi áp dụng công thức:
          s = 255 * (r / 255) ** gamma (hoặc tối ưu qua bảng tra LUT 256 phần tử).
      """
  ```

### 2.2. Global Histogram Equalization (`source/algorithms/hist_equal.py`) — Phụ trách: TV2
* **Chữ ký hàm tính Histogram:**
  ```python
  def compute_histogram(y: np.ndarray) -> np.ndarray:
      """
      Tính lược đồ mức xám (histogram) của ảnh/kênh y.
      
      Args:
          y: Mảng 2D np.ndarray (H, W), dtype uint8.
          
      Returns:
          np.ndarray 1D kích thước (256,), dtype int64 chứa số lượng pixel của mỗi mức xám [0..255].
      """
  ```
* **Chữ ký hàm cân bằng lược đồ mức xám toàn cục:**
  ```python
  def equalize_hist(y: np.ndarray) -> np.ndarray:
      """
      Cân bằng lược đồ mức xám toàn cục (HE) sử dụng hàm phân phối tích lũy (CDF) và bảng tra (LUT).
      
      Args:
          y: Mảng 2D np.ndarray (H, W), dtype uint8.
          
      Returns:
          np.ndarray 2D (H, W), dtype uint8 đã được cân bằng độ tương phản.
          
      Yêu cầu đặc biệt:
          - Xử lý an toàn trường hợp ảnh hằng (constant image: cdf_max == cdf_min), tránh lỗi ZeroDivisionError.
      """
  ```

### 2.3. Contrast Limited Adaptive Histogram Equalization (`source/algorithms/clahe.py`) — Phụ trách: TV3
* **Chữ ký hàm:**
  ```python
  def clahe(y: np.ndarray, grid: tuple[int, int] = (8, 8), clip_factor: float = 2.0) -> np.ndarray:
      """
      Cân bằng lược đồ thích nghi giới hạn độ tương phản (CLAHE) kèm nội suy song tuyến tính.
      
      Args:
          y: Mảng 2D np.ndarray (H, W), dtype uint8.
          grid: (grid_y, grid_x) số ô lưới theo chiều dọc và ngang (mặc định (8, 8)).
          clip_factor: Hệ số cắt ngưỡng (clip limit multiplier, mặc định 2.0).
          
      Returns:
          np.ndarray 2D (H, W), dtype uint8.
          
      Quy trình:
          1. Chia ảnh thành các ô lưới (tiles).
          2. Tính histogram cho từng ô, cắt ngưỡng tại clip_limit và phân phối đều phần dư vào các bin.
          3. Tính mapping LUT (hoặc CDF chuẩn hóa) cho từng ô.
          4. Nội suy song tuyến tính (bilinear interpolation) giữa 4 tâm ô lân cận để loại bỏ viền ô (checkerboard).
      """
  ```

---

## 3. QUY CHUẨN CÁC MODULE DÙNG CHUNG (CORE MODULES)

### 3.1. Chuyển đổi màu sắc (`source/core/color.py`) — TV1
* **Chuẩn chuyển đổi:** ITU-R BT.601
* **Chữ ký:**
  ```python
  def rgb_to_ycbcr(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      """
      Chuyển đổi ảnh RGB (uint8) sang (Y, Cb, Cr).
      Y: uint8 [0, 255]
      Cb, Cr: float64
      """

  def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
      """
      Tái tạo ảnh RGB từ kênh Y (uint8) và Cb, Cr (float64).
      Có clipping trong khoảng [0, 255] và chuyển về uint8.
      """
  ```

### 3.2. Đọc / Ghi ảnh (`source/core/io.py`) — TV1
* **Chữ ký:**
  ```python
  def load_image(path: str | Path) -> np.ndarray:
      """
      Đọc ảnh từ đĩa (hỗ trợ JPG, PNG, WEBP, BMP...).
      Tự động xử lý xoay EXIF, gỡ bỏ alpha channel ghép trên nền trắng.
      Trả về uint8 mảng 2D (nếu là ảnh xám) hoặc 3D (H x W x 3 RGB).
      """

  def save_image(image: np.ndarray, path: str | Path) -> Path:
      """
      Lưu ảnh uint8 ra file. Tự tạo thư mục cha nếu chưa có.
      """
  ```

### 3.3. Pipeline điều phối (`source/core/pipeline.py`) — TV1
* **Chữ ký:**
  ```python
  def enhance(image: np.ndarray, method: str, params: dict | None = None) -> dict:
      """
      Pipeline hợp nhất xử lý ảnh màu hoặc ảnh xám với thuật toán được chọn.
      
      Args:
          image: np.ndarray uint8 (2D hoặc 3D RGB).
          method: 'gamma', 'he', hoặc 'clahe'.
          params: dict chứa các tham số phụ (ví dụ: {'gamma': 0.5}).
          
      Returns:
          dict: {
              "image": np.ndarray (ảnh kết quả cùng shape và dtype với input),
              "y_original": np.ndarray (kênh Y ban đầu),
              "y_result": np.ndarray (kênh Y sau tăng cường),
              "method": str,
              "params": dict,
              "algorithm_ms": float (thời gian chạy tính bằng milliseconds)
          }
      """
  ```

---

## 4. QUY TRÌNH KIỂM THỬ VÀ NGHIỆM THU (TESTING CONTRACT)

* Mỗi thành viên chịu trách nhiệm viết unit test tương ứng trong thư mục `tests/`:
  - `tests/test_color.py`: TV1
  - `tests/test_io.py`: TV1
  - `tests/test_gamma.py`: TV1
  - `tests/test_he.py`: TV2
  - `tests/test_clahe.py`: TV3
* Toàn bộ test phải chạy qua lệnh:
  ```bash
  python -m unittest discover -s tests
  ```


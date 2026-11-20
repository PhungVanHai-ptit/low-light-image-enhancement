# Cải thiện chất lượng ảnh thiếu sáng (Low-Light Image Enhancement)

Ứng dụng cải thiện chất lượng ảnh thiếu sáng cho môn học **Xử lý ảnh số (INT13146)** - Học viện Công nghệ Bưu chính Viễn thông (PTIT).

---

## 1. Thông tin thành viên và phân công nhiệm vụ

| Thành viên | Mã sinh viên | Nhiệm vụ chuyên môn | Đóng góp |
|---|:---:|---|:---:|
| **Phùng Văn Hải** | B23DCCN281 | **TV1** — Gamma Correction; chuyển đổi màu; đọc/lưu ảnh; pipeline; giao diện và tích hợp mã nguồn. | 1/3 |
| **Vũ Văn Hiếu** | B23DCCN316 | **TV2** — Histogram Equalization; histogram và CDF; dữ liệu; đối chứng OpenCV; thống kê và phân tích kết quả. | 1/3 |
| **Khương Bá Duy** | B23DCCN238 | **TV3** — CLAHE; chia ô; cắt và phân phối histogram; nội suy song tuyến tính; kiểm thử thuật toán CLAHE. | 1/3 |

---

## 2. Giới thiệu dự án

Dự án tập trung vào việc **tự cài đặt từ đầu (from scratch)** các thuật toán nâng cao chất lượng ảnh chụp thiếu sáng bằng **NumPy**, không sử dụng các hàm xử lý ảnh có sẵn của thư viện. Thư viện OpenCV chỉ được sử dụng cho mục đích **đối chứng và đánh giá định lượng**.

### Các thuật toán chính:
1. **Gamma Correction** (Hiệu chỉnh Gamma phi tuyến qua bảng tra LUT).
2. **Global Histogram Equalization (HE)** (Cân bằng lược đồ mức xám toàn cục qua hàm phân phối tích lũy CDF và bảng tra LUT).
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** (Cân bằng lược đồ thích nghi giới hạn độ tương phản, chia ô lưới, cắt ngưỡng phân phối lại phần dư và nội suy song tuyến tính).

---

## 3. Tính năng nổi bật

* **Bảo toàn màu sắc:** Chuyển đổi ảnh sang không gian màu YCbCr theo tiêu chuẩn ITU-R BT.601; chỉ tăng cường kênh độ sáng Y, giữ nguyên hai kênh màu Cb, Cr để tránh sai lệch màu.
* **Giao diện người dùng Desktop (Tkinter):**
  * Tùy chỉnh tham số thanh trượt theo thời gian thực (Real-time slider).
  * Chế độ nhấn giữ chuột so sánh trực tiếp Trước / Sau (Before / After).
  * Tích hợp biểu đồ trực quan hóa lược đồ mức xám (Matplotlib Histogram 256 mức).
  * Chế độ tổng quan so sánh đồng thời 4 khung ảnh (Ảnh gốc, Gamma, HE, CLAHE).
  * Tích hợp thanh cuộn (Scrollbar) và con lăn chuột tự động khi kích thước ảnh phóng to, bố cục tự động lấp đầy màn hình khi thu nhỏ.
  * Tab đối chứng trực tiếp với thư viện OpenCV kèm phân tích sai số MAE thời gian thực cho cả 3 thuật toán (Gamma, HE, CLAHE).
* **Đánh giá và đối chứng định lượng:** Đo thời gian thực thi (ms) và đối chứng kết quả trực tiếp với các hàm tương ứng của OpenCV (`cv2.LUT`, `cv2.equalizeHist`, `cv2.createCLAHE`) qua sai số tuyệt đối trung bình (MAE).

---

## 4. Cấu trúc thư mục dự án

```text
low-light-image-enhancement/
├── docs/                     # Tài liệu thiết kế, quy ước API (contracts.md, kế hoạch chi tiết)
├── data/
│   ├── input/                # Tập ảnh thử nghiệm (sample_dark.png, ...)
│   └── manifest.csv          # Danh sách metadata bộ ảnh thử nghiệm
├── results/                  # Kết quả đo đạc benchmark (benchmark.csv)
├── source/
│   ├── algorithms/           # Cài đặt thuật toán from-scratch bằng NumPy
│   │   ├── gamma.py          # Gamma Correction (TV1)
│   │   ├── hist_equal.py     # Global Histogram Equalization (TV2)
│   │   └── clahe.py          # CLAHE with Bilinear Interpolation (TV3)
│   ├── core/                 # Module lõi dùng chung
│   │   ├── color.py          # Chuyển đổi RGB <-> YCbCr BT.601 (TV1)
│   │   ├── io.py             # Đọc/ghi ảnh chuẩn hóa (TV1)
│   │   └── pipeline.py       # Pipeline điều phối thuật toán và đo thời gian (TV1)
│   ├── evaluation/           # Đối chứng OpenCV và đo đạc số liệu (TV2)
│   │   ├── metrics.py        # Tính Mean, Std, White ratio
│   │   └── benchmark.py      # Đo tốc độ, tính MAE với OpenCV, xuất CSV
│   ├── ui/                   # Giao diện người dùng Tkinter (TV1)
│   │   └── main_window.py
│   └── app.py                # Điểm khởi chạy ứng dụng (GUI / CLI)
├── tests/                    # 31 Unit tests tự động kiểm thử toàn diện
│   ├── test_color.py
│   ├── test_io.py
│   ├── test_gamma.py
│   ├── test_he.py
│   └── test_clahe.py
├── run_app.bat               # Script khởi động ứng dụng trên Windows
├── run_tests.bat             # Script chạy kiểm thử unit tests trên Windows
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Cài đặt và chạy ứng dụng

### 5.1. Yêu cầu môi trường
* Python 3.10 trở lên (khuyên dùng Python 3.11 hoặc 3.12)

### 5.2. Thiết lập môi trường ảo và cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ trợ
pip install -r requirements.txt
```

### 5.3. Chạy ứng dụng giao diện (GUI)
* Cách 1: Nhấp đúp vào file `run_app.bat` trên Windows.  
* Cách 2: Chạy lệnh terminal:
```bash
python -m source.app
```

### 5.4. Chạy kiểm thử tự động (Unit Tests)
* Cách 1: Nhấp đúp vào file `run_tests.bat` trên Windows.  
* Cách 2: Chạy lệnh terminal:
```bash
python -m unittest discover -s tests -v
```

### 5.5. Chạy module Benchmark đối chứng với OpenCV
Chạy lệnh xuất bảng dữ liệu đối chứng ra file `results/benchmark.csv`:
```bash
python -c "from source.evaluation.benchmark import run_benchmark; run_benchmark(['data/input/sample_dark.png'], 'results/benchmark.csv')"
```

---

## 6. Kết quả đối chứng thực nghiệm (Trên ảnh mẫu sample_dark.png)

| Thuật toán | Thời gian thực thi (From-scratch) | Thời gian OpenCV | Sai số MAE vs OpenCV | Độ sáng trung bình (Trước -> Sau) | Độ tương phản (Std) |
|---|---|---|---|---|---|
| **Gamma (gamma=0.6)** | ~0.65 ms | ~0.27 ms | **0.0000** | 33.47 -> 73.80 | 13.33 -> 18.79 |
| **Global HE** | ~1.63 ms | ~0.27 ms | **0.0000** | 33.47 -> 124.89 | 13.33 -> 76.66 |
| **CLAHE (8x8, clip=2.0)** | ~19.85 ms | ~11.58 ms | **8.0525** | 33.47 -> 44.38 | 13.33 -> 16.96 |

> **Nhận xét:**
> - Thuật toán **Gamma** và **Global HE** tự cài đặt đạt độ chính xác tuyệt đối (MAE = 0.0000) so với các hàm tương ứng của OpenCV (`cv2.LUT` và `cv2.equalizeHist`).
> - Thuật toán **CLAHE** tự viết kèm nội suy song tuyến tính giúp dải sáng tự nhiên, hạn chế hiện tượng phóng đại nhiễu và vỡ mảng so với HE toàn cục.

# Low-Light Image Enhancement

Ứng dụng cải thiện chất lượng ảnh thiếu sáng (Low-Light Image Enhancement) cho môn học **Xử lý ảnh số (INT13146)** - Học viện Công nghệ Bưu chính Viễn thông (PTIT).

---

##  Giới thiệu dự án

Dự án tập trung vào việc **tự cài đặt từ đầu (from scratch)** các thuật toán nâng cao chất lượng ảnh bằng **NumPy**, không sử dụng các hàm xử lý ảnh có sẵn của thư viện. Thư viện OpenCV chỉ được sử dụng cho mục đích **đối chứng và đánh giá định lượng**.

### Các thuật toán chính:
1. **Gamma Correction** (Hiệu chỉnh Gamma phi tuyến).
2. **Global Histogram Equalization (HE)** (Cân bằng lược đồ mức xám toàn cục).
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** (Cân bằng lược đồ thích nghi giới hạn độ tương phản với nội suy song tuyến tính).

---

##  Tính năng nổi bật

* **Bảo toàn màu sắc:** Chuyển đổi ảnh sang không gian màu $YC_bC_r$ bằng công thức chuẩn ITU-R BT.601; chỉ tăng cường kênh sáng $Y$, giữ nguyên hai kênh màu $C_b, C_r$ để tránh sai lệch màu.
* **Giao diện trực quan (GUI):** Cho phép tải ảnh, tùy chỉnh tham số theo thời gian thực, xem ảnh trước/sau và biểu đồ histogram song song.
* **Đánh giá & Đối chứng:** Đo thời gian thực thi và đối chứng kết quả trực tiếp với các hàm tương ứng của OpenCV (`cv2.equalizeHist`, `cv2.createCLAHE`).

---

##  Cấu trúc thư mục dự kiến

```text
low-light-image-enhancement/
├── docs/                     # Tài liệu thiết kế, quy ước API (contracts.md)
├── data/                     # Tập ảnh thử nghiệm (indoor, outdoor, difficult)
├── results/                  # Kết quả xử lý, biểu đồ và CSV đánh giá
├── source/
│   ├── algorithms/           # Cài đặt thuật toán from-scratch
│   │   ├── gamma.py          # Gamma Correction (TV1)
│   │   ├── hist_equal.py     # Histogram Equalization (TV2)
│   │   └── clahe.py          # CLAHE (TV3)
│   ├── core/                 # Xử lý màu sắc, I/O và pipeline điều phối (TV1)
│   │   ├── color.py
│   │   ├── io.py
│   │   └── pipeline.py
│   ├── evaluation/           # Đối chứng OpenCV & đo đạc số liệu (TV2)
│   ├── ui/                   # Giao diện người dùng Tkinter (TV1)
│   └── app.py                # Điểm khởi chạy ứng dụng
├── tests/                    # Unit tests kiểm tra tính đúng thuật toán
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Cài đặt & Chạy ứng dụng

### 1. Yêu cầu môi trường
* Python 3.10+ (khuyên dùng Python 3.11)

### 2. Thiết lập môi trường ảo
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ trợ
pip install -r requirements.txt
```

### 3. Chạy ứng dụng giao diện
```bash
python -m source.app
```

### 4. Chạy kiểm thử tự động
```bash
python -m unittest discover -s tests
```

---

##  Phân công nhiệm vụ (Nhóm 3 thành viên)

* **TV1 (Lead & Tích hợp):** Thuật toán Gamma, xử lý không gian màu $YC_bC_r$, pipeline điều phối và giao diện ứng dụng (GUI).
* **TV2 (Đánh giá & Dữ liệu):** Thuật toán Histogram Equalization (HE), bộ dữ liệu thử nghiệm, module đối chứng OpenCV và phân tích số liệu.
* **TV3 (Thuật toán CLAHE):** Cài đặt thuật toán CLAHE (chia ô, clipping, phân phối lại phần dư và nội suy song tuyến tính).

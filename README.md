# Low-Light Image Enhancement

Ung dung cai thien chat luong anh thieu sang (Low-Light Image Enhancement) cho mon hoc **Xu ly anh so (INT13146)** - Hoc vien Cong nghe Buu chinh Vien thong (PTIT).

---

## 1. Gioi thieu du an

Du an tap trung vao viec **tu cai dat tu dau (from scratch)** cac thuat toan nang cao chat luong anh bang **NumPy**, khong su dung cac ham xu ly anh co san cua thu vien. Thu vien OpenCV chi duoc su dung cho muc dich **doi chung va danh gia dinh luong**.

### Cac thuat toan chinh:
1. **Gamma Correction** (Hieu chinh Gamma phi tuyen qua bang tra LUT).
2. **Global Histogram Equalization (HE)** (Can bang luoc do muc xam toan cuc qua ham phan phoi tich luy CDF va bang tra LUT).
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** (Can bang luoc do thich nghi gioi han do tuong phan, chia o luoi, cat nguong phan phoi lai phan du va noi suy song tuyen tinh).

---

## 2. Tinh nang noi bat

* **Bao toan mau sac:** Chuyen doi anh sang khong gian mau YCbCr theo tieu chuan ITU-R BT.601; chi tang cuong kenh do sang Y, giu nguyen hai kenh mau Cb, Cr de tranh sai lech mau.
* **Giao dien nguoi dung (Desktop GUI - Tkinter):**
  * Tuy chinh tham so thanh truot theo thoi gian thuc (Real-time slider).
  * Che do nhan giu chuot so sanh truc tiep Truoc / Sau (Before / After).
  * Tich hop bieu do truc quan hoa luoc do muc xam (Matplotlib Histogram 256 muc).
  * Che do tong quan so sanh dong thoi 4 khung anh (Anh goc, Gamma, HE, CLAHE).
* **Danh gia va doi chung dinh luong:** Do thoi gian thuc thi (ms) va doi chung ket qua truc tiep voi cac ham tuong ung cua OpenCV (`cv2.equalizeHist`, `cv2.createCLAHE`) qua sai so tuyet doi trung binh (MAE).

---

## 3. Cau truc thu muc du an

```text
low-light-image-enhancement/
├── docs/                     # Tai lieu thiet ke, quy uoc API (contracts.md, ke hoach chi tiet)
├── data/
│   ├── input/                # Tap anh thu nghiem (sample_dark.png, ...)
│   └── manifest.csv          # Danh sach metadata bo anh thu nghiem
├── results/                  # Ket qua do dac benchmark (benchmark.csv)
├── source/
│   ├── algorithms/           # Cai dat thuat toan from-scratch bang NumPy
│   │   ├── gamma.py          # Gamma Correction (TV1)
│   │   ├── hist_equal.py     # Global Histogram Equalization (TV2)
│   │   └── clahe.py          # CLAHE with Bilinear Interpolation (TV3)
│   ├── core/                 # Module loi dung chung
│   │   ├── color.py          # Chuyen doi RGB <-> YCbCr BT.601 (TV1)
│   │   ├── io.py             # Doc/ghi anh chuan hoa (TV1)
│   │   └── pipeline.py       # Pipeline dieu phoi thuat toan va do thoi gian (TV1)
│   ├── evaluation/           # Doi chung OpenCV va do dac so lieu (TV2)
│   │   ├── metrics.py        # Tinh Mean, Std, White ratio
│   │   └── benchmark.py      # Do toc do, tinh MAE voi OpenCV, xuat CSV
│   ├── ui/                   # Giao dien nguoi dung Tkinter (TV1)
│   │   └── main_window.py
│   └── app.py                # Diem khoi chay ung dung (GUI / CLI)
├── tests/                    # 31 Unit tests tu dong kiem thu toan dien
│   ├── test_color.py
│   ├── test_io.py
│   ├── test_gamma.py
│   ├── test_he.py
│   └── test_clahe.py
├── run_app.bat               # Script khoi dong ung dung tren Windows
├── run_tests.bat             # Script chay kiem thu unit tests tren Windows
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Cai dat va chay ung dung

### 4.1. Yeu cau moi truong
* Python 3.10 tro len (khuyen dung Python 3.11 hoac 3.12)

### 4.2. Thiet lap moi truong ao va cai dat thu vien
```bash
# Tao moi truong ao
python -m venv .venv

# Kich hoat moi truong ao (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cai dat thu vien phu tro
pip install -r requirements.txt
```

### 4.3. Chay ung dung giao dien (GUI)
* Cach 1: Nhap dup vao file `run_app.bat` tren Windows.  
* Cach 2: Chay lenh terminal:
```bash
python -m source.app
```

### 4.4. Chay kiem thu tu dong (Unit Tests)
* Cach 1: Nhap dup vao file `run_tests.bat` tren Windows.  
* Cach 2: Chay lenh terminal:
```bash
python -m unittest discover -s tests -v
```

### 4.5. Chay module Benchmark doi chung voi OpenCV
Chay lenh xuat bang du lieu doi chung ra file `results/benchmark.csv`:
```bash
python -c "from source.evaluation.benchmark import run_benchmark; run_benchmark(['data/input/sample_dark.png'], 'results/benchmark.csv')"
```

---

## 5. Ket qua doi chung thuc nghiem (Tren anh mau sample_dark.png)

| Thuat toan | Thoi gian thuc thi (From-scratch) | Thoi gian OpenCV | Sai so MAE vs OpenCV | Do sang trung binh (Truoc -> Sau) | Do tuong phan (Std) |
|---|---|---|---|---|---|
| **Gamma (gamma=0.6)** | ~0.65 ms | N/A | N/A | 33.47 -> 73.80 | 13.33 -> 18.79 |
| **Global HE** | ~1.63 ms | ~0.27 ms | **0.0000** | 33.47 -> 124.89 | 13.33 -> 76.66 |
| **CLAHE (8x8, clip=2.0)** | ~19.85 ms | ~11.58 ms | **8.0525** | 33.47 -> 44.38 | 13.33 -> 16.96 |

> **Nhan xet:**
> - Thuat toan **Global HE** tu cai dat dat do chinh xac tuyet doi (MAE = 0.0000) so voi ham `cv2.equalizeHist`.
> - Thuat toan **CLAHE** tu viet kem noi suy song tuyen tinh cho anh sang tu nhien, han che hien tuong phong dai nhieu va vo mang so voi HE toan cuc.

---

## 6. Phan cong nhiem vu (Nhom 3 thanh vien)

* **TV1 (Lead va Tich hop):** Thuat toan Gamma Correction, xu ly khong gian mau YCbCr, pipeline dieu phoi va giao dien Desktop GUI (Tkinter nhung Matplotlib).
* **TV2 (Danh gia va Du lieu):** Thuat toan Global Histogram Equalization (HE), bo du lieu thu nghiem, module doi chung OpenCV va phan tich so lieu (MAE, thoi gian).
* **TV3 (Thuat toan CLAHE):** Cai dat thuat toan CLAHE from-scratch (chia luoi contextual regions, cat nguong clipping, phan phoi lai phan du va noi suy song tuyen tinh bilinear).

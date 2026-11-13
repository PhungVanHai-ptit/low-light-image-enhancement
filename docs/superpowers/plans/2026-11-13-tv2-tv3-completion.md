# TV2 + TV3 Completion & App Finalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện toàn bộ phần việc TV2 (Histogram Equalization + Evaluation + Dataset) và TV3 (CLAHE), tích hợp vào app sẵn có để ứng dụng chạy đầy đủ 3 thuật toán thật (không còn mock).

**Architecture:** TV2 cài đặt HE from-scratch bằng NumPy (histogram → CDF → LUT), kèm module evaluation đối chứng OpenCV; TV3 cài đặt CLAHE from-scratch (chia ô → histogram cục bộ + clipping → nội suy song tuyến tính). Cả hai tuân thủ API contract trong `docs/contracts.md`: nhận mảng 2D uint8, trả mảng 2D uint8. Pipeline và UI của TV1 đã sẵn sàng hook vào — chỉ cần file đúng tên và đúng chữ ký.

**Tech Stack:** Python 3.11, NumPy, Pillow, Matplotlib, Tkinter, OpenCV (chỉ trong `evaluation/`)

**Spec:** `docs/contracts.md` (chữ ký hàm bắt buộc) + `local/ke_hoach_tong_quan.md` (phân công & yêu cầu)

## Global Constraints

- Thuật toán phải **from-scratch bằng NumPy** — không dùng `cv2.equalizeHist`, `cv2.createCLAHE`, `scipy` hay bất kỳ hàm xử lý ảnh có sẵn nào cho phần cài đặt chính.
- OpenCV **chỉ** được phép dùng trong thư mục `source/evaluation/` cho mục đích đối chứng.
- Đầu vào/đầu ra thuật toán: `np.ndarray` 2D `(H, W)`, `dtype=np.uint8`, dải `[0, 255]`.
- Không sửa mảng đầu vào (no in-place mutation) — luôn trả mảng mới.
- Test chạy bằng: `python -m unittest discover -s tests`

## Review Focus

1. **Ảnh hằng (constant image):** HE với ảnh toàn pixel = 128 phải trả ảnh giống hệt (không crash chia 0).
2. **Ảnh 1 pixel hoặc rất nhỏ:** CLAHE với ảnh nhỏ hơn grid size (ví dụ 3×3 với grid 8×8) phải trả kết quả hợp lệ, không crash.
3. **Ảnh kích thước không chia hết grid:** CLAHE phải padding/xử lý biên đúng khi H hoặc W không chia hết cho grid_y hoặc grid_x.
4. **Overflow/clipping:** Với ảnh toàn pixel = 255, gamma < 1 hoặc HE phải giữ giá trị ≤ 255 (uint8 không tràn).
5. **CLAHE clip_factor = 0 hoặc rất lớn:** Kết quả phải ổn định — clip_factor → ∞ phải cho kết quả gần giống HE cục bộ thông thường.

---

### Task 1: Histogram Equalization (TV2 — `hist_equal.py` + `test_he.py`)

**Files:**
- Create: `source/algorithms/hist_equal.py`
- Create: `tests/test_he.py`

**Interfaces:**
- Consumes: Không phụ thuộc task khác — nhận mảng 2D uint8 trực tiếp.
- Produces:
  - `compute_histogram(y: np.ndarray) -> np.ndarray` — trả 1D `(256,)` int64
  - `equalize_hist(y: np.ndarray) -> np.ndarray` — trả 2D `(H, W)` uint8

- [ ] **Step 1: Write failing tests in `tests/test_he.py`**

```python
import unittest
import numpy as np

class TestComputeHistogram(unittest.TestCase):
    def test_uniform_image(self):
        """Ảnh hằng (tất cả pixel = 100): histogram phải có đúng 1 bin khác 0."""
        from source.algorithms.hist_equal import compute_histogram
        y = np.full((10, 10), 100, dtype=np.uint8)
        hist = compute_histogram(y)
        self.assertEqual(hist.shape, (256,))
        self.assertEqual(hist.dtype, np.int64)
        self.assertEqual(hist[100], 100)
        self.assertEqual(hist.sum(), 100)

    def test_known_distribution(self):
        """Ảnh 2x2 với 4 giá trị khác nhau."""
        from source.algorithms.hist_equal import compute_histogram
        y = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        hist = compute_histogram(y)
        for i in range(4):
            self.assertEqual(hist[i], 1)
        self.assertEqual(hist[4:].sum(), 0)

class TestEqualizeHist(unittest.TestCase):
    def test_output_shape_dtype(self):
        from source.algorithms.hist_equal import equalize_hist
        y = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = equalize_hist(y)
        self.assertEqual(result.shape, y.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_constant_image_safe(self):
        """Ảnh hằng phải trả ảnh không đổi, không crash chia 0."""
        from source.algorithms.hist_equal import equalize_hist
        y = np.full((50, 50), 128, dtype=np.uint8)
        result = equalize_hist(y)
        np.testing.assert_array_equal(result, y)

    def test_no_inplace_mutation(self):
        from source.algorithms.hist_equal import equalize_hist
        y = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        y_copy = y.copy()
        equalize_hist(y)
        np.testing.assert_array_equal(y, y_copy)

    def test_contrast_increases(self):
        """Ảnh thiếu sáng (pixel tập trung vùng tối) -> sau HE std phải tăng."""
        from source.algorithms.hist_equal import equalize_hist
        y = np.random.randint(10, 60, (100, 100), dtype=np.uint8)
        result = equalize_hist(y)
        self.assertGreater(float(np.std(result)), float(np.std(y)))

    def test_all_white_image(self):
        """Ảnh toàn trắng phải giữ nguyên."""
        from source.algorithms.hist_equal import equalize_hist
        y = np.full((20, 20), 255, dtype=np.uint8)
        result = equalize_hist(y)
        np.testing.assert_array_equal(result, y)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_he -v`
Expected: FAIL / ERROR with `ImportError` (module `hist_equal` does not exist yet)

- [ ] **Step 3: Implement `compute_histogram` and `equalize_hist` in `source/algorithms/hist_equal.py`**

`compute_histogram`: Duyệt 256 bin, đếm tần suất bằng `np.bincount(y.ravel(), minlength=256).astype(np.int64)`.

`equalize_hist`:
1. Gọi `compute_histogram` lấy histogram 256 bin.
2. Tính CDF tích lũy: `cdf = np.cumsum(hist)`.
3. Tìm `cdf_min` = giá trị CDF khác 0 đầu tiên. Nếu `cdf_min == cdf[-1]` (ảnh hằng) → trả ảnh gốc copy.
4. Tính LUT 256 phần tử: `lut[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)`, clip `[0, 255]`, cast `uint8`.
5. Ánh xạ `return lut[y]`.

Validation đầu vào: kiểm tra `ndim == 2`, `dtype == uint8` — raise `ValueError` nếu sai.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_he -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Verify integration with pipeline (pipeline đã có dynamic import cho `hist_equal`)**

Run: `python -c "import numpy as np; from source.core.pipeline import enhance; r = enhance(np.random.randint(10,60,(100,100,3),dtype=np.uint8), 'he'); print('is_mock:', r['is_mock'], 'mean:', r['image'].mean():.1f)"`
Expected: `is_mock: False`

- [ ] **Step 6: Commit**

```bash
git add source/algorithms/hist_equal.py tests/test_he.py
git commit -m "feat(tv2): implement Global Histogram Equalization from scratch"
```

---

### Task 2: CLAHE Algorithm (TV3 — `clahe.py` + `test_clahe.py`)

**Files:**
- Create: `source/algorithms/clahe.py`
- Create: `tests/test_clahe.py`

**Interfaces:**
- Consumes: Không phụ thuộc task khác.
- Produces:
  - `clahe(y: np.ndarray, grid: tuple[int, int] = (8, 8), clip_factor: float = 2.0) -> np.ndarray` — trả 2D `(H, W)` uint8

- [ ] **Step 1: Write failing tests in `tests/test_clahe.py`**

```python
import unittest
import numpy as np

class TestCLAHE(unittest.TestCase):
    def test_output_shape_dtype(self):
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = clahe(y)
        self.assertEqual(result.shape, y.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_constant_image(self):
        """Ảnh hằng -> kết quả phải giống ảnh gốc."""
        from source.algorithms.clahe import clahe
        y = np.full((64, 64), 128, dtype=np.uint8)
        result = clahe(y)
        np.testing.assert_array_equal(result, y)

    def test_no_inplace_mutation(self):
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        y_copy = y.copy()
        clahe(y)
        np.testing.assert_array_equal(y, y_copy)

    def test_non_divisible_size(self):
        """Ảnh có kích thước không chia hết cho grid vẫn phải chạy đúng."""
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (67, 53), dtype=np.uint8)
        result = clahe(y, grid=(8, 8))
        self.assertEqual(result.shape, (67, 53))
        self.assertEqual(result.dtype, np.uint8)

    def test_small_image(self):
        """Ảnh rất nhỏ (< grid size) phải trả kết quả hợp lệ."""
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (3, 3), dtype=np.uint8)
        result = clahe(y, grid=(8, 8))
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_contrast_enhancement(self):
        """Ảnh thiếu sáng -> CLAHE phải tăng độ tương phản (std tăng)."""
        from source.algorithms.clahe import clahe
        y = np.random.randint(10, 60, (128, 128), dtype=np.uint8)
        result = clahe(y, grid=(8, 8), clip_factor=2.0)
        self.assertGreater(float(np.std(result)), float(np.std(y)))

    def test_different_grid_sizes(self):
        """Chạy được với nhiều kích thước grid khác nhau."""
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        for grid in [(4, 4), (8, 8), (16, 16)]:
            result = clahe(y, grid=grid)
            self.assertEqual(result.shape, y.shape)

    def test_validation_2d_uint8(self):
        from source.algorithms.clahe import clahe
        with self.assertRaises(ValueError):
            clahe(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            clahe(np.zeros((10, 10), dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_clahe -v`
Expected: FAIL / ERROR with `ImportError`

- [ ] **Step 3: Implement `clahe` in `source/algorithms/clahe.py`**

Thuật toán CLAHE from-scratch gồm 4 giai đoạn:

**Giai đoạn 1 — Chia ô & Padding:**
- Tính `tile_h = ceil(H / grid_y)`, `tile_w = ceil(W / grid_x)`.
- Pad ảnh bằng `np.pad(..., mode='reflect')` để kích thước chia hết: `padded_h = tile_h * grid_y`, `padded_w = tile_w * grid_x`.

**Giai đoạn 2 — Histogram cục bộ + Clipping + LUT cho từng ô:**
- Với mỗi ô `(r, c)` trong lưới `grid_y × grid_x`:
  - Tính histogram 256 bin bằng `np.bincount`.
  - `clip_limit = max(1, int(clip_factor * tile_h * tile_w / 256))`.
  - Clip: cắt bin nào vượt `clip_limit`, cộng dồn phần dư, chia đều phần dư vào tất cả 256 bin (lặp cho đến khi hết dư).
  - Tính CDF → chuẩn hóa thành mapping LUT 256 phần tử `uint8`.
- Lưu tất cả LUT vào mảng `mappings[grid_y][grid_x][256]`.

**Giai đoạn 3 — Tính tọa độ tâm ô:**
- Tâm ô `(r, c)` = `(r * tile_h + tile_h // 2, c * tile_w + tile_w // 2)`.

**Giai đoạn 4 — Nội suy song tuyến tính (bilinear interpolation):**
- Với mỗi pixel `(py, px)` trên ảnh đã pad:
  - Xác định 4 tâm ô lân cận (top-left, top-right, bottom-left, bottom-right) dựa trên vị trí pixel so với lưới tâm.
  - Xử lý biên: pixel ngoài rìa lưới tâm → dùng tâm ô gần nhất (clamp).
  - Tính trọng số nội suy `(sy, sx)` theo khoảng cách tương đối đến 4 tâm.
  - `result_pixel = (1-sy)*(1-sx)*map_tl[v] + (1-sy)*sx*map_tr[v] + sy*(1-sx)*map_bl[v] + sy*sx*map_br[v]` với `v = padded[py, px]`.
- Crop kết quả về kích thước gốc `(H, W)`.

Tối ưu: dùng vòng lặp vectorize theo hàng hoặc dùng `np.clip` + `np.round` + `.astype(np.uint8)`.

Validation đầu vào: kiểm tra `ndim == 2`, `dtype == uint8` — raise `ValueError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_clahe -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Verify integration with pipeline**

Run: `python -c "import numpy as np; from source.core.pipeline import enhance; r = enhance(np.random.randint(10,60,(100,100,3),dtype=np.uint8), 'clahe'); print('is_mock:', r['is_mock'], 'mean:', r['image'].mean():.1f)"`
Expected: `is_mock: False`

- [ ] **Step 6: Commit**

```bash
git add source/algorithms/clahe.py tests/test_clahe.py
git commit -m "feat(tv3): implement CLAHE with bilinear interpolation from scratch"
```

---

### Task 3: Evaluation Module (TV2 — `metrics.py` + `benchmark.py`)

**Files:**
- Create: `source/evaluation/metrics.py`
- Create: `source/evaluation/benchmark.py`

**Interfaces:**
- Consumes:
  - `source.algorithms.hist_equal.equalize_hist` (Task 1)
  - `source.algorithms.clahe.clahe` (Task 2)
  - `source.algorithms.gamma.gamma_correct` (đã có)
  - `source.core.pipeline.enhance` (đã có)
- Produces:
  - `compute_metrics(y: np.ndarray) -> dict` — trả `{"mean": float, "std": float, "white_ratio": float}`
  - `compare_with_opencv(y: np.ndarray, method: str, params: dict) -> dict` — trả `{"our_result": ndarray, "cv_result": ndarray, "mae": float, "our_ms": float, "cv_ms": float}`
  - `run_benchmark(image_paths: list[str], output_csv: str) -> Path` — chạy benchmark toàn bộ dataset, xuất CSV

- [ ] **Step 1: Implement `source/evaluation/metrics.py`**

```python
def compute_metrics(y: np.ndarray) -> dict:
    """
    Tính các chỉ số định lượng cho kênh Y.
    Returns: {"mean": float, "std": float, "white_ratio": float}
    white_ratio = tỷ lệ pixel có giá trị >= 200 trên tổng số pixel.
    """
```

Triển khai bằng `np.mean`, `np.std`, `np.count_nonzero(y >= 200) / y.size`.

- [ ] **Step 2: Implement `source/evaluation/benchmark.py`**

```python
def compare_with_opencv(y: np.ndarray, method: str, params: dict | None = None) -> dict:
    """
    So sánh kết quả from-scratch vs OpenCV.
    method: 'he' hoặc 'clahe'.
    Returns: {"our_result", "cv_result", "mae", "our_ms", "cv_ms"}
    mae = Mean Absolute Error giữa 2 kết quả.
    """

def run_benchmark(image_paths: list, output_csv: str) -> Path:
    """
    Chạy benchmark cho mỗi ảnh × mỗi thuật toán (gamma, he, clahe).
    Xuất CSV với cột: filename, method, our_ms, cv_ms, mae, mean_before, mean_after, std_before, std_after.
    """
```

- `compare_with_opencv` cho HE: dùng `cv2.equalizeHist(y)`.
- `compare_with_opencv` cho CLAHE: dùng `cv2.createCLAHE(clipLimit=..., tileGridSize=...).apply(y)`.
- Đo thời gian bằng `time.perf_counter()`.
- `run_benchmark`: load ảnh, chuyển YCbCr, chạy 3 thuật toán, thu thập metrics, ghi CSV bằng `csv` module.

- [ ] **Step 3: Commit**

```bash
git add source/evaluation/metrics.py source/evaluation/benchmark.py
git commit -m "feat(tv2): add evaluation metrics and OpenCV benchmark"
```

---

### Task 4: Dataset Manifest (TV2 — `data/manifest.csv`)

**Files:**
- Create: `data/manifest.csv`

**Interfaces:**
- Consumes: Ảnh trong `data/input/`
- Produces: File CSV với cột `filename,category,description` cho benchmark script đọc.

- [ ] **Step 1: Tạo `data/manifest.csv` với ít nhất ảnh mẫu hiện có**

```csv
filename,category,description
sample_dark.png,indoor,Ảnh mẫu thiếu sáng trong nhà
```

Thêm ảnh thực tế sau: ít nhất 8 ảnh indoor, 8 ảnh outdoor, 8 ảnh difficult (nhiều nhiễu / cháy sáng cục bộ) theo yêu cầu 24 ảnh trong kế hoạch.

- [ ] **Step 2: Commit**

```bash
git add data/manifest.csv
git commit -m "feat(tv2): add dataset manifest for benchmark evaluation"
```

---

### Task 5: Full Integration Smoke Test & Final Verification

**Files:**
- Modify: Không sửa file nào — chỉ chạy kiểm tra tích hợp.

**Interfaces:**
- Consumes: Tất cả Tasks 1–4.

- [ ] **Step 1: Run full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: Tất cả tests PASS (test_color, test_io, test_gamma, test_he, test_clahe)

- [ ] **Step 2: Run CLI pipeline test cho cả 3 thuật toán**

Run: `python -m source.app --cli`
Expected: Cả 3 thuật toán hiển thị `REAL` (không còn `MOCK`), mean thay đổi có ý nghĩa.

- [ ] **Step 3: Kiểm tra GUI chạy đầy đủ 3 thuật toán**

Run: `python -m source.app`
Kiểm tra thủ công:
1. Mở ảnh `data/input/sample_dark.png`.
2. Chọn từng thuật toán (Gamma, HE, CLAHE) — ảnh kết quả phải sáng hơn, không còn nhãn `[MOCK]`.
3. Tab 3: Nhấn "Chạy đối sánh 3 thuật toán" — cả 4 khung đều có ảnh thật.
4. Tab 2: Histogram trước/sau phải khác biệt rõ rệt.

- [ ] **Step 4: Run benchmark (nếu đã có đủ ảnh)**

Run: `python -c "from source.evaluation.benchmark import run_benchmark; run_benchmark(['data/input/sample_dark.png'], 'results/benchmark.csv')"`
Expected: File `results/benchmark.csv` được tạo với dữ liệu hợp lệ.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify full integration — all 3 algorithms operational"
```


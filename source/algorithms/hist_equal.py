"""
source/algorithms/hist_equal.py
Thuật toán cân bằng biểu đồ mức xám toàn cục (Global Histogram Equalization - HE) from-scratch bằng NumPy.
Phụ trách: TV2 (Thành viên 2)

Nguyên lý:
    1. Tính biểu đồ mức xám (histogram) 256 mức của ảnh đầu vào.
    2. Tính hàm phân phối tích lũy (Cumulative Distribution Function - CDF).
    3. Tìm giá trị cdf_min (giá trị tích lũy khác 0 đầu tiên).
    4. Xây dựng bảng tra LUT:
       lut[v] = round((cdf[v] - cdf_min) / (total_pixels - cdf_min) * 255)
    5. Ánh xạ các giá trị pixel qua LUT để cân bằng độ tương phản toàn cục.
"""
import numpy as np


def compute_histogram(y: np.ndarray) -> np.ndarray:
    """
    Tính biểu đồ tần số xuất hiện của 256 mức xám [0..255].

    Args:
        y: Mảng 2D np.ndarray (H, W), dtype uint8.

    Returns:
        np.ndarray: Mảng 1D (256,), dtype int64 biểu diễn số lượng pixel tại mỗi mức xám.

    Raises:
        ValueError: Nếu mảng đầu vào không phải 2D hoặc không phải kiểu uint8.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Đầu vào phải là mảng numpy.ndarray")
    if y.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là mảng 2 chiều (H x W), nhận được {y.ndim} chiều")
    if y.dtype != np.uint8:
        raise ValueError(f"Dữ liệu ảnh phải ở kiểu uint8, nhận được {y.dtype}")

    return np.bincount(y.ravel(), minlength=256).astype(np.int64)


def equalize_hist(y: np.ndarray) -> np.ndarray:
    """
    Cân bằng biểu đồ mức xám toàn cục (Global Histogram Equalization) cho kênh sáng Y.

    Args:
        y: Mảng 2D np.ndarray (H, W), dtype uint8.

    Returns:
        np.ndarray: Mảng 2D (H, W), dtype uint8 sau khi cân bằng biểu đồ xám.

    Raises:
        ValueError: Nếu mảng đầu vào không phải 2D hoặc không phải kiểu uint8.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Đầu vào phải là mảng numpy.ndarray")
    if y.ndim != 2:
        raise ValueError(f"Ảnh đầu vào phải là mảng 2 chiều (H x W), nhận được {y.ndim} chiều")
    if y.dtype != np.uint8:
        raise ValueError(f"Dữ liệu ảnh phải ở kiểu uint8, nhận được {y.dtype}")

    total_pixels = y.size
    if total_pixels == 0:
        return y.copy()

    # 1. Tính histogram 256 mức
    hist = compute_histogram(y)

    # 2. Tính CDF (Cumulative Distribution Function)
    cdf = np.cumsum(hist)

    # 3. Tìm cdf_min (giá trị tích lũy khác 0 đầu tiên)
    nonzero_indices = np.flatnonzero(cdf)
    if len(nonzero_indices) == 0:
        return y.copy()

    cdf_min = cdf[nonzero_indices[0]]

    # Nếu cdf_min == cdf[-1], toàn bộ ảnh chỉ có 1 mức sáng duy nhất (ảnh đơn sắc)
    if cdf_min == cdf[-1]:
        return y.copy()

    # 4. Xây dựng bảng tra LUT
    # lut[i] = round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
    lut = np.clip(
        np.round(((cdf.astype(np.float64) - cdf_min) / (total_pixels - cdf_min)) * 255.0),
        0,
        255
    ).astype(np.uint8)

    # 5. Ánh xạ pixel qua LUT (không thay đổi in-place mảng gốc)
    return lut[y]

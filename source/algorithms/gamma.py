"""
source/algorithms/gamma.py
Thuật toán Gamma Correction phi tuyến from-scratch bằng NumPy.
Phụ trách: TV1 (Lead & Tích hợp)

Nguyên lý:
    s = c * (r / 255.0) ** gamma
    Với ảnh thiếu sáng, chọn gamma < 1.0 để kéo sáng các vùng tối,
    đồng thời nén dải sáng các vùng đã đủ sáng để không bị cháy.
    Để tối ưu tốc độ thực thi O(1) trên từng pixel, thuật toán sử dụng bảng tra LUT 256 phần tử.
"""
from numbers import Real
import numpy as np


def gamma_correct(y: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Hiệu chỉnh gamma phi tuyến trên kênh sáng Y bằng bảng tra LUT 256 mức.

    Args:
        y: Mảng 2D np.ndarray (H, W), dtype uint8.
        gamma: Hệ số gamma (> 0). Mặc định là 0.6 (làm sáng ảnh thiếu sáng).

    Returns:
        np.ndarray: Mảng 2D (H, W), dtype uint8 sau khi áp dụng Gamma Correction.

    Raises:
        ValueError: Nếu mảng đầu vào không phải 2D, không phải kiểu uint8,
                    hoặc hệ số gamma <= 0.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Đầu vào phải là mảng numpy.ndarray")

    if y.ndim != 2:
        raise ValueError(f"Kênh sáng Y phải là mảng 2 chiều (H x W), nhận được {y.ndim} chiều")

    if y.dtype != np.uint8:
        raise ValueError(f"Dữ liệu kênh Y phải ở kiểu uint8, nhận được {y.dtype}")

    if (isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, Real)
            or not np.isfinite(gamma) or gamma <= 0):
        raise ValueError(f"Hệ số gamma phải là số dương lớn hơn 0, nhận được {gamma}")

    # Xây dựng bảng tra LUT (Look-Up Table) cho 256 mức xám [0..255]
    # s = 255 * (r / 255) ** gamma
    inv_255 = 1.0 / 255.0
    r_values = np.arange(256, dtype=np.float64) * inv_255
    lut = np.clip(np.round(255.0 * (r_values ** gamma)), 0, 255).astype(np.uint8)

    # Ánh xạ pixel qua LUT (tốc độ cực nhanh O(1) cho mỗi pixel)
    return lut[y]


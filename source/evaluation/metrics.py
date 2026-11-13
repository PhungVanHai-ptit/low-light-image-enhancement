"""
source/evaluation/metrics.py
Định lượng các chỉ số chất lượng ảnh trên kênh sáng Y.
Phụ trách: TV2 (Đánh giá & Benchmark)
"""
import numpy as np


def compute_metrics(y: np.ndarray) -> dict:
    """
    Compute quantitative metrics for a Y channel image.

    Args:
        y: 2D ndarray (H, W), dtype uint8

    Returns:
        dict with keys:
        - "mean": float — average brightness
        - "std": float — contrast (standard deviation)
        - "white_ratio": float — ratio of pixels >= 200 to total pixels
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Đầu vào phải là mảng numpy.ndarray")
    if y.ndim != 2:
        raise ValueError(f"Kênh sáng Y phải là mảng 2 chiều (H x W), nhận được {y.ndim} chiều")
    if y.dtype != np.uint8:
        raise ValueError(f"Dữ liệu kênh Y phải ở kiểu uint8, nhận được {y.dtype}")
    if y.size == 0:
        raise ValueError("Mảng không được rỗng")

    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "white_ratio": float(np.count_nonzero(y >= 200) / y.size)
    }

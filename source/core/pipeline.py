"""
source/core/pipeline.py
Điều phối xử lý ảnh và đo thời gian thực thi (do TV1 triển khai).
"""
import time
from typing import Optional, Dict, Any
import numpy as np
from source.core.color import rgb_to_ycbcr, ycbcr_to_rgb


# Cung cấp hàm giả lập (mock) nếu các module chưa hoàn thành
def _mock_enhance(y: np.ndarray, *args, **kwargs) -> np.ndarray:
    return y.copy()


def enhance(image: np.ndarray, method: str, params: Optional[Dict[str, Any]] = None) -> dict:
    """
    Pipeline chính: Tách Y -> Chạy thuật toán -> Ghép màu -> Đo thời gian.
    """
    if params is None:
        params = {}

    method = method.lower()
    if method not in ("gamma", "he", "clahe"):
        raise ValueError(f"Thuật toán không hợp lệ: {method}")

    is_color = (image.ndim == 3 and image.shape[2] == 3)

    if is_color:
        y_orig, cb, cr = rgb_to_ycbcr(image)
    else:
        y_orig = image.copy()

    # Nhập hàm thực tế hoặc hàm mock
    algo_fn = _mock_enhance
    try:
        if method == "gamma":
            from source.algorithms.gamma import gamma_correct as algo_fn
        elif method == "he":
            from source.algorithms.hist_equal import equalize_hist as algo_fn
        elif method == "clahe":
            from source.algorithms.clahe import clahe as algo_fn
    except ImportError:
        algo_fn = _mock_enhance

    # Đo thời gian thuật toán thực thi chính xác (tính bằng ms)
    t_start = time.perf_counter()
    y_result = algo_fn(y_orig, **params)
    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0

    # Ghép lại ảnh kết quả
    if is_color:
        final_image = ycbcr_to_rgb(y_result, cb, cr)
    else:
        final_image = y_result.copy()

    is_mock = (algo_fn == _mock_enhance)

    return {
        "image": final_image,
        "y_original": y_orig,
        "y_result": y_result,
        "method": method,
        "params": params,
        "algorithm_ms": duration_ms,
        "is_mock": is_mock
    }


"""
source/core/color.py
Chuyển đổi không gian màu RGB <-> YCbCr from-scratch dùng chuẩn ITU-R BT.601.
Do TV1 chịu trách nhiệm triển khai.
"""
from typing import Tuple
import numpy as np


def rgb_to_ycbcr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển đổi ảnh RGB (uint8) sang (Y, Cb, Cr).
    Y: uint8 [0, 255]
    Cb, Cr: float64 giữ nguyên độ chính xác để bảo toàn màu sắc
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Đầu vào phải là ảnh màu RGB 3 chiều (H x W x 3)")
    if image.dtype != np.uint8:
        raise ValueError("Dữ liệu ảnh phải ở kiểu uint8")

    img_f = image.astype(np.float64)
    R = img_f[:, :, 0]
    G = img_f[:, :, 1]
    B = img_f[:, :, 2]

    # Công thức BT.601
    Y  =  0.299000 * R + 0.587000 * G + 0.114000 * B
    Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.500000 * B
    Cr = 128.0 + 0.500000 * R - 0.418688 * G - 0.081312 * B

    # Làm tròn và kẹp giá trị kênh Y sang uint8
    Y_uint8 = np.clip(np.round(Y), 0, 255).astype(np.uint8)

    return Y_uint8, Cb, Cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """
    Tái tạo ảnh RGB uint8 từ Y, Cb, Cr.
    Có kẹp giá trị (clipping [0, 255]) để chống tràn số.
    """
    if y.shape != cb.shape or y.shape != cr.shape:
        raise ValueError("Kích thước các kênh Y, Cb, Cr phải trùng khớp")

    Y = y.astype(np.float64)

    # Công thức ngược tái tạo RGB
    R = Y + 1.402000 * (cr - 128.0)
    G = Y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    B = Y + 1.772000 * (cb - 128.0)

    RGB = np.stack([R, G, B], axis=2)
    RGB = np.clip(np.round(RGB), 0, 255).astype(np.uint8)
    return RGB


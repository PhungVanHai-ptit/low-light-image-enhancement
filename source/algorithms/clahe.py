"""
source/algorithms/clahe.py
Thuật toán Contrast Limited Adaptive Histogram Equalization (CLAHE) from-scratch bằng NumPy.
Phụ trách: TV3 (Thuật toán CLAHE)

Nguyên lý:
    1. Chia ảnh xám thành lưới các ô nhỏ (tiles/contextual regions) kích thước grid = (grid_y, grid_x).
    2. Với mỗi ô, tính lược đồ mức xám (histogram 256 bins), cắt ngưỡng (clipping) tại clip_limit
       và phân phối đều lượng pixel vượt ngưỡng vào tất cả các bin để hạn chế khuếch đại nhiễu quá mức.
    3. Tính hàm phân phối tích lũy (CDF) và chuẩn hóa thành bảng ánh xạ LUT 256 mức cho từng ô.
    4. Sử dụng nội suy song tuyến tính (bilinear interpolation) giữa 4 tâm ô lân cận để tính mức xám
       cho từng pixel, loại bỏ hiện tượng phân mảng khối (checkerboard artifacts).
"""
from numbers import Integral, Real
import numpy as np


def _redistribute_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """Clip once, then redistribute every removed count.

    Redistribution may raise bins above the initial clipping threshold.
    """
    clipped = np.minimum(hist, clip_limit).astype(np.int64)
    excess = int(np.sum(hist, dtype=np.int64) - clipped.sum())
    bonus, residual = divmod(excess, 256)
    clipped += bonus
    if residual:
        indices = np.arange(residual) * 256 // residual
        clipped[indices] += 1
    return clipped



def clahe(
    y: np.ndarray,
    grid: tuple[int, int] = (8, 8),
    clip_factor: float = 2.0
) -> np.ndarray:
    """
    Cân bằng lược đồ thích nghi giới hạn độ tương phản (CLAHE) kèm nội suy song tuyến tính.

    Args:
        y: Mảng 2D np.ndarray (H, W), dtype uint8.
        grid: (grid_y, grid_x) số ô lưới theo chiều dọc và ngang (mặc định (8, 8)).
        clip_factor: Hệ số cắt ngưỡng (clip limit multiplier, mặc định 2.0).

    Returns:
        np.ndarray: Mảng 2D (H, W), dtype uint8 sau khi áp dụng CLAHE.

    Raises:
        ValueError: Nếu mảng đầu vào không phải 2D, không phải kiểu uint8,
                    hoặc tham số grid / clip_factor không hợp lệ.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Đầu vào phải là mảng numpy.ndarray")

    if y.ndim != 2:
        raise ValueError(f"Kênh sáng Y phải là mảng 2 chiều (H x W), nhận được {y.ndim} chiều")

    if y.dtype != np.uint8:
        raise ValueError(f"Dữ liệu kênh Y phải ở kiểu uint8, nhận được {y.dtype}")

    if (not isinstance(grid, (tuple, list)) or len(grid) != 2
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, Integral) or value <= 0
                   for value in grid)):
        raise ValueError(f"grid phải là tuple (grid_y, grid_x) gồm 2 số nguyên dương, nhận được {grid}")

    if (isinstance(clip_factor, (bool, np.bool_))
            or not isinstance(clip_factor, Real)
            or not np.isfinite(clip_factor) or clip_factor <= 0):
        raise ValueError(f"clip_factor phải là số thực dương lớn hơn 0, nhận được {clip_factor}")

    H, W = y.shape
    if H == 0 or W == 0:
        return y.copy()

    # Trường hợp ảnh đồng nhất (constant image)
    if y.min() == y.max():
        return y.copy()

    grid_y, grid_x = grid

    # Xử lý trường hợp ảnh nhỏ hơn kích thước lưới: thu về grid (1, 1)
    if H < grid_y or W < grid_x:
        grid_y, grid_x = 1, 1

    # Phase 1: Kích thước tile và Padding đối xứng
    tile_h = int(np.ceil(H / grid_y))
    tile_w = int(np.ceil(W / grid_x))
    padded_h = tile_h * grid_y
    padded_w = tile_w * grid_x
    pad_h = padded_h - H
    pad_w = padded_w - W

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(y, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = y.copy()

    # Phase 2: Tính Histogram, Clipping, Redistribution và Mapping LUT cho từng tile
    tile_pixels = tile_h * tile_w
    clip_limit = max(1, int(clip_factor * tile_pixels / 256))

    mappings = np.zeros((grid_y, grid_x, 256), dtype=np.uint8)
    for r in range(grid_y):
        for c in range(grid_x):
            tile = padded[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            hist = np.bincount(tile.ravel(), minlength=256)
            hist = _redistribute_histogram(hist, clip_limit)
            cdf = np.cumsum(hist)
            # Chuẩn hóa CDF về dải [0, 255]
            mappings[r, c] = np.clip(np.round(cdf * (255.0 / tile_pixels)), 0, 255).astype(np.uint8)

    # Phase 3 & 4: Xác định tâm tile và Nội suy song tuyến tính (Bilinear Interpolation)
    if grid_y == 1:
        r1 = np.zeros(padded_h, dtype=int)
        r2 = np.zeros(padded_h, dtype=int)
        sy = np.zeros(padded_h, dtype=np.float64)
    else:
        center_0_y = tile_h // 2
        ty = (np.arange(padded_h) - center_0_y) / tile_h
        r1 = np.clip(np.floor(ty).astype(int), 0, grid_y - 2)
        r2 = r1 + 1
        sy = np.clip(ty - r1, 0.0, 1.0)

    if grid_x == 1:
        c1 = np.zeros(padded_w, dtype=int)
        c2 = np.zeros(padded_w, dtype=int)
        sx = np.zeros(padded_w, dtype=np.float64)
    else:
        center_0_x = tile_w // 2
        tx = (np.arange(padded_w) - center_0_x) / tile_w
        c1 = np.clip(np.floor(tx).astype(int), 0, grid_x - 2)
        c2 = c1 + 1
        sx = np.clip(tx - c1, 0.0, 1.0)

    # Vector hóa tra cứu và nội suy giá trị pixel
    R1 = r1[:, None]
    R2 = r2[:, None]
    C1 = c1[None, :]
    C2 = c2[None, :]
    SY = sy[:, None]
    SX = sx[None, :]

    v_tl = mappings[R1, C1, padded]
    v_tr = mappings[R1, C2, padded]
    v_bl = mappings[R2, C1, padded]
    v_br = mappings[R2, C2, padded]

    w_tl = (1.0 - SY) * (1.0 - SX)
    w_tr = (1.0 - SY) * SX
    w_bl = SY * (1.0 - SX)
    w_br = SY * SX

    interp = w_tl * v_tl + w_tr * v_tr + w_bl * v_bl + w_br * v_br

    # Cắt lại về kích thước ban đầu (H, W) và làm tròn
    result = np.clip(np.round(interp[:H, :W]), 0, 255).astype(np.uint8)
    return result

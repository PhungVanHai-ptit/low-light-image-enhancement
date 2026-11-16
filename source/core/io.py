"""
source/core/io.py
Nhập/xuất ảnh và chuẩn hóa định dạng dữ liệu (do TV1 triển khai).
"""
from pathlib import Path
from typing import Union
from PIL import Image, ImageOps
import numpy as np


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Đọc ảnh từ file, chuẩn hóa định dạng.
    - Ảnh xám -> trả về mảng uint8 2D (H x W)
    - Ảnh màu -> trả về mảng uint8 3D (H x W x 3)
    - Ảnh có kênh alpha -> ghép trên nền trắng rồi chuyển RGB
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

    try:
        with Image.open(path) as img:
            # Tự động xoay ảnh theo hướng cảm biến chụp (EXIF)
            img = ImageOps.exif_transpose(img)

            # Xử lý kênh Alpha (RGBA, LA) hoặc Palette có trong suốt
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                img_rgb = img.convert("RGBA")
                bg.paste(img_rgb, mask=img_rgb.split()[3])
                return np.array(bg, dtype=np.uint8)

            if img.mode == "L":
                return np.array(img, dtype=np.uint8)

            # Chuyển về RGB chuẩn 8-bit
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise OSError(f"Lỗi khi đọc file ảnh {path}: {str(e)}")


def save_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Lưu mảng numpy (uint8) thành file ảnh PNG/JPEG.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Ảnh cần lưu phải có kiểu dữ liệu uint8")

    if image.size == 0 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("Ảnh cần lưu phải là ảnh xám hoặc RGB không rỗng")

    path = Path(path)
    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
    image_format = formats.get(path.suffix.lower())
    if image_format is None:
        raise ValueError("Đuôi file phải là .png, .jpg hoặc .jpeg")

    img = Image.fromarray(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "JPEG":
        img.save(path, format=image_format, quality=95)
    else:
        img.save(path, format=image_format)

    return path


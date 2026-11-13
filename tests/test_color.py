"""
tests/test_color.py
Kiểm thử tính đúng của module chuyển đổi màu RGB <-> YCbCr.
"""
import unittest
import numpy as np
from source.core.color import rgb_to_ycbcr, ycbcr_to_rgb


class TestColorConversion(unittest.TestCase):
    def test_roundtrip_precision(self):
        """Kiểm tra sai số khứ hồi RGB -> YCbCr -> RGB không quá 2 mức xám"""
        # Tạo ảnh ngẫu nhiên 100x100x3
        np.random.seed(42)
        orig_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        y, cb, cr = rgb_to_ycbcr(orig_rgb)
        recon_rgb = ycbcr_to_rgb(y, cb, cr)

        diff = np.abs(orig_rgb.astype(int) - recon_rgb.astype(int))
        max_diff = np.max(diff)

        self.assertLessEqual(max_diff, 2, f"Sai số khứ hồi vượt quá ngưỡng 2: {max_diff}")

    def test_grayscale_preservation(self):
        """Với ảnh xám (R=G=B), kênh Y phải xấp xỉ bằng chính giá trị đó"""
        gray_val = 150
        rgb = np.full((10, 10, 3), gray_val, dtype=np.uint8)
        y, _, _ = rgb_to_ycbcr(rgb)
        self.assertTrue(np.all(np.abs(y.astype(int) - gray_val) <= 1))


if __name__ == "__main__":
    unittest.main()


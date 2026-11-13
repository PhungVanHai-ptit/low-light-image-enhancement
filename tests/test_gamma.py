"""
tests/test_gamma.py
Kiểm thử tính đúng đắn của thuật toán Gamma Correction (TV1).
"""
import unittest
import numpy as np
from source.algorithms.gamma import gamma_correct


class TestGammaCorrection(unittest.TestCase):
    def setUp(self):
        # Tạo ảnh giả lập mức xám 64x64
        np.random.seed(42)
        self.dark_image = np.random.randint(10, 60, (64, 64), dtype=np.uint8)

    def test_output_shape_and_dtype(self):
        """Kích thước và kiểu dữ liệu phải giữ nguyên uint8 và 2D."""
        res = gamma_correct(self.dark_image, gamma=0.6)
        self.assertEqual(res.shape, self.dark_image.shape)
        self.assertEqual(res.dtype, np.uint8)

    def test_brightening_effect_when_gamma_less_than_one(self):
        """Khi gamma < 1.0 (ví dụ 0.6), ảnh phải sáng hơn (giá trị trung bình tăng)."""
        res = gamma_correct(self.dark_image, gamma=0.6)
        self.assertGreater(res.mean(), self.dark_image.mean())

    def test_darkening_effect_when_gamma_greater_than_one(self):
        """Khi gamma > 1.0 (ví dụ 1.8), ảnh phải tối hơn (giá trị trung bình giảm)."""
        res = gamma_correct(self.dark_image, gamma=1.8)
        self.assertLess(res.mean(), self.dark_image.mean())

    def test_identity_when_gamma_one(self):
        """Khi gamma = 1.0, ảnh kết quả phải trùng khớp hoàn toàn với ảnh gốc."""
        res = gamma_correct(self.dark_image, gamma=1.0)
        np.testing.assert_array_equal(res, self.dark_image)

    def test_extreme_values_preservation(self):
        """Mức 0 vẫn ánh xạ về 0, mức 255 vẫn ánh xạ về 255."""
        test_arr = np.array([[0, 255]], dtype=np.uint8)
        res = gamma_correct(test_arr, gamma=0.5)
        self.assertEqual(res[0, 0], 0)
        self.assertEqual(res[0, 1], 255)

    def test_immutability(self):
        """Mảng gốc đầu vào không bị thay đổi in-place."""
        copy_before = self.dark_image.copy()
        _ = gamma_correct(self.dark_image, gamma=0.5)
        np.testing.assert_array_equal(self.dark_image, copy_before)

    def test_invalid_inputs_raise_errors(self):
        """Các đầu vào không hợp lệ phải ném ValueError."""
        # 3D array thay vì 2D
        with self.assertRaises(ValueError):
            gamma_correct(np.zeros((10, 10, 3), dtype=np.uint8), gamma=0.6)

        # Dtype float thay vì uint8
        with self.assertRaises(ValueError):
            gamma_correct(np.zeros((10, 10), dtype=np.float32), gamma=0.6)

        # Gamma <= 0
        with self.assertRaises(ValueError):
            gamma_correct(self.dark_image, gamma=0.0)
        with self.assertRaises(ValueError):
            gamma_correct(self.dark_image, gamma=-0.5)


if __name__ == "__main__":
    unittest.main()


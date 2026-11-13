"""
tests/test_clahe.py
Unit tests for Contrast Limited Adaptive Histogram Equalization (CLAHE) module (TV3).
"""
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
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (67, 53), dtype=np.uint8)
        result = clahe(y, grid=(8, 8))
        self.assertEqual(result.shape, (67, 53))
        self.assertEqual(result.dtype, np.uint8)

    def test_small_image(self):
        from source.algorithms.clahe import clahe
        y = np.random.randint(0, 256, (3, 3), dtype=np.uint8)
        result = clahe(y, grid=(8, 8))
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_contrast_enhancement(self):
        from source.algorithms.clahe import clahe
        y = np.random.randint(10, 60, (128, 128), dtype=np.uint8)
        result = clahe(y, grid=(8, 8), clip_factor=2.0)
        self.assertGreater(float(np.std(result)), float(np.std(y)))

    def test_different_grid_sizes(self):
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

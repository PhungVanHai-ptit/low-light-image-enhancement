"""
tests/test_he.py
Unit tests for Global Histogram Equalization (HE) module (TV2).
"""
import unittest
import numpy as np
from source.algorithms.hist_equal import compute_histogram, equalize_hist


class TestComputeHistogram(unittest.TestCase):
    def test_uniform_image(self):
        y = np.full((10, 10), 100, dtype=np.uint8)
        hist = compute_histogram(y)
        self.assertEqual(hist.shape, (256,))
        self.assertEqual(hist.dtype, np.int64)
        self.assertEqual(hist[100], 100)
        self.assertEqual(hist.sum(), 100)

    def test_known_distribution(self):
        y = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        hist = compute_histogram(y)
        for i in range(4):
            self.assertEqual(hist[i], 1)
        self.assertEqual(hist[4:].sum(), 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_histogram(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            compute_histogram(np.zeros((10, 10), dtype=np.float32))
        with self.assertRaises(ValueError):
            compute_histogram(np.zeros((10,), dtype=np.uint8))


class TestEqualizeHist(unittest.TestCase):
    def test_output_shape_dtype(self):
        y = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = equalize_hist(y)
        self.assertEqual(result.shape, y.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_constant_image_safe(self):
        y = np.full((50, 50), 128, dtype=np.uint8)
        result = equalize_hist(y)
        np.testing.assert_array_equal(result, y)

    def test_no_inplace_mutation(self):
        y = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        y_copy = y.copy()
        equalize_hist(y)
        np.testing.assert_array_equal(y, y_copy)

    def test_contrast_increases(self):
        y = np.random.randint(10, 60, (100, 100), dtype=np.uint8)
        result = equalize_hist(y)
        self.assertGreater(float(np.std(result)), float(np.std(y)))

    def test_all_white_image(self):
        y = np.full((20, 20), 255, dtype=np.uint8)
        result = equalize_hist(y)
        np.testing.assert_array_equal(result, y)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            equalize_hist(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            equalize_hist(np.zeros((10, 10), dtype=np.int32))
        with self.assertRaises(ValueError):
            equalize_hist(np.zeros((10,), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

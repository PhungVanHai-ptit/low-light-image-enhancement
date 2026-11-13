import unittest
from unittest.mock import Mock, patch
import numpy as np
from source.algorithms.clahe import _redistribute_histogram
from source.evaluation.benchmark import _measure_algorithm


class TestRegressions(unittest.TestCase):
    def test_redistribution_preserves_counts(self):
        hist = np.zeros(256, dtype=np.int64)
        hist[64], hist[65] = 399, 1
        original = hist.copy()
        for limit in (1, 2, 10, 400):
            with self.subTest(limit=limit):
                output = _redistribute_histogram(hist, limit)
                self.assertEqual(int(output.sum()), 400)
                np.testing.assert_array_equal(hist, original)
                self.assertTrue(np.all(output >= 0))
        self.assertIsNot(_redistribute_histogram(hist, 400), hist)

    def test_benchmark_warmup_and_median(self):
        y = np.zeros((2, 2), dtype=np.uint8)
        operation = Mock(return_value=y.copy())
        ticks = [0, .005, 1, 1.001, 2, 2.003, 3, 3.002, 4, 4.100]
        with patch("source.evaluation.benchmark.time.perf_counter", side_effect=ticks):
            result, duration = _measure_algorithm(operation, y, gamma=0.6)
        self.assertEqual(operation.call_count, 6)
        self.assertAlmostEqual(duration, 3.0)
        np.testing.assert_array_equal(result, y)


class TestParameterValidation(unittest.TestCase):
    def test_gamma_rejects_invalid_numbers(self):
        from source.algorithms.gamma import gamma_correct
        y = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for value in (float("nan"), float("inf"), -float("inf"), True, "0.6", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gamma_correct(y, value)

    def test_clahe_rejects_invalid_grid(self):
        from source.algorithms.clahe import clahe
        y = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for grid in ((2.5, 2), (True, 2), (0, 2), (2,), None, 8):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                clahe(y, grid=grid)

    def test_clahe_rejects_invalid_clip(self):
        from source.algorithms.clahe import clahe
        y = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for value in (float("nan"), float("inf"), True, "2", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clahe(y, clip_factor=value)

    def test_pipeline_rejects_extra_parameters(self):
        from source.core.pipeline import enhance
        y = np.arange(16, dtype=np.uint8).reshape(4, 4)
        for method in ("gamma", "he", "clahe"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                enhance(y, method, {"extra": 1})
        for method, params in (("he", []), (None, {})):
            with self.assertRaises(ValueError):
                enhance(y, method, params)

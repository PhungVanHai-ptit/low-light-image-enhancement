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

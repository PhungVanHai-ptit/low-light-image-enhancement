import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from source.core.io import save_image
from source.ui import main_window as gui


class TestUIResults(unittest.TestCase):
    def setUp(self):
        self.app = gui.MainWindow()
        self.app.withdraw()
        self.addCleanup(self.app._on_close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.first = Path(self.temp.name) / "first.png"
        self.second = Path(self.temp.name) / "second.png"
        rng = np.random.default_rng(42)
        save_image(rng.integers(0, 100, (32, 48, 3), dtype=np.uint8), self.first)
        save_image(rng.integers(0, 100, (19, 31), dtype=np.uint8), self.second)

    def load(self, path):
        with patch.object(gui.filedialog, "askopenfilename", return_value=str(path)):
            self.app._on_load_image()

    def test_comparison_uses_selected_clip_factor(self):
        with patch.object(gui.messagebox, "showerror", side_effect=AssertionError):
            self.load(self.first)
            self.app.algo_var.set("clahe")
            self.app.clip_scale.set(4)
            self.app._on_process()
            expected = self.app.enhanced_image.copy()
            self.app._on_run_all_algorithms()
            np.testing.assert_array_equal(
                self.app.multi_compare_images["clahe"], expected
            )

    def test_new_image_clears_comparison_results_even_after_zoom(self):
        with patch.object(gui.messagebox, "showerror", side_effect=AssertionError):
            self.load(self.first)
            self.app._on_run_all_algorithms()
            self.app._on_run_opencv_comparison()
            self.load(self.second)
            self.assertEqual(set(self.app.multi_compare_images), {"orig"})
            self.assertEqual(self.app.opencv_compare_images, {})
            self.app._on_zoom_reset_tab3()
            self.app._on_zoom_reset_tab4()
            for key in ("gamma", "he", "clahe"):
                self.assertFalse(self.app.multi_boxes[key]["lbl_img"].cget("image"))
                self.assertEqual(self.app.multi_boxes[key]["lbl_info"].cget("text"), "")
            self.assertFalse(self.app.lbl_cv_our_img.cget("image"))
            self.assertFalse(self.app.lbl_cv_lib_img.cget("image"))
            self.assertEqual(self.app.enhanced_image.shape, (19, 31))

    def test_failed_load_preserves_current_results(self):
        self.load(self.first)
        expected = self.app.enhanced_image.copy()
        with patch.object(gui.messagebox, "showerror") as error:
            self.load(Path(self.temp.name) / "missing.png")
            error.assert_called_once()
        np.testing.assert_array_equal(self.app.enhanced_image, expected)

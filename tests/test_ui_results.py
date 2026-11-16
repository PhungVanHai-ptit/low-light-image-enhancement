import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from source.core.io import load_image, save_image
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


    def test_save_name_matches_processed_image_after_selection_change(self):
        self.app.realtime_var.set(False)
        with patch.object(gui.messagebox, "showerror", side_effect=AssertionError):
            self.load(self.first)
            for processed, selected in (("he", "clahe"), ("clahe", "gamma"), ("gamma", "he")):
                with self.subTest(processed=processed, selected=selected):
                    self.app.algo_var.set(processed)
                    self.app._on_process()
                    expected = self.app.enhanced_image.copy()
                    self.app.algo_var.set(selected)
                    self.app._on_algo_changed()
                    path = Path(self.temp.name) / f"{processed}.png"
                    with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(path)) as dialog, patch.object(gui.messagebox, "showinfo"):
                        self.app._on_save_image()
                    self.assertEqual(
                        dialog.call_args.kwargs["initialfile"],
                        f"enhanced_{processed}_first.png",
                    )
                    np.testing.assert_array_equal(load_image(path), expected)

    def test_save_after_failed_processing_uses_last_successful_method(self):
        self.app.realtime_var.set(False)
        self.load(self.first)
        self.app.algo_var.set("he")
        self.app._on_process()
        self.app.algo_var.set("clahe")
        with patch.object(gui, "enhance", side_effect=ValueError("test failure")), patch.object(gui.messagebox, "showerror") as error:
            self.app._on_process()
            error.assert_called_once()
        with patch.object(gui.filedialog, "asksaveasfilename", return_value="") as dialog, patch.object(gui, "save_image") as writer:
            self.app._on_save_image()
            self.assertEqual(dialog.call_args.kwargs["initialfile"], "enhanced_he_first.png")
            writer.assert_not_called()

    def test_save_without_result_does_not_open_dialog(self):
        with patch.object(gui.filedialog, "asksaveasfilename") as dialog:
            self.app._on_save_image()
            self.app.enhanced_image = np.zeros((2, 2), dtype=np.uint8)
            self.app._on_save_image()
            dialog.assert_not_called()


    def test_bmp_source_defaults_to_png_and_rejects_invalid_save(self):
        from PIL import Image
        source = Path(self.temp.name) / "source.photo.bmp"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(source)
        with patch.object(gui.messagebox, "showerror", side_effect=AssertionError):
            self.load(source)
        with patch.object(gui.filedialog, "asksaveasfilename", return_value="") as dialog:
            self.app._on_save_image()
        method = self.app.last_pipeline_result["method"]
        self.assertEqual(dialog.call_args.kwargs["initialfile"],
                         f"enhanced_{method}_source.photo.png")
        with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(source)), patch.object(gui.messagebox, "showerror") as error:
            before = source.read_bytes()
            self.app._on_save_image()
            error.assert_called_once()
            self.assertEqual(source.read_bytes(), before)

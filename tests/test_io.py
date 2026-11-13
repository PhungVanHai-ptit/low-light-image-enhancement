"""
tests/test_io.py
Kiểm thử module đọc/ghi ảnh source/core/io.py.
"""
import unittest
import numpy as np
import tempfile
from pathlib import Path
from source.core.io import load_image, save_image


class TestIO(unittest.TestCase):
    def setUp(self):
        """Tạo thư mục tạm dùng chung cho các test case."""
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_rgb(self):
        """Lưu rồi tải lại ảnh RGB PNG — dữ liệu phải khớp 100%."""
        original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        path = self.tmp_dir / "test_rgb.png"
        save_image(original, path)
        loaded = load_image(path)
        np.testing.assert_array_equal(original, loaded,
            err_msg="Dữ liệu ảnh RGB sau khi lưu/tải không khớp")

    def test_save_and_load_grayscale(self):
        """Lưu rồi tải lại ảnh xám — shape phải 2D và dữ liệu khớp."""
        from PIL import Image
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        path = self.tmp_dir / "test_gray.png"
        Image.fromarray(gray, mode="L").save(path)
        loaded = load_image(path)
        self.assertEqual(loaded.ndim, 2, "Ảnh xám phải trả về mảng 2D")
        np.testing.assert_array_equal(gray, loaded)

    def test_load_missing_file_raises(self):
        """Tải file không tồn tại phải ném FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_image(self.tmp_dir / "nonexistent.png")

    def test_save_creates_parent_dirs(self):
        """save_image phải tự tạo thư mục cha nếu chưa tồn tại."""
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        nested_path = self.tmp_dir / "a" / "b" / "c" / "out.png"
        save_image(img, nested_path)
        self.assertTrue(nested_path.exists(), "File phải được tạo cùng thư mục cha")

    def test_save_requires_uint8(self):
        """save_image phải ném ValueError nếu dtype không phải uint8."""
        img_float = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            save_image(img_float, self.tmp_dir / "bad.png")


if __name__ == "__main__":
    unittest.main()


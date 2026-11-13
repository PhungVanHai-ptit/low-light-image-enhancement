"""
source/app.py
Điểm khởi chạy chính của ứng dụng Cải thiện ảnh thiếu sáng.
Chạy ứng dụng GUI: python -m source.app
Chạy kiểm thử pipeline dòng lệnh: python -m source.app --cli
"""
import sys
import numpy as np
from source.core.pipeline import enhance


def run_cli_test():
    print("=== KIỂM THỬ PIPELINE CÁC THUẬT TOÁN (FROM-SCRATCH) ===")
    dummy_image = np.random.randint(25, 75, (256, 256, 3), dtype=np.uint8)
    print(f"Ảnh đầu vào: shape={dummy_image.shape}, dtype={dummy_image.dtype}, mean={dummy_image.mean():.2f}")

    for method in ("gamma", "he", "clahe"):
        result = enhance(dummy_image, method=method)
        img_out = result["image"]
        ms = result["algorithm_ms"]
        mean_out = img_out.mean()
        algo_author = {"gamma": "TV1", "he": "TV2", "clahe": "TV3"}.get(method, "")
        status = "MOCK (chờ hoàn thiện)" if result.get("is_mock") else f"REAL ({algo_author})"
        print(f"  [{method.upper():5s} - {status}] shape={img_out.shape}, dtype={img_out.dtype}, "
              f"time={ms:.3f} ms, mean: {dummy_image.mean():.1f} -> {mean_out:.1f}")

    print("\n✅ Tất cả các thuật toán chạy thành công!")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_test()
    else:
        try:
            from source.ui.main_window import start_app
            print("Đang khởi chạy giao diện Desktop (Tkinter)...")
            start_app()
        except Exception as e:
            print(f"Không thể khởi động giao diện đồ họa: {e}")
            print("Đang chạy kiểm thử dòng lệnh thay thế...")
            run_cli_test()


if __name__ == "__main__":
    main()

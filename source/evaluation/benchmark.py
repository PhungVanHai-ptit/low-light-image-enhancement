"""
source/evaluation/benchmark.py
So sánh hiệu năng và độ sai khác giữa thuật toán from-scratch và thư viện OpenCV.
Phụ trách: TV2 (Đánh giá & Benchmark)
"""
import time
import csv
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import cv2

from source.core.io import load_image
from source.core.color import rgb_to_ycbcr
from source.algorithms.gamma import gamma_correct
from source.algorithms.hist_equal import equalize_hist
from source.algorithms.clahe import clahe as clahe_custom
from source.evaluation.metrics import compute_metrics


def compare_with_opencv(y: np.ndarray, method: str, params: Optional[dict] = None) -> dict:
    """
    Compare from-scratch result vs OpenCV result.
    method: 'he' or 'clahe'
    Returns: {"our_result": ndarray, "cv_result": ndarray, "mae": float, "our_ms": float, "cv_ms": float}
    """
    if params is None:
        params = {}

    method = method.lower()
    if method not in ("he", "clahe"):
        raise ValueError(f"Unsupported method for OpenCV comparison: {method}")

    # Our implementation
    t0 = time.perf_counter()
    if method == "he":
        our_result = equalize_hist(y)
    elif method == "clahe":
        our_result = clahe_custom(y, **params)
    our_ms = (time.perf_counter() - t0) * 1000

    # OpenCV implementation
    t0 = time.perf_counter()
    if method == "he":
        cv_result = cv2.equalizeHist(y)
    elif method == "clahe":
        grid = params.get("grid", (8, 8))
        clip = params.get("clip_factor", 2.0)
        # OpenCV clipLimit nhận trực tiếp hệ số multiplier tương đương clip_factor
        cv_clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid[1], grid[0]))
        cv_result = cv_clahe.apply(y)
    cv_ms = (time.perf_counter() - t0) * 1000

    mae = float(np.mean(np.abs(our_result.astype(np.float64) - cv_result.astype(np.float64))))

    return {
        "our_result": our_result,
        "cv_result": cv_result,
        "mae": mae,
        "our_ms": our_ms,
        "cv_ms": cv_ms
    }


def run_benchmark(image_paths: List[Union[str, Path]], output_csv: Union[str, Path]) -> Path:
    """
    Run benchmark for each image × each method.
    Export CSV: filename, method, our_ms, cv_ms, mae, mean_before, mean_after, std_before, std_after
    For gamma: cv_ms and mae are N/A (no OpenCV equivalent).
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in image_paths:
        img = load_image(img_path)
        if img.ndim == 3:
            y, _, _ = rgb_to_ycbcr(img)
        else:
            y = img

        m_before = compute_metrics(y)

        # Gamma
        t0 = time.perf_counter()
        y_gamma = gamma_correct(y, gamma=0.6)
        gamma_ms = (time.perf_counter() - t0) * 1000
        m_gamma = compute_metrics(y_gamma)
        rows.append({
            "filename": Path(img_path).name,
            "method": "gamma",
            "our_ms": f"{gamma_ms:.3f}",
            "cv_ms": "N/A",
            "mae": "N/A",
            "mean_before": f"{m_before['mean']:.2f}",
            "mean_after": f"{m_gamma['mean']:.2f}",
            "std_before": f"{m_before['std']:.2f}",
            "std_after": f"{m_gamma['std']:.2f}"
        })

        # HE
        cmp_he = compare_with_opencv(y, "he")
        m_he = compute_metrics(cmp_he["our_result"])
        rows.append({
            "filename": Path(img_path).name,
            "method": "he",
            "our_ms": f"{cmp_he['our_ms']:.3f}",
            "cv_ms": f"{cmp_he['cv_ms']:.3f}",
            "mae": f"{cmp_he['mae']:.4f}",
            "mean_before": f"{m_before['mean']:.2f}",
            "mean_after": f"{m_he['mean']:.2f}",
            "std_before": f"{m_before['std']:.2f}",
            "std_after": f"{m_he['std']:.2f}"
        })

        # CLAHE
        clahe_params = {"grid": (8, 8), "clip_factor": 2.0}
        cmp_clahe = compare_with_opencv(y, "clahe", clahe_params)
        m_clahe = compute_metrics(cmp_clahe["our_result"])
        rows.append({
            "filename": Path(img_path).name,
            "method": "clahe",
            "our_ms": f"{cmp_clahe['our_ms']:.3f}",
            "cv_ms": f"{cmp_clahe['cv_ms']:.3f}",
            "mae": f"{cmp_clahe['mae']:.4f}",
            "mean_before": f"{m_before['mean']:.2f}",
            "mean_after": f"{m_clahe['mean']:.2f}",
            "std_before": f"{m_before['std']:.2f}",
            "std_after": f"{m_clahe['std']:.2f}"
        })

    fieldnames = [
        "filename", "method", "our_ms", "cv_ms", "mae",
        "mean_before", "mean_after", "std_before", "std_after"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path

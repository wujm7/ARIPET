"""Whole-volume SUV metrics and equal-weight within-patient aggregation."""
from collections import defaultdict
import numpy as np
from skimage.metrics import structural_similarity


def volume_metrics(prediction, reference):
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metrics require aligned finite 3D volumes")
    peak, absolute_sum, squared_sum = b.max(), np.abs(b).sum(), np.square(b).sum()
    if peak <= 0 or absolute_sum <= 0 or squared_sum <= 0:
        raise ValueError("Reference SUV cannot define the metric denominators")
    diff = a - b
    mse = np.square(diff).mean()
    window = min(7, *b.shape)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("3D SSIM requires at least three voxels on each axis")
    return {"psnr": float(10 * np.log10(peak ** 2 / mse)) if mse else float("inf"),
            "ssim": float(structural_similarity(a, b, data_range=peak, win_size=window)),
            "nmae_percent": float(100 * np.abs(diff).sum() / absolute_sum),
            "nmse": float(np.square(diff).sum() / squared_sum)}


def patient_means(scan_results):
    groups = defaultdict(list)
    for row in scan_results:
        groups[row["patient_id"]].append(row)
    keys = ("psnr", "ssim", "nmae_percent", "nmse")
    return [{"patient_id": patient, **{key: float(np.mean([r[key] for r in rows])) for key in keys}}
            for patient, rows in sorted(groups.items())]

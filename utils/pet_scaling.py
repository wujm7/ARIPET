"""Shared, invertible NASC-only scaling for SUV-valued axial slices."""
import cv2
import numpy as np

NORMALIZATION = "nasc_slice_maxabs_v1"


def resize_slice(image, size=(200, 200)):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D SUV slice")
    return cv2.resize(image, (int(size[1]), int(size[0])), interpolation=cv2.INTER_CUBIC)


def normalize_pair(nasc, asc=None, size=(200, 200)):
    """One multiplicative factor from resized NASC; never inspect ASC for scale.

    ASC may exceed one after scaling. No clipping, target min/max or offset is
    used. Constant and empty input slices are retained; a zero input uses 1.
    """
    low = resize_slice(nasc, size)
    scale = float(np.max(np.abs(low)))
    if scale == 0:
        scale = 1.0
    high = None if asc is None else resize_slice(asc, size) / scale
    return low / scale, high, np.float32(scale)


def restore_suv(prediction, scale, native_shape=None):
    restored = np.asarray(prediction, dtype=np.float32) * float(scale)
    if native_shape is not None and tuple(restored.shape) != tuple(native_shape):
        restored = resize_slice(restored, native_shape)
    return restored

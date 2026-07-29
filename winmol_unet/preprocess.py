"""Resize/normalize shared by training and inference (no TensorFlow)."""
import numpy as np
from skimage.transform import resize as _sk_resize

from .contract import IMG_SIZE

_ORDER = {"nearest": 0, "bilinear": 1, "bicubic": 3}


def to_float01(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr / 255.0).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def resize_batch(batch_nhwc, size=IMG_SIZE, mode="bicubic"):
    """Resize an NHWC float32 batch to (size, size). Deterministic, no antialias."""
    batch_nhwc = np.asarray(batch_nhwc, dtype=np.float32)
    n = batch_nhwc.shape[0]
    c = batch_nhwc.shape[3]
    out = np.empty((n, size, size, c), dtype=np.float32)
    order = _ORDER[mode]
    for i in range(n):
        out[i] = _sk_resize(
            batch_nhwc[i], (size, size), order=order,
            mode="edge", anti_aliasing=False, preserve_range=True,
        ).astype(np.float32)
    return out


def normalize_depth(arr, vmin=None, vmax=None):
    """Depth array (any numeric dtype, any shape) -> float32 in [0, 1].

    Defaults to per-image min-max (robust to unknown sensor units); pass
    vmin/vmax for a fixed physical range shared across a dataset. A constant
    image (vmax <= vmin) maps to zeros rather than dividing by zero.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min()) if vmin is None else float(vmin)
    hi = float(arr.max()) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)

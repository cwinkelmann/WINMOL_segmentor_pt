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


def normalize_depth(arr, vmin=None, vmax=None, nodata=None, nodata_fill=0.0):
    """Depth array (any numeric dtype, any shape) -> float32 in [0, 1].

    Real photogrammetric depth is not clean: it carries NaN holes where matching
    failed and sentinel values (-9999 and friends) from the exporting tool. Both
    are treated as missing rather than as extreme heights — otherwise a single
    NaN turns the whole normalized tile into NaN, and a single -9999 compresses
    every real height into the top of the range. Missing pixels are filled with
    `nodata_fill` AFTER normalization.

    `vmin`/`vmax` pin a fixed physical range shared across a dataset. Prefer them
    for real depth: per-image min-max (the default) rescales every tile
    independently, so a 0.4 m log and a 2 m root plate both become 1.0 and the
    absolute height that makes depth informative is thrown away.

    Raises ValueError when nothing valid remains — an all-nodata tile is a
    dataset problem, not something to silently emit as zeros.
    """
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    if not valid.any():
        raise ValueError("normalize_depth: no valid pixels (all NaN/nodata)")

    lo = float(arr[valid].min()) if vmin is None else float(vmin)
    hi = float(arr[valid].max()) if vmax is None else float(vmax)
    out = np.full(arr.shape, float(nodata_fill), dtype=np.float32)
    if hi <= lo:
        out[valid] = 0.0
        return out
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import tile_origins, split_tiles_spatially, corrupt_mask


def test_tiles_cover_the_image_and_stay_in_bounds():
    origins = tile_origins(H=100, W=140, tile=64, stride=32)
    assert origins, "expected some tiles"
    for r, c in origins:
        assert 0 <= r <= 100 - 64 and 0 <= c <= 140 - 64
    # the far edges are reachable (last origin touches the border)
    assert max(r for r, _ in origins) == 100 - 64
    assert max(c for _, c in origins) == 140 - 64


def test_spatial_split_is_disjoint_with_a_gap():
    origins = tile_origins(H=200, W=200, tile=32, stride=32)
    train, val = split_tiles_spatially(origins, W=200, tile=32, val_frac=0.3, gap=32)
    assert train and val
    assert not (set(map(tuple, train)) & set(map(tuple, val)))
    # every val tile lies strictly right of every train tile (+ the gap) -> no pixel overlap
    train_right = max(c + 32 for _, c in train)
    val_left = min(c for _, c in val)
    assert val_left >= train_right, "train/val tiles must not share pixels"


def test_corrupt_mask_changes_pixels_but_keeps_shape_and_dtype():
    rng = np.random.default_rng(0)
    m = np.zeros((64, 64), bool)
    m[20:40, 28:34] = True                       # a stem-ish band
    out = corrupt_mask(m, rng, drop_p=0.3, blob_p=0.5)
    assert out.shape == m.shape and out.dtype == bool
    assert not np.array_equal(out, m)            # something actually changed

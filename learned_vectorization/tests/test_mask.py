import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fields import render_mask


def test_mask_band_width_follows_diameter():
    # vertical stem at col 5, diameter 4 px -> radius 2 -> cols 3..7 foreground
    line = np.array([[2.0, 5.0], [18.0, 5.0]])
    diam = np.array([4.0, 4.0])
    mask = render_mask([line], [diam], (21, 12))

    assert mask.dtype == bool or mask.dtype == np.uint8
    m = np.asarray(mask).astype(bool)
    assert m[10, 5]                 # centre
    assert m[10, 3] and m[10, 7]    # +/- radius 2
    assert not m[10, 1]             # 4 px away -> background
    assert not m[10, 9]


def test_thicker_stem_makes_a_wider_band():
    line = np.array([[2.0, 8.0], [18.0, 8.0]])
    thin = render_mask([line], [np.array([2.0, 2.0])], (21, 17))
    thick = render_mask([line], [np.array([8.0, 8.0])], (21, 17))
    assert np.asarray(thick).sum() > np.asarray(thin).sum() * 2

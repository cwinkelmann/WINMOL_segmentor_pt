"""Tests for scripts/simulate_depth.py (synthetic terrain depth from stem masks)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from simulate_depth import bulge_from, fractal_terrain, simulate_depth_map


def _stem_mask(size=128, y0=40, y1=56, x0=10, x1=118):
    """Horizontal bar 'stem' mask (16px wide -> inradius ~8)."""
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_fractal_terrain_shape_and_amplitude():
    rng = np.random.default_rng(0)
    t = fractal_terrain((128, 128), rng, amplitude=4.0)
    assert t.shape == (128, 128) and t.dtype == np.float32
    assert 3.0 < t.std() < 5.0                      # std normalized to ~amplitude


def test_bulge_is_rounded_and_scaled_by_inradius():
    m = _stem_mask()
    h, r = bulge_from(m)
    assert h.shape == m.shape
    assert h[~m].max() == 0.0                       # bulge only inside the region
    assert abs(h.max() - r) / r < 0.15              # peak ~ inradius (half-buried cylinder)
    # rounded: interior height exceeds edge height (no flat plateau / step)
    edge_h = h[48, 10]                              # on the stem edge column
    center_h = h[48, 64]                            # stem center
    assert center_h > edge_h


def test_simulate_deterministic_per_rng_seed():
    m = _stem_mask()
    a = simulate_depth_map(m, np.random.default_rng(42))
    b = simulate_depth_map(m, np.random.default_rng(42))
    c = simulate_depth_map(m, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stems_elevated_above_nearby_ground():
    from scipy.ndimage import binary_dilation
    m = _stem_mask()
    d = simulate_depth_map(m, np.random.default_rng(0), stem_drop_p=0.0, distractors=0)
    ring = binary_dilation(m, iterations=6) & ~m    # nearby ground
    assert d[m].mean() > d[ring].mean() + 1.0       # clearly above local terrain


def test_drop_p_one_removes_mask_correlation():
    from scipy.ndimage import binary_dilation
    m = _stem_mask()
    d = simulate_depth_map(m, np.random.default_rng(0), stem_drop_p=1.0, distractors=0)
    ring = binary_dilation(m, iterations=6) & ~m
    assert abs(d[m].mean() - d[ring].mean()) < 1.0  # no bulge left on the stem


def test_distractors_add_offmask_bulges():
    m = _stem_mask()
    flat = simulate_depth_map(m, np.random.default_rng(7), stem_drop_p=1.0, distractors=0)
    with_d = simulate_depth_map(m, np.random.default_rng(7), stem_drop_p=1.0, distractors=3)
    # same rng seed consumes identical draws up to the distractor stage, so any
    # difference off-mask comes from distractor bulges
    assert (with_d - flat).max() > 1.0

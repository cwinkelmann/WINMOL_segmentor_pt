"""Inpainting mask math + the label-safety guarantee. No GPU, no diffusers."""
import numpy as np
import pytest

from synthgen.inpaint import composite, repaint_region, stems_are_intact


def _stem_mask(size=64):
    m = np.zeros((size, size), bool)
    m[28:36, 8:56] = True          # one horizontal stem
    return m


def test_repaint_region_excludes_stems():
    stem = _stem_mask()
    region = repaint_region(stem, margin_px=0)
    assert not region[stem].any()          # never repaint a labelled pixel
    assert region[0, 0]                    # far background is repaintable


def test_margin_widens_the_protected_collar():
    stem = _stem_mask()
    tight = repaint_region(stem, margin_px=0)
    wide = repaint_region(stem, margin_px=4)
    assert wide.sum() < tight.sum()        # more protected => less repaintable
    # the collar protects pixels adjacent to the stem, where blending bleeds
    assert tight[27, 30] and not wide[27, 30]


def test_composite_keeps_protected_pixels_bit_identical():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    gen = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    region = repaint_region(_stem_mask(), margin_px=2)
    out = composite(orig, gen, region)
    np.testing.assert_array_equal(out[~region], orig[~region])   # protected
    np.testing.assert_array_equal(out[region], gen[region])      # repainted


def test_stems_are_intact_detects_a_violation():
    rng = np.random.default_rng(1)
    orig = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    stem = _stem_mask()
    ok = orig.copy()
    ok[0, 0] = 255 - ok[0, 0]                       # background change is fine
    assert stems_are_intact(orig, ok, stem)
    bad = orig.copy()
    bad[30, 30] = 255 - bad[30, 30]                 # a stem pixel moved
    assert not stems_are_intact(orig, bad, stem)


def test_repaint_region_of_an_empty_mask_is_everything():
    empty = np.zeros((32, 32), bool)
    assert repaint_region(empty, margin_px=3).all()


def test_fully_covered_tile_leaves_nothing_to_repaint():
    full = np.ones((32, 32), bool)
    assert not repaint_region(full, margin_px=1).any()


def test_composite_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        composite(np.zeros((8, 8, 3), np.uint8), np.zeros((9, 9, 3), np.uint8),
                  np.zeros((8, 8), bool))

"""Figure helpers that carry a claim must be pinned like any other metric.

`stem_contrast` is the number the LOSO report uses to separate a hard fold from an
uninformative one, and `pick` decides whether a panel is a sample or a selection — which
`docs/process.md` requires be stated correctly.
"""
import numpy as np
import pytest
from PIL import Image

from scripts.fold_examples import _outline, pick
from scripts.site_gallery import _ids, stem_contrast


def _tile(d, n, stem_lum, bg_lum, size=666, band=slice(300, 340)):
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "mask").mkdir(parents=True, exist_ok=True)
    a = np.full((size, size, 3), int(bg_lum * 255), np.uint8)
    a[band] = int(stem_lum * 255)
    m = np.zeros((size, size), np.uint8)
    m[band] = 255
    Image.fromarray(a, "RGB").save(d / "train" / f"train{n}.jpeg", quality=100)
    Image.fromarray(m, "L").save(d / "mask" / f"mask{n}.gif")


def test_stem_contrast_is_positive_when_stems_are_brighter(tmp_path):
    _tile(tmp_path, 1, stem_lum=0.9, bg_lum=0.3)
    assert stem_contrast(str(tmp_path), [1]) > 0.5


def test_stem_contrast_goes_negative_when_stems_are_darker(tmp_path):
    """Bachsee_north's stems sit under foliage and read darker — the sign matters."""
    _tile(tmp_path, 1, stem_lum=0.2, bg_lum=0.7)
    assert stem_contrast(str(tmp_path), [1]) < -0.5


def test_stem_contrast_is_near_zero_for_an_invisible_stem(tmp_path):
    _tile(tmp_path, 1, stem_lum=0.5, bg_lum=0.5)
    c = stem_contrast(str(tmp_path), [1])
    assert abs(c) < 0.05 or np.isnan(c)


def test_stem_contrast_skips_tiles_with_no_stem(tmp_path):
    """A tile that is all background would divide by a meaningless spread."""
    _tile(tmp_path, 1, 0.9, 0.3, band=slice(0, 0))
    assert np.isnan(stem_contrast(str(tmp_path), [1]))


def test_outline_is_a_boundary_not_a_fill(tmp_path):
    m = np.zeros((64, 64), bool)
    m[20:44, 20:44] = True
    o = _outline(m, width=1)
    assert o.sum() < m.sum(), "outline must be thinner than the fill"
    assert not o[32, 32], "the interior must be hollow"
    assert o[20, 32], "the edge must be marked"


def test_pick_is_deterministic_in_the_seed_and_flags_unselected(tmp_path):
    for i in range(1, 9):
        _tile(tmp_path, i, 0.9, 0.3)
    a, sel_a = pick(str(tmp_path), 4, seed=1)
    b, _ = pick(str(tmp_path), 4, seed=1)
    c, _ = pick(str(tmp_path), 4, seed=2)
    assert a == b, "same seed must give the same panel"
    assert a != c, "different seeds must give different panels"
    assert sel_a is False, "a random draw must not be labelled as selected"


def test_ids_finds_the_tiles(tmp_path):
    for i in (3, 1, 2):
        _tile(tmp_path, i, 0.9, 0.3)
    assert _ids(str(tmp_path)) == [1, 2, 3]

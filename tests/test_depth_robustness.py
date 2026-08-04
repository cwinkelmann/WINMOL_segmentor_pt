"""Real-depth robustness: nodata, fixed physical ranges, and dataset preflight."""
import os
import sys

import numpy as np
import pytest
from PIL import Image

from winmol_unet.preprocess import normalize_depth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from validate_dataset import validate  # noqa: E402


# --- nodata ---------------------------------------------------------------

def test_nan_does_not_poison_the_tile():
    # real photogrammetric depth has holes; without handling, one NaN makes the
    # whole normalized tile NaN and silently destroys the sample
    a = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    out = normalize_depth(a)
    assert np.isfinite(out).all()
    assert out[0, 0] == 0.0 and out[1, 1] == 1.0     # finite values still span [0,1]


def test_nodata_sentinel_is_treated_as_missing():
    a = np.array([[-9999.0, 5.0], [10.0, 15.0]], dtype=np.float32)
    out = normalize_depth(a, nodata=-9999.0)
    assert np.isfinite(out).all()
    assert out[0, 1] == 0.0 and out[1, 1] == 1.0     # range comes from valid pixels


def test_nodata_fill_is_configurable():
    a = np.array([[np.nan, 0.0], [1.0, 2.0]], dtype=np.float32)
    assert normalize_depth(a, nodata_fill=0.0)[0, 0] == 0.0
    assert normalize_depth(a, nodata_fill=0.5)[0, 0] == 0.5


def test_all_nodata_is_rejected_not_silently_zeroed():
    with pytest.raises(ValueError, match="no valid"):
        normalize_depth(np.full((4, 4), np.nan, np.float32))


# --- fixed physical range -------------------------------------------------

def test_fixed_range_makes_tiles_comparable():
    # per-tile min-max maps a 0.4 m log and a 2 m root plate both to 1.0, which
    # throws away the absolute height that makes real depth useful
    lo = np.array([[0.0, 0.4]], np.float32)
    hi = np.array([[0.0, 2.0]], np.float32)
    assert normalize_depth(lo).max() == normalize_depth(hi).max() == 1.0
    a, b = normalize_depth(lo, vmin=0, vmax=3), normalize_depth(hi, vmin=0, vmax=3)
    assert a.max() < b.max()


# --- dataset preflight ----------------------------------------------------

def _make_ds(root, n=3, with_depth=True, size=32):
    for sub in ("train", "mask") + (("depth",) if with_depth else ()):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for i in range(1, n + 1):
        Image.fromarray((np.random.rand(size, size, 3) * 255).astype(np.uint8)).save(
            os.path.join(root, "train", f"train{i}.jpeg"))
        m = np.zeros((size, size), np.uint8)
        m[8:16, :] = 255
        Image.fromarray(m).convert("P").save(os.path.join(root, "mask", f"mask{i}.gif"))
        if with_depth:
            Image.fromarray((np.random.rand(size, size) * 65535).astype(np.uint16),
                            mode="I;16").save(os.path.join(root, "depth", f"depth{i}.png"))
    return str(root)


def test_validate_accepts_a_good_rgbd_dataset(tmp_path):
    rep = validate(_make_ds(tmp_path), rgbd=True)
    assert rep["ok"] and rep["pairs"] == 3 and not rep["errors"]


def test_validate_flags_missing_depth_when_rgbd_requested(tmp_path):
    rep = validate(_make_ds(tmp_path, with_depth=False), rgbd=True)
    assert not rep["ok"]
    assert any("depth" in e for e in rep["errors"])


def test_validate_flags_a_partial_depth_dir(tmp_path):
    # the silent killer: fewer depth files than tiles shrinks the training set
    root = _make_ds(tmp_path)
    os.remove(os.path.join(root, "depth", "depth2.png"))
    rep = validate(root, rgbd=True)
    assert not rep["ok"]
    assert any("2" in e or "depth" in e for e in rep["errors"])


def test_validate_flags_size_mismatch(tmp_path):
    root = _make_ds(tmp_path)
    Image.fromarray(np.zeros((16, 16), np.uint16), mode="I;16").save(
        os.path.join(root, "depth", "depth1.png"))
    rep = validate(root, rgbd=True)
    assert not rep["ok"]
    assert any("size" in e.lower() for e in rep["errors"])


def test_validate_flags_a_constant_depth_tile(tmp_path):
    root = _make_ds(tmp_path)
    Image.fromarray(np.zeros((32, 32), np.uint16), mode="I;16").save(
        os.path.join(root, "depth", "depth1.png"))
    rep = validate(root, rgbd=True)
    assert any("constant" in w.lower() for w in rep["warnings"])


def test_validate_reports_depth_mask_leakage(tmp_path):
    # depth that merely restates the mask makes an RGB-vs-RGBD comparison
    # meaningless; the check exists so that is noticed before training, not after
    root = _make_ds(tmp_path)
    for i in range(1, 4):
        m = np.zeros((32, 32), np.uint16)
        m[8:16, :] = 65535
        Image.fromarray(m, mode="I;16").save(os.path.join(root, "depth", f"depth{i}.png"))
    rep = validate(root, rgbd=True)
    assert rep["depth_mask_auc"] > 0.9
    assert any("leak" in w.lower() for w in rep["warnings"])

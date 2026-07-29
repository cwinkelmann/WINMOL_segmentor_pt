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


# ---- CLI tests ----

from simulate_depth import main


def _write_mask(mask_dir, n, size=96):
    os.makedirs(mask_dir, exist_ok=True)
    arr = np.zeros((size, size), np.uint8)
    arr[20 + n: 36 + n, 8: size - 8] = 255          # position varies with n
    Image.fromarray(arr).convert("P").save(os.path.join(mask_dir, f"mask{n}.gif"))


def _read_depth(ds, n):
    return np.asarray(Image.open(os.path.join(ds, "depth", f"depth{n}.png")))


def test_cli_writes_one_depth_per_mask(tmp_path):
    ds = str(tmp_path)
    for n in (1, 2, 7):
        _write_mask(os.path.join(ds, "mask"), n)
    assert main(["--dataset", ds, "--seed", "1"]) == 0
    for n in (1, 2, 7):
        arr = _read_depth(ds, n)
        assert arr.shape == (96, 96)
        assert arr.dtype in (np.uint16, np.int32)   # PIL I;16 may decode as int32
        assert arr.max() > arr.min()                # min-max scaled to uint16 range


def test_cli_deterministic_and_order_independent(tmp_path):
    ds1, ds2 = str(tmp_path / "a"), str(tmp_path / "b")
    for ds, ns in ((ds1, (1, 2)), (ds2, (2,))):     # ds2 lacks mask1
        for n in ns:
            _write_mask(os.path.join(ds, "mask"), n)
    main(["--dataset", ds1, "--seed", "5"])
    main(["--dataset", ds2, "--seed", "5"])
    # per-image seed depends on (seed, N) only, not on which other ids exist
    np.testing.assert_array_equal(_read_depth(ds1, 2), _read_depth(ds2, 2))


def test_cli_overwrite_guard(tmp_path):
    ds = str(tmp_path)
    _write_mask(os.path.join(ds, "mask"), 1)
    assert main(["--dataset", ds]) == 0
    assert main(["--dataset", ds]) == 2             # refuses without --overwrite
    assert main(["--dataset", ds, "--overwrite"]) == 0
    # verify --overwrite removes stale files: create masks {1,2}, delete mask2, re-run
    mask_dir = os.path.join(ds, "mask")
    _write_mask(mask_dir, 2)
    assert main(["--dataset", ds, "--overwrite"]) == 0  # now have depth{1,2}
    assert os.path.exists(os.path.join(ds, "depth", "depth1.png"))
    assert os.path.exists(os.path.join(ds, "depth", "depth2.png"))
    os.remove(os.path.join(mask_dir, "mask2.gif"))       # delete one mask
    assert main(["--dataset", ds, "--overwrite"]) == 0  # regenerate (only mask1 now)
    assert os.path.exists(os.path.join(ds, "depth", "depth1.png"))  # kept
    assert not os.path.exists(os.path.join(ds, "depth", "depth2.png"))  # stale file removed


def test_cli_missing_mask_dir_errors(tmp_path):
    assert main(["--dataset", str(tmp_path / "nope")]) == 2


def test_cli_mismatch_pairs_depth_with_other_mask(tmp_path):
    ds_m, ds_f = str(tmp_path / "mm"), str(tmp_path / "faith")
    for ds in (ds_m, ds_f):
        for n in (1, 2):
            _write_mask(os.path.join(ds, "mask"), n)
    main(["--dataset", ds_f, "--seed", "3"])
    main(["--dataset", ds_m, "--seed", "3", "--mismatch"])
    # mismatch: depth1 is generated from mask2 (ids rotated), under depth1's seed
    assert not np.array_equal(_read_depth(ds_m, 1), _read_depth(ds_f, 1))

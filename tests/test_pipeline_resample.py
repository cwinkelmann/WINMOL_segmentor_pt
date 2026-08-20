"""All resampling in this pipeline happens here, once, so that cutting can be a pure
window read. Two things must hold: the target grid is exactly what the arithmetic
says, and the footprint mask survives the downsample as a majority vote.
"""
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fiona")

from tests.pipeline_fixtures import ORIGIN, write_ortho  # noqa: E402
from winmol_unet.pipeline.resample import gsd_dir_name, resample  # noqa: E402


def test_gsd_dir_names_are_millimetres(tmp_path):
    assert gsd_dir_name(0.02) == "gsd_020"
    assert gsd_dir_name(0.05) == "gsd_050"
    assert gsd_dir_name(0.2) == "gsd_200"


def test_downsamples_to_the_requested_grid(tmp_path):
    src = str(tmp_path / "o.tif")
    write_ortho(src, size_m=16.0, gsd=0.02)          # 800 x 800
    stats = resample(src, str(tmp_path / "01_gsd"), gsd=0.08)

    assert stats["width"] == 200 and stats["height"] == 200
    with rasterio.open(stats["path"]) as dst:
        assert dst.res == pytest.approx((0.08, 0.08))
        assert dst.transform.c == pytest.approx(ORIGIN[0])
        assert dst.transform.f == pytest.approx(ORIGIN[1])
        assert dst.count == 3


def test_the_mask_survives_as_a_majority_vote(tmp_path):
    src = str(tmp_path / "o.tif")
    # 3 invalid source columns; at 4x each target pixel covers 4 source pixels, so
    # target column 0 is 25% valid (dropped) and column 1 is 100% valid (kept).
    write_ortho(src, size_m=16.0, gsd=0.02, blank_edge_m=0.06)
    stats = resample(src, str(tmp_path / "01_gsd"), gsd=0.08)

    with rasterio.open(stats["path"]) as dst:
        m = dst.dataset_mask()
    assert m[0, 0] == 0
    assert m[0, 1] == 255
    assert set(np.unique(m)) <= {0, 255}      # binary, never a grey edge


def test_native_gsd_is_a_passthrough_not_a_copy(tmp_path):
    src = str(tmp_path / "o.tif")
    write_ortho(src, size_m=16.0, gsd=0.02)
    stats = resample(src, str(tmp_path / "01_gsd"), gsd=0.02)
    assert stats["ratio"] == pytest.approx(1.0)
    # A copy would also satisfy "the path exists" -- and would cost a JPEG generation
    # for nothing, which is the whole reason this path is a symlink rather than a
    # write. Assert the link itself, and that it resolves back to the source: the
    # property the test's name promises, not just that some file landed there.
    assert os.path.islink(stats["path"])
    assert os.path.realpath(stats["path"]) == os.path.realpath(src)


def test_refuses_to_upsample(tmp_path):
    src = str(tmp_path / "o.tif")
    write_ortho(src, size_m=16.0, gsd=0.02)
    # Inventing detail is exactly the GenDS10 failure in data-inventory.md section 4.
    with pytest.raises(ValueError, match="finer than the source"):
        resample(src, str(tmp_path / "01_gsd"), gsd=0.01)


def test_writes_a_manifest(tmp_path):
    src = str(tmp_path / "o.tif")
    write_ortho(src, size_m=16.0, gsd=0.02)
    stats = resample(src, str(tmp_path / "01_gsd"), gsd=0.08)
    from winmol_unet.pipeline.manifest import read_manifest
    m = read_manifest(os.path.dirname(stats["path"]))
    assert m["stage"] == "resample"
    assert m["params"]["resampling"] == "average"

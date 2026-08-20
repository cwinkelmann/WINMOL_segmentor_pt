"""Clipping is a speed knob that must not become a correctness knob: the pixels it does
keep have to be bit-identical to the pixels the unclipped run would have produced, or the
tiles from a clipped run and a full run would differ.
"""
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fiona")
pytest.importorskip("shapely")

from tests.pipeline_fixtures import CRS, box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.pipeline.resample import resample  # noqa: E402


def _scene(tmp_path):
    ortho = str(tmp_path / "o.tif")
    write_ortho(ortho, size_m=32.0, gsd=0.02, seed=5)
    aoi = str(tmp_path / "aoi.gpkg")
    # Two AOIs, deliberately far apart, so the union box is much bigger than either.
    write_polygons(aoi, [box_m(2, 2, 6, 6), box_m(22, 22, 6, 6)], layer="aoi",
                   props=[{"stem_id": 1}, {"stem_id": 2}])
    return ortho, aoi


def test_clipping_shrinks_the_raster_to_the_aoi_union(tmp_path):
    ortho, aoi = _scene(tmp_path)
    full = resample(ortho, str(tmp_path / "full"), gsd=0.04, quiet=True)
    clip = resample(ortho, str(tmp_path / "clip"), gsd=0.04, clip_aoi=aoi, quiet=True)
    assert full["width"] == 800 and full["height"] == 800
    # union box spans 2 m to 28 m on both axes = 26 m -> 650 px at 4 cm before JPEG-MCU
    # alignment. Snapped outward again to whole 16-px (target GSD) blocks -- required so
    # the clipped file's JPEG tiling reproduces the same bytes an unclipped run would have
    # written (see _clip_window's docstring) -- that becomes: source px 100 floor(/32)*32
    # = 96, source px 1400 ceil(/32)*32 = 1408, so 1312 source px = 656 px at 4 cm.
    assert clip["width"] == 656 and clip["height"] == 656
    assert clip["clipped"] is True and full["clipped"] is False


def test_selecting_one_aoi_clips_to_that_one(tmp_path):
    ortho, aoi = _scene(tmp_path)
    clip = resample(ortho, str(tmp_path / "c"), gsd=0.04, clip_aoi=aoi, aoi_ids=[1],
                    quiet=True)
    # 6 m at 4 cm = 150 px before JPEG-MCU alignment; snapped outward to whole 16-px
    # blocks (source px 100 floor(/32)*32 = 96, source px 400 ceil(/32)*32 = 416) gives
    # 320 source px = 160 px at 4 cm.
    assert clip["width"] == 160 and clip["height"] == 160


def test_kept_pixels_are_identical_to_the_unclipped_run(tmp_path):
    ortho, aoi = _scene(tmp_path)
    full = resample(ortho, str(tmp_path / "full"), gsd=0.04, quiet=True)
    clip = resample(ortho, str(tmp_path / "clip"), gsd=0.04, clip_aoi=aoi, aoi_ids=[1],
                    quiet=True)
    with rasterio.open(clip["path"]) as c:
        cdata = c.read()
        cb = c.bounds
    with rasterio.open(full["path"]) as f:
        win = rasterio.windows.from_bounds(*cb, transform=f.transform)
        fdata = f.read(window=win)
    # Clipping must not change a single pixel it keeps.
    assert np.array_equal(cdata, fdata)


def test_ground_outside_the_aoi_is_masked_invalid(tmp_path):
    ortho, aoi = _scene(tmp_path)
    clip = resample(ortho, str(tmp_path / "c"), gsd=0.04, clip_aoi=aoi, quiet=True)
    with rasterio.open(clip["path"]) as c:
        m = c.dataset_mask()
    # The union box, widened to 656x656 px (26.24x26.24 m) by JPEG-MCU alignment (see
    # test_clipping_shrinks_the_raster_to_the_aoi_union); the two 6x6 m AOIs cover 72 of
    # its ~688.5 m2 -> ~0.105, still comfortably inside the brief's 0.09-0.12 band.
    frac = float((m > 0).mean())
    assert 0.09 < frac < 0.12
    # The corner between the two AOIs is inside the box and outside both AOIs.
    assert m[m.shape[0] // 2, m.shape[1] // 2] == 0


def test_clip_is_recorded_in_the_manifest(tmp_path):
    ortho, aoi = _scene(tmp_path)
    clip = resample(ortho, str(tmp_path / "c"), gsd=0.04, clip_aoi=aoi, aoi_ids=[1],
                    quiet=True)
    from winmol_unet.pipeline.manifest import read_manifest
    m = read_manifest(os.path.dirname(clip["path"]))
    assert m["params"]["clip_aoi"] is not None
    assert m["params"]["aoi_ids"] == [1]


def test_an_unmatched_aoi_id_raises_rather_than_clipping_to_nothing(tmp_path):
    ortho, aoi = _scene(tmp_path)
    with pytest.raises(ValueError, match="no AOI"):
        resample(ortho, str(tmp_path / "c"), gsd=0.04, clip_aoi=aoi, aoi_ids=[99],
                 quiet=True)

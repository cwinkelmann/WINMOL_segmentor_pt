"""The pipeline's central claim is that cutting does not interpolate. If
test_a_cut_tile_is_byte_identical_to_its_window fails, that claim is false and the
whole staging argument goes with it -- so it is the first test here.
"""
import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fiona")
pytest.importorskip("shapely")

from tests.pipeline_fixtures import box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.geo.rasterize import rasterize  # noqa: E402
from winmol_unet.pipeline.cut import cut  # noqa: E402
from winmol_unet.pipeline.layout import layout  # noqa: E402


def _laid_out(tmp_path, blank_edge_m=0.0):
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=16.0, gsd=0.02, blank_edge_m=blank_edge_m)
    aoi_path = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi_path, [box_m(0, 0, 10, 10)], layer="aoi")
    stems_path = str(tmp_path / "stems.gpkg")
    write_polygons(stems_path, [box_m(1, 1, 1, 1)], layer="stems")
    # The label raster on this raster's own grid -- the existing --rasterize mode,
    # unmodified. Windowing it is what keeps masks aligned with images.
    stem_map = str(tmp_path / "stem_map.tif")
    rasterize(stems_path, ortho, stem_map)
    lay = str(tmp_path / "02_layout")
    layout(ortho, aoi_path, stems_path, lay, mode="grid", extent_m=5.0)
    return ortho, stem_map, lay


def test_a_cut_tile_is_byte_identical_to_its_window(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)

    rec = json.loads(open(os.path.join(lay, "tiles.jsonl")).readline())
    with rasterio.open(os.path.join(out, "train", rec["id"] + ".tif")) as t:
        tile = t.read()
    with rasterio.open(ortho) as src:
        win = rasterio.windows.from_bounds(
            rec["minx"], rec["miny"], rec["maxx"], rec["maxy"], transform=src.transform)
        direct = src.read(window=win)
    # Not "close" -- equal. Any resampling at cut time would break this.
    assert np.array_equal(tile, direct)


def test_image_and_mask_share_a_grid(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    rec = json.loads(open(os.path.join(lay, "tiles.jsonl")).readline())
    with rasterio.open(os.path.join(out, "train", rec["id"] + ".tif")) as a, \
            rasterio.open(os.path.join(out, "mask", rec["id"] + ".tif")) as b:
        assert a.transform == b.transform
        assert a.crs == b.crs
        assert a.shape == b.shape


def test_tiles_are_lossless(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    rec = json.loads(open(os.path.join(lay, "tiles.jsonl")).readline())
    with rasterio.open(os.path.join(out, "train", rec["id"] + ".tif")) as t:
        # The source is already lossy JPEG; there is no reason to lose more here.
        assert t.profile["compress"].lower() == "deflate"


def test_the_exact_check_drops_what_the_overview_screen_let_through(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path, blank_edge_m=2.6)
    out = str(tmp_path / "03_tiles")
    stats = cut(lay, ortho, out, stem_map=stem_map, min_valid_frac=0.75)
    assert stats["n_dropped_exact"] >= 1
    assert stats["n_written"] + stats["n_dropped_exact"] == \
        sum(1 for _ in open(os.path.join(lay, "tiles.jsonl")))


def test_output_layout_feeds_from_folder_unchanged(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    # data/build.py pairs <src>/train with <src>/mask by shared key.
    assert sorted(os.listdir(os.path.join(out, "train"))) == \
           sorted(os.listdir(os.path.join(out, "mask")))


def test_records_carry_exact_stats(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    recs = [json.loads(l) for l in open(os.path.join(out, "tiles.jsonl"))]
    assert all(r["stage"] == "cut" for r in recs)
    assert all(0.0 <= r["valid_frac"] <= 1.0 for r in recs)
    assert any(r["stem_frac"] > 0 for r in recs)

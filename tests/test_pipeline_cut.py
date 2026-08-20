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


def _laid_out_with_overhang(tmp_path):
    """An AOI that exactly covers the raster, so --min-aoi-frac's skirt candidates
    necessarily hang off the raster edge too -- the branch's own recommended recipe
    (--clip-aoi then --min-aoi-frac < 1) produces exactly this geometry, since
    --clip-aoi shrinks the raster to the AOI union first.
    """
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=10.0, gsd=0.02)
    aoi_path = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi_path, [box_m(0, 0, 10, 10)], layer="aoi")
    stems_path = str(tmp_path / "stems.gpkg")
    write_polygons(stems_path, [box_m(1, 1, 1, 1)], layer="stems")
    stem_map = str(tmp_path / "stem_map.tif")
    rasterize(stems_path, ortho, stem_map)
    lay = str(tmp_path / "02_layout")
    layout(ortho, aoi_path, stems_path, lay, mode="grid", extent_m=5.0,
           min_aoi_frac=0.25, min_valid_frac=0.0)
    return ortho, stem_map, lay


def _assert_every_written_tile_matches_a_direct_read(ortho, lay, out):
    """The pipeline's central claim, checked against every record cut left behind --
    not just the first line of tiles.jsonl, which is always an interior tile and so
    can never see a footprint that overhangs the raster edge.

    boundless=True, fill_value=0 on the "direct" read matches production `cut()`:
    zero-fill for ground the raster doesn't cover is not resampling, so the
    byte-identity claim still applies to it.
    """
    with rasterio.open(ortho) as src:
        for line in open(os.path.join(lay, "tiles.jsonl")):
            rec = json.loads(line)
            tile_path = os.path.join(out, "train", "train" + rec["id"] + ".tif")
            if not os.path.exists(tile_path):
                continue  # dropped by min_valid_frac/min_stem_frac in cut()
            with rasterio.open(tile_path) as t:
                tile = t.read()
            win = rasterio.windows.from_bounds(
                rec["minx"], rec["miny"], rec["maxx"], rec["maxy"], transform=src.transform)
            direct = src.read(window=win, boundless=True, fill_value=0)
            # Not "close" -- equal. Any resampling at cut time would break this.
            assert np.array_equal(tile, direct), f"tile {rec['id']} does not match"


def test_a_cut_tile_is_byte_identical_to_its_window(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    _assert_every_written_tile_matches_a_direct_read(ortho, lay, out)


def test_an_overhanging_tile_is_zero_filled_not_stretched(tmp_path):
    # Before the fix: a plain (non-boundless) read of an overhanging window silently
    # returned a SMALLER array, and `dst.write` then stretched it nearest-neighbour to
    # width_px x height_px -- a corrupted tile that `test_image_and_mask_share_a_grid`
    # cannot catch, because the image and mask stretch identically.
    ortho, stem_map, lay = _laid_out_with_overhang(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map, min_valid_frac=0.0)

    recs = [json.loads(l) for l in open(os.path.join(lay, "tiles.jsonl"))]
    overhanging = [r for r in recs if r["aoi_frac"] < 1.0]
    assert overhanging, "fixture must produce at least one skirt tile"

    _assert_every_written_tile_matches_a_direct_read(ortho, lay, out)

    out_recs = {r["id"]: r for r in
                (json.loads(l) for l in open(os.path.join(out, "tiles.jsonl")))}
    for r in overhanging:
        assert out_recs[r["id"]]["valid_frac"] < 1.0


def test_image_and_mask_share_a_grid(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    rec = json.loads(open(os.path.join(lay, "tiles.jsonl")).readline())
    with rasterio.open(os.path.join(out, "train", "train" + rec["id"] + ".tif")) as a, \
            rasterio.open(os.path.join(out, "mask", "mask" + rec["id"] + ".tif")) as b:
        assert a.transform == b.transform
        assert a.crs == b.crs
        assert a.shape == b.shape


def test_tiles_are_lossless(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    rec = json.loads(open(os.path.join(lay, "tiles.jsonl")).readline())
    with rasterio.open(os.path.join(out, "train", "train" + rec["id"] + ".tif")) as t:
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
    # Call build_dataset for real. Asserting the two directory listings match is NOT a
    # test of this seam: bare `<id>.tif` names match each other perfectly and still pair
    # to nothing, because _shared_key strips a literal "train"/"mask" off the FILENAME.
    from winmol_unet.data.build import build_dataset
    ds = str(tmp_path / "04_dataset")
    build_dataset(out, ds)
    assert sorted(os.listdir(os.path.join(ds, "train"))) == \
           ["train%d.jpeg" % i for i in range(1, 5)]
    assert sorted(os.listdir(os.path.join(ds, "mask"))) == \
           ["mask%d.gif" % i for i in range(1, 5)]


def test_records_carry_exact_stats(tmp_path):
    ortho, stem_map, lay = _laid_out(tmp_path)
    out = str(tmp_path / "03_tiles")
    cut(lay, ortho, out, stem_map=stem_map)
    recs = [json.loads(l) for l in open(os.path.join(out, "tiles.jsonl"))]
    assert all(r["stage"] == "cut" for r in recs)
    # This fixture's AOI sits fully inside a raster with no blank edge, at the default
    # min_aoi_frac=1.0 -- every kept tile is strictly interior, so valid_frac must be
    # exactly 1.0, not merely "some value in range" (which cannot fail).  A regression
    # in the exact valid-pixel count -- e.g. counting the off-raster fill from the
    # boundless read as valid, or a wrong denominator -- would show up here.
    assert all(r["valid_frac"] == 1.0 for r in recs)
    assert any(r["stem_frac"] > 0 for r in recs)

"""Snapping to the raster grid is what lets stage 5 be a pure window read. If a
footprint lands off-grid, cutting has to interpolate and the pipeline's central claim
is gone -- so the snapping is tested here, and the byte-identity it buys is tested in
test_pipeline_cut.py.
"""
import json
import os

import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
pytest.importorskip("shapely")

from tests.pipeline_fixtures import CRS, ORIGIN, box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.pipeline.layout import layout, tile_geometry  # noqa: E402


def _scene(tmp_path, blank_edge_m=0.0, aoi=None):
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=16.0, gsd=0.02, blank_edge_m=blank_edge_m)
    aoi_path = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi_path, [aoi or box_m(0, 0, 10, 10)], layer="aoi")
    stems_path = str(tmp_path / "stems.gpkg")
    write_polygons(stems_path, [box_m(1, 1, 1, 1)], layer="stems")
    return ortho, aoi_path, stems_path


def test_extent_and_tile_px_are_mutually_exclusive():
    assert tile_geometry(0.02, extent_m=5.0) == (5.0, 250)
    assert tile_geometry(0.02, tile_px=250) == (5.0, 250)
    with pytest.raises(ValueError, match="exactly one"):
        tile_geometry(0.02, extent_m=5.0, tile_px=250)
    with pytest.raises(ValueError, match="exactly one"):
        tile_geometry(0.02)


def test_a_grid_tiles_the_aoi(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    stats = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0)
    assert stats["n_kept"] == 4          # 10 m AOI / 5 m tiles
    assert stats["tile_px"] == 250


def test_halving_the_stride_overlaps_the_tiles(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    stats = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0,
                   stride_frac=0.5)
    # Not 16: a tile must fit entirely inside the 10 m AOI, so the last start is at
    # 5 m, giving starts {0, 2.5, 5} on each axis. Overlap raises the count without
    # ever letting a footprint hang over the edge.
    assert stats["n_kept"] == 9


def test_every_footprint_is_snapped_to_the_raster_grid(tmp_path):
    # An AOI deliberately offset by a third of a pixel.
    ortho, aoi_path, stems_path = _scene(tmp_path, aoi=box_m(0.0067, 0.0067, 10, 10))
    out = str(tmp_path / "02_layout")
    layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0)
    with open(os.path.join(out, "tiles.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            assert abs(round((r["minx"] - ORIGIN[0]) / 0.02) * 0.02
                       - (r["minx"] - ORIGIN[0])) < 1e-9
            assert r["width_px"] == 250 and r["height_px"] == 250
            assert r["rotation"] == 0.0


def test_min_valid_frac_drops_tiles_over_the_blank_edge(tmp_path):
    # 2.5 m of blank down the left edge; a 5 m tile there is exactly 50% valid.
    ortho, aoi_path, stems_path = _scene(tmp_path, blank_edge_m=2.5)
    out = str(tmp_path / "02_layout")
    strict = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0,
                    min_valid_frac=0.75)
    assert strict["dropped_valid"] == 2       # the two tiles in the left column
    loose = layout(ortho, aoi_path, stems_path, str(tmp_path / "loose"), mode="grid",
                   extent_m=5.0, min_valid_frac=0.25)
    assert loose["dropped_valid"] == 0


def test_min_stem_frac_zero_keeps_empty_tiles(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    stats = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0,
                   min_stem_frac=0.0)
    # A hard-negative set is 99.9% background by construction; rejecting empty tiles
    # is what makes SpecDS stem-dense and is exactly wrong here.
    assert stats["n_kept"] == 4
    fracs = [json.loads(l)["stem_frac"] for l in open(os.path.join(out, "tiles.jsonl"))]
    assert sum(1 for f in fracs if f == 0.0) == 3


def test_random_mode_is_reproducible_from_its_seed(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    layout(ortho, aoi_path, stems_path, a, mode="random", extent_m=5.0,
           n_tiles=8, seed=7)
    layout(ortho, aoi_path, stems_path, b, mode="random", extent_m=5.0,
           n_tiles=8, seed=7)
    assert open(os.path.join(a, "tiles.jsonl")).read() == \
           open(os.path.join(b, "tiles.jsonl")).read()


def test_random_footprints_stay_axis_aligned_and_inside_the_aoi(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    layout(ortho, aoi_path, stems_path, out, mode="random", extent_m=5.0,
           n_tiles=8, seed=3)
    from shapely.geometry import box as shapely_box, shape
    from shapely.ops import unary_union
    with fiona.open(aoi_path, layer="aoi") as al:
        aoi_poly = unary_union([shape(f["geometry"]) for f in al])
    with fiona.open(os.path.join(out, "footprints.gpkg"), layer="footprints") as s:
        assert len(s) > 0
        for f in s:
            ring = f["geometry"]["coordinates"][0]
            xs = {round(c[0], 6) for c in ring}
            ys = {round(c[1], 6) for c in ring}
            assert len(xs) == 2 and len(ys) == 2      # a true rectangle
            fp = shapely_box(min(xs), min(ys), max(xs), max(ys))
            # "inside the AOI", not merely "a rectangle somewhere": a bounds-math
            # regression (e.g. sampling up to maxx instead of maxx - extent_m) would
            # let a footprint hang over the AOI edge without breaking axis-alignment.
            assert fp.difference(aoi_poly).area < 1e-9


def test_records_carry_the_aoi_id_for_leakage_grouping(tmp_path):
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0)
    ids = {json.loads(l)["aoi_id"] for l in open(os.path.join(out, "tiles.jsonl"))}
    assert ids == {1}


def _scene_overhanging_the_raster(tmp_path):
    """An AOI that exactly covers the raster, so a skirt candidate (min_aoi_frac < 1)
    necessarily hangs off the raster edge too -- the branch's own recommended recipe
    (--clip-aoi then --min-aoi-frac < 1) shrinks the raster to the AOI union first,
    which produces exactly this geometry.
    """
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=10.0, gsd=0.02)
    aoi_path = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi_path, [box_m(0, 0, 10, 10)], layer="aoi")
    stems_path = str(tmp_path / "stems.gpkg")
    write_polygons(stems_path, [box_m(1, 1, 1, 1)], layer="stems")
    return ortho, aoi_path, stems_path


def test_a_skirt_tile_counts_off_raster_ground_as_invalid(tmp_path):
    # Before the fix: read_masks on the overhanging window was silently averaged over
    # only the in-raster part, so a tile half off the raster edge reported
    # valid_frac == 1.0 -- exactly wrong, since valid_frac exists to measure how much
    # of a tile is real, photographed ground.
    ortho, aoi_path, stems_path = _scene_overhanging_the_raster(tmp_path)
    out = str(tmp_path / "02_layout")
    stats = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0,
                   min_aoi_frac=0.25, min_valid_frac=0.0)
    recs = [json.loads(l) for l in open(os.path.join(out, "tiles.jsonl"))]
    skirts = [r for r in recs if r["aoi_frac"] < 1.0]
    assert skirts, "fixture must produce at least one skirt tile"
    for r in skirts:
        # This AOI exactly coincides with the raster's own footprint, so the fraction
        # of the tile outside the AOI is also the fraction outside the raster: the two
        # should read back equal (within the overview screen's own tolerance).
        assert r["valid_frac"] == pytest.approx(r["aoi_frac"], abs=0.05)
        assert r["valid_frac"] < 1.0
    assert stats["regions_skipped"] == 0


def test_layout_skips_a_region_entirely_outside_the_raster(tmp_path):
    # Before the fix: the overview screen's read_masks(window=..., out_shape=...,
    # resampling=...) raised RasterioIOError as soon as any AOI region did not
    # intersect the raster at all -- reachable from a plain multi-AOI ingest, not just
    # a mistaken flight/AOI pairing.
    ortho = str(tmp_path / "ortho.tif")
    write_ortho(ortho, size_m=16.0, gsd=0.02)
    aoi_path = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi_path,
                   [box_m(0, 0, 10, 10), box_m(100, 100, 10, 10)], layer="aoi")
    stems_path = str(tmp_path / "stems.gpkg")
    write_polygons(stems_path, [box_m(1, 1, 1, 1)], layer="stems")
    out = str(tmp_path / "02_layout")
    stats = layout(ortho, aoi_path, stems_path, out, mode="grid", extent_m=5.0)
    assert stats["regions_skipped"] == 1
    assert stats["n_kept"] == 4          # only the in-raster AOI's tiles
    ids = {json.loads(l)["aoi_id"] for l in open(os.path.join(out, "tiles.jsonl"))}
    assert ids == {1}                    # the out-of-raster AOI (id 2) contributed none


def test_random_mode_refuses_min_aoi_frac(tmp_path):
    # Random candidates are drawn strictly inside [minx, maxx - extent_m], so they can
    # never straddle the AOI boundary and min_aoi_frac is silently inert. An error is
    # the fix, not skirts for random mode.
    ortho, aoi_path, stems_path = _scene(tmp_path)
    out = str(tmp_path / "02_layout")
    with pytest.raises(ValueError, match="min-aoi-frac") as exc:
        layout(ortho, aoi_path, stems_path, out, mode="random", extent_m=5.0,
               n_tiles=8, min_aoi_frac=0.25)
    assert "random" in str(exc.value)


def test_random_mode_names_the_aoi_and_extent_when_the_aoi_is_too_small(tmp_path):
    # Before the fix this raised a bare `ValueError: high - low < 0` from
    # rng.uniform, naming neither the AOI nor the extent that didn't fit inside it.
    ortho, aoi_path, stems_path = _scene(tmp_path)   # a 10 m AOI, aoi_id=1
    out = str(tmp_path / "02_layout")
    with pytest.raises(ValueError) as exc:
        layout(ortho, aoi_path, stems_path, out, mode="random", extent_m=20.0,
               n_tiles=4)
    msg = str(exc.value)
    assert "aoi_id=1" in msg
    assert "20" in msg

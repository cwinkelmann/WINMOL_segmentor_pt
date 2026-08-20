"""Edge tiles are a deliberate choice, not an accident: a model that never saw the
border of a flight will meet one in production.
"""
import json
import os

import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
pytest.importorskip("shapely")

from tests.pipeline_fixtures import box_m, write_ortho, write_polygons  # noqa: E402
from winmol_unet.pipeline.layout import layout  # noqa: E402


def _scene(tmp_path, blank_edge_m=0.0):
    ortho = str(tmp_path / "o.tif")
    write_ortho(ortho, size_m=32.0, gsd=0.02, blank_edge_m=blank_edge_m)
    aoi = str(tmp_path / "aoi.gpkg")
    write_polygons(aoi, [box_m(5, 5, 10, 10)], layer="aoi")
    stems = str(tmp_path / "stems.gpkg")
    write_polygons(stems, [box_m(6, 6, 1, 1)], layer="stems")
    return ortho, aoi, stems


def test_the_default_still_requires_full_containment(tmp_path):
    ortho, aoi, stems = _scene(tmp_path)
    out = str(tmp_path / "lay")
    stats = layout(ortho, aoi, stems, out, mode="grid", extent_m=5.0)
    assert stats["n_kept"] == 4                 # 10 m AOI, 5 m tiles, as before
    from shapely.geometry import shape
    from shapely.ops import unary_union
    with fiona.open(aoi, layer="aoi") as l:
        aoi_poly = unary_union([shape(f["geometry"]) for f in l])
    with fiona.open(os.path.join(out, "footprints.gpkg"), layer="footprints") as l:
        outside = [shape(f["geometry"]).difference(aoi_poly).area for f in l]
    # At the default, a tile may touch the AOI boundary but must never cross it.
    assert max(outside) < 1e-9
    assert all(json.loads(x)["aoi_frac"] == pytest.approx(1.0)
               for x in open(os.path.join(out, "tiles.jsonl")))


def test_lowering_min_aoi_frac_admits_edge_tiles(tmp_path):
    ortho, aoi, stems = _scene(tmp_path)
    strict = layout(ortho, aoi, stems, str(tmp_path / "a"), mode="grid", extent_m=5.0)
    loose = layout(ortho, aoi, stems, str(tmp_path / "b"), mode="grid", extent_m=5.0,
                   min_aoi_frac=0.25)
    # The AOI is 10 m; a 5 m grid over it plus a one-tile skirt is 4x4.
    assert loose["n_kept"] > strict["n_kept"]
    assert loose["n_kept"] == 16


def test_edge_tiles_actually_touch_the_aoi_boundary(tmp_path):
    ortho, aoi, stems = _scene(tmp_path)
    out = str(tmp_path / "lay")
    layout(ortho, aoi, stems, out, mode="grid", extent_m=5.0, min_aoi_frac=0.25)
    from shapely.geometry import shape
    from shapely.ops import unary_union
    with fiona.open(aoi, layer="aoi") as l:
        edge = unary_union([shape(f["geometry"]) for f in l]).boundary
    with fiona.open(os.path.join(out, "footprints.gpkg"), layer="footprints") as l:
        touching = sum(1 for f in l if shape(f["geometry"]).intersects(edge))
    # This is the property the user asked for and the old code could never satisfy.
    assert touching >= 12


def test_aoi_frac_is_recorded_per_tile(tmp_path):
    ortho, aoi, stems = _scene(tmp_path)
    out = str(tmp_path / "lay")
    layout(ortho, aoi, stems, out, mode="grid", extent_m=5.0, min_aoi_frac=0.25)
    fr = [json.loads(x)["aoi_frac"] for x in open(os.path.join(out, "tiles.jsonl"))]
    assert min(fr) >= 0.25 and max(fr) == pytest.approx(1.0)
    # A corner tile overlaps the AOI in exactly one quadrant.
    assert any(abs(f - 0.25) < 1e-6 for f in fr)


def test_edge_tiles_may_carry_black_border(tmp_path):
    # 6 m of blank down the left edge, and an AOI that starts inside it.
    ortho, aoi, stems = _scene(tmp_path, blank_edge_m=6.0)
    out = str(tmp_path / "lay")
    stats = layout(ortho, aoi, stems, out, mode="grid", extent_m=5.0,
                   min_aoi_frac=0.25, min_valid_frac=0.0)
    assert stats["dropped_valid"] == 0
    vf = [json.loads(x)["valid_frac"] for x in open(os.path.join(out, "tiles.jsonl"))]
    # The point of the exercise: tiles that are partly outside the flight footprint.
    assert min(vf) < 0.5

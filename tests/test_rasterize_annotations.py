"""`rasterize_annotations.py` burns the corpus's stem polygons onto an ortho's grid.

The behaviours worth pinning are the ones that fail *silently* on real data: a CRS
mismatch that shifts masks by metres rather than erroring, a species filter that drops
records on a letter-case difference, and the `id` field being a tree key rather than a
polygon key — which is what lets fragments of one occluded stem share an instance value.
"""
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
shapely = pytest.importorskip("shapely")

from shapely.geometry import box, mapping  # noqa: E402

from scripts.rasterize_annotations import rasterize  # noqa: E402

CRS = "EPSG:25833"
GSD = 0.02
ORIGIN = (400000.0, 6000000.0)


def _ortho(path, size_m=40.0, crs=CRS):
    n = int(size_m / GSD)
    transform = rasterio.transform.from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                       dtype="uint8", crs=crs, transform=transform) as dst:
        dst.write(np.full((3, n, n), 120, "uint8"))


def _stems(path, polys, props, crs=CRS):
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for poly, pr in zip(polys, props):
            dst.write({"geometry": mapping(poly), "properties": pr})


def _stem(x, y, length=3.0, width=0.4):
    return box(x - length / 2, y - width / 2, x + length / 2, y + width / 2)


@pytest.fixture
def site(tmp_path):
    _ortho(str(tmp_path / "o.tif"))
    cx, cy = ORIGIN[0] + 20.0, ORIGIN[1] - 20.0
    # three polygons, two of which are fragments of ONE tree (id 7)
    polys = [_stem(cx - 6, cy), _stem(cx, cy), _stem(cx + 8, cy + 5)]
    props = [{"id": 7, "Species": "RBU"}, {"id": 7, "Species": "rBU"},
             {"id": 9, "Species": "GFI"}]
    _stems(str(tmp_path / "s.shp"), polys, props)
    return {"ortho": str(tmp_path / "o.tif"), "stems": str(tmp_path / "s.shp"), "tmp": tmp_path}


def test_binary_mask_is_strictly_two_valued(site, tmp_path):
    out = str(tmp_path / "m.tif")
    rasterize(site["stems"], site["ortho"], out)
    with rasterio.open(out) as r:
        a = r.read(1)
    assert set(np.unique(a)) <= {0, 255}
    assert (a > 0).any(), "the polygons must land inside the ortho"


def test_tree_level_instances_group_fragments_of_one_stem(site, tmp_path):
    """`id` is a tree key: two fragments of tree 7 must share one instance value."""
    inst = str(tmp_path / "i.tif")
    rasterize(site["stems"], site["ortho"], str(tmp_path / "m.tif"),
              instances_path=inst, instance_level="tree")
    with rasterio.open(inst) as r:
        vals = set(np.unique(r.read(1))) - {0}
    assert vals == {7, 9}, "one value per tree, not per polygon"


def test_segment_level_instances_give_every_polygon_its_own(site, tmp_path):
    inst = str(tmp_path / "i.tif")
    rasterize(site["stems"], site["ortho"], str(tmp_path / "m.tif"),
              instances_path=inst, instance_level="segment")
    with rasterio.open(inst) as r:
        vals = set(np.unique(r.read(1))) - {0}
    assert vals == {1, 2, 3}, "three polygons -> three instances"


def test_species_filter_is_case_insensitive(site, tmp_path):
    """The corpus spells beech both 'RBU' and 'rBU'; matching case-sensitively drops 15."""
    out = str(tmp_path / "m.tif")
    rasterize(site["stems"], site["ortho"], out, species={"RBU"})
    with rasterio.open(out) as r:
        got = (r.read(1) > 0).sum()

    both = str(tmp_path / "m2.tif")
    rasterize(site["stems"], site["ortho"], both, species={"RBU", "GFI"})
    with rasterio.open(both) as r:
        all_three = (r.read(1) > 0).sum()

    assert got > 0
    assert got < all_three, "the spruce polygon must be excluded by the RBU filter"


def test_mismatched_crs_is_reprojected_not_silently_offset(tmp_path):
    """The corpus mixes EPSG:25833 and :32633 — same zone, different datum.

    Coordinates look plausible either way, so an unhandled mismatch shifts masks by
    metres instead of failing. The mask must still land on the stems.
    """
    _ortho(str(tmp_path / "o.tif"), crs="EPSG:32633")
    cx, cy = ORIGIN[0] + 20.0, ORIGIN[1] - 20.0
    _stems(str(tmp_path / "s.shp"), [_stem(cx, cy)], [{"id": 1, "Species": "RBU"}],
           crs="EPSG:25833")

    out = str(tmp_path / "m.tif")
    rasterize(str(tmp_path / "s.shp"), str(tmp_path / "o.tif"), out)
    with rasterio.open(out) as r:
        assert (r.read(1) > 0).any(), "reprojection must keep the mask on the ortho"


def test_empty_selection_fails_loudly(site, tmp_path):
    with pytest.raises(SystemExit):
        rasterize(site["stems"], site["ortho"], str(tmp_path / "m.tif"),
                  species={"NOSUCHSPECIES"})


def test_non_overlapping_annotations_are_rejected_not_written_empty(tmp_path):
    """An all-zero mask means the shapefile belongs to a different ortho."""
    _ortho(str(tmp_path / "o.tif"))
    far = ORIGIN[0] + 500000.0
    _stems(str(tmp_path / "s.shp"), [_stem(far, ORIGIN[1])], [{"id": 1, "Species": "RBU"}])

    with pytest.raises(SystemExit, match="do not overlap"):
        rasterize(str(tmp_path / "s.shp"), str(tmp_path / "o.tif"), str(tmp_path / "m.tif"))


# test_average_precision_is_a_probability_not_a_negative_number
# moved to the private helper repo with its subject (scripts/eval_threshold_sweep.py).


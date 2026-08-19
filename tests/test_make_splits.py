"""The halving split is only honest if the two halves genuinely cannot share a tile."""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("rasterio")
pytest.importorskip("fiona")

from shapely.geometry import box  # noqa: E402

from winmol_unet.geo.splits import _long_axis, halve_aoi  # noqa: E402


def test_halves_are_separated_by_the_full_buffer():
    """A tile centred in one half must not reach the other at any rotation."""
    aoi = box(0, 0, 120, 60)
    buf = 10.24 * np.sqrt(2) / 2
    a, b = halve_aoi(aoi, buf)

    assert not a.is_empty and not b.is_empty
    assert a.intersection(b).area == 0
    # the halves must be at least 2*buffer apart: buffer held out on each side of the cut
    assert a.distance(b) == pytest.approx(2 * buf, rel=0.02)


def test_cut_runs_across_the_long_axis_by_default():
    """Cutting a 120x60 AOI along its length halves the area; across it does not."""
    aoi = box(0, 0, 120, 60)
    a, b = halve_aoi(aoi, 5.0)

    # both halves should be wide in y and short in x — i.e. the cut was vertical
    for half in (a, b):
        minx, miny, maxx, maxy = half.bounds
        assert (maxy - miny) > (maxx - minx), "cut should be across the long axis"
    assert a.area == pytest.approx(b.area, rel=0.02)


def test_forced_compass_axis_overrides_the_fitted_one():
    aoi = box(0, 0, 120, 60)
    a, _ = halve_aoi(aoi, 5.0, cut_axis="ns")
    minx, miny, maxx, maxy = a.bounds
    # cutting north-south splits the y extent, leaving a wide, short half
    assert (maxx - minx) > (maxy - miny)


def test_halving_preserves_almost_all_the_area_minus_the_gap():
    aoi = box(0, 0, 100, 50)
    buf = 4.0
    a, b = halve_aoi(aoi, buf)
    lost = aoi.area - a.area - b.area
    # the gap is 2*buf wide across the 50 m width
    assert lost == pytest.approx(2 * buf * 50, rel=0.05)


def test_long_axis_of_a_wide_rectangle_points_along_its_length():
    _, direction = _long_axis(box(0, 0, 100, 10))
    assert abs(direction[0]) > abs(direction[1])


def test_an_aoi_too_small_to_halve_is_reported_not_silently_emptied():
    """A 10 m AOI cannot survive a cut that holds 7.24 m out on each side."""
    tiny = box(0, 0, 10, 10)
    a, b = halve_aoi(tiny, 7.24)
    assert a.is_empty or b.is_empty, "caller must detect this and refuse"


def _tiny_site(tmp_path):
    """A site whose stem shapefile contains a self-intersecting ring."""
    import fiona
    import numpy as np
    import rasterio
    from shapely.geometry import Polygon, box, mapping

    crs, gsd = "EPSG:25833", 0.02
    ox, oy = 400000.0, 6000000.0
    n = int(80 / gsd)
    with rasterio.open(tmp_path / "o.tif", "w", driver="GTiff", width=n, height=n,
                       count=3, dtype="uint8", crs=crs,
                       transform=rasterio.transform.from_origin(ox, oy, gsd, gsd)) as dst:
        dst.write(np.full((3, n, n), 120, "uint8"))

    cx, cy = ox + 40, oy - 40
    bowtie = Polygon([(cx, cy), (cx + 3, cy + 1), (cx, cy + 1), (cx + 3, cy)])
    assert not bowtie.is_valid, "the fixture must actually be invalid"
    stems = [bowtie] + [box(cx + dx, cy + dy, cx + dx + 3, cy + dy + 0.4)
                        for dx in range(-24, 25, 4) for dy in range(-24, 25, 4)]
    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(tmp_path / "s.shp", "w", driver="ESRI Shapefile", crs=crs,
                    schema=schema) as dst:
        for i, g in enumerate(stems):
            dst.write({"geometry": mapping(g), "properties": {"id": i, "Species": "RBU"}})
    with fiona.open(tmp_path / "a.shp", "w", driver="ESRI Shapefile", crs=crs,
                    schema={"geometry": "Polygon", "properties": {"n": "int"}}) as dst:
        dst.write({"geometry": mapping(box(cx - 30, cy - 30, cx + 30, cy + 30)),
                   "properties": {"n": 1}})
    return {"name": "S", "ortho": str(tmp_path / "o.tif"),
            "stems": str(tmp_path / "s.shp"), "aoi": str(tmp_path / "a.shp")}


def test_pipeline_repairs_geometry_before_sampling(tmp_path):
    """fix -> sample -> split must be structural, not left to the caller.

    An invalid ring aborts GEOS on the first set operation, killing a sampling run
    partway through, so the repair cannot be optional or ordered by convention.
    """
    import json

    from winmol_unet.geo.splits import run

    site = _tiny_site(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"extent_m": 10.24, "caps": {"train": 4, "test": 3},
                               "sites": [dict(site, halves=["train", "test"])]}))

    manifest = run(str(cfg), str(tmp_path / "ds"), "halve", quiet=True)

    rec = manifest["sites"][0]["geometry"]
    assert rec["repaired"] == 1, "the bowtie must have been repaired in step 1"
    assert rec["dropped"] == 0
    # the cleaned shapefile is an artefact, not just an in-memory fix
    assert (tmp_path / "ds" / "_clean" / "S.shp").exists()
    assert (tmp_path / "ds" / "_clean" / "S_geom.json").exists()
    assert manifest["sites"][0]["halves"]["train"]["tiles"] > 0


def test_skip_fix_bypasses_step_one(tmp_path):
    import json

    from winmol_unet.geo.splits import run

    site = _tiny_site(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"extent_m": 10.24, "caps": {"train": 3, "test": 2},
                               "sites": [dict(site, halves=["train", "test"])]}))

    manifest = run(str(cfg), str(tmp_path / "ds"), "halve", skip_fix=True, quiet=True)
    assert manifest["sites"][0]["geometry"] is None
    assert not (tmp_path / "ds" / "_clean").exists()

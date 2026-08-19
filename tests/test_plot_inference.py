"""World-space scoring is the only fair comparison between models at different scales.

These pin the two properties that make it fair: the reference grid is independent of the
model's own tiling, and the AOI mask actually excludes ground outside the annotated area.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from winmol_unet.geo.predict import _grid, score


def _aoi_and_stems(tmp_path, gsd=0.03, size=300):
    import fiona
    from shapely.geometry import box, mapping
    aoi = box(0, 0, size * gsd, size * gsd)
    stems = [box(1.0, 1.0, 1.3, 5.0), box(3.0, 2.0, 3.3, 6.0)]
    for name, geoms in (("aoi", [aoi]), ("stems", stems)):
        p = tmp_path / f"{name}.gpkg"
        with fiona.open(p, "w", driver="GPKG", crs="EPSG:25833",
                        schema={"geometry": "Polygon", "properties": {}}) as dst:
            for g in geoms:
                dst.write({"geometry": mapping(g), "properties": {}})
    return aoi, str(tmp_path / "aoi.gpkg"), str(tmp_path / "stems.gpkg")


def test_grid_is_independent_of_any_model_tiling():
    """Same bounds and ref GSD must give the same grid whatever tile size produced it."""
    tf, w, h = _grid((0, 0, 10, 10), 0.029297)
    assert (w, h) == (342, 342)
    assert tf.a == pytest.approx(0.029297) and tf.e == pytest.approx(-0.029297)


def test_perfect_prediction_scores_one(tmp_path):
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    ref = 0.03
    tf, W, H = _grid(aoi.bounds, ref)
    from rasterio.features import rasterize
    from winmol_unet.geo.sample import _load_geoms
    geoms, _ = _load_geoms(stems_p, rasterio.crs.CRS.from_epsg(25833), None, "s", quiet=True)
    gt = rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=tf, fill=0)
    rows = score(gt.astype(np.float32), tf, W, H, aoi, rasterio.crs.CRS.from_epsg(25833),
                 stems_p, threshold=0.5, edge_buffer_m=0.0)
    assert rows[0]["f1"] == pytest.approx(1.0, abs=1e-6)


def test_edge_buffer_shrinks_the_scored_area(tmp_path):
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    tf, W, H = _grid(aoi.bounds, 0.03)
    prob = np.zeros((H, W), np.float32)
    crs = rasterio.crs.CRS.from_epsg(25833)
    wide = score(prob, tf, W, H, aoi, crs, stems_p, edge_buffer_m=0.0)[0]
    tight = score(prob, tf, W, H, aoi, crs, stems_p, edge_buffer_m=1.0)[0]
    assert tight["valid_px"] < wide["valid_px"], "buffer must exclude boundary ground"


def test_threshold_sweep_is_monotone_in_recall(tmp_path):
    """Sanity: raising the threshold cannot increase recall."""
    aoi, aoi_p, stems_p = _aoi_and_stems(tmp_path)
    tf, W, H = _grid(aoi.bounds, 0.03)
    rng = np.random.default_rng(0)
    prob = rng.random((H, W)).astype(np.float32)
    crs = rasterio.crs.CRS.from_epsg(25833)
    rows = score(prob, tf, W, H, aoi, crs, stems_p, thresholds=[0.2, 0.5, 0.8])
    rec = [r["recall"] for r in rows]
    assert rec == sorted(rec, reverse=True)

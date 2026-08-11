"""The two coordinate traps in the Analyzer's vectoriser, pinned.

Both produce plausible-looking output rather than an error, which is why they need tests:

1. `find_segments` pads the raster by `max_tree_height / px_size` before skeletonising and
   returns coordinates still carrying that offset. At 2 cm/px that is ~1530 px — well
   inside a large orthomosaic, so every lookup succeeds and every value is wrong.
2. Paths are `(row, col)` pixel indices, not world coordinates, and `path.length` is a
   pixel count. Feeding them to `Quant._xy_to_rowcol` reads them as easting/northing.

Diameter came back 0.0000 for all 1447 stations under trap 1, and the centreline measured
29 km on a half-hectare site under trap 2.
"""
import math

import numpy as np
import pytest

from scripts.gt_centerlines import cone_volume, padding_px, sample_profile


class _Cfg:
    max_tree_height = 32


def _profile(px=0.02, w=200, h=100):
    from affine import Affine
    return {"transform": Affine(px, 0.0, 400000.0, 0.0, -px, 5800000.0),
            "width": w, "height": h}


def test_padding_matches_find_segments_arithmetic():
    """Must equal Skeletonization.find_segments' own `int(max_tree_height/px) + 1`."""
    prof = _profile(px=0.020_925_6)
    assert padding_px(_Cfg(), prof) == int(32 / 0.020_925_6) + 1 == 1530
    # coarser imagery pads fewer pixels for the same ground distance
    assert padding_px(_Cfg(), _profile(px=0.0639)) == int(32 / 0.0639) + 1


class _Stem:
    def __init__(self, coords):
        from shapely.geometry import LineString
        self.path = LineString(coords)


def test_diameter_is_twice_the_edt_at_fixed_ground_spacing():
    """A 10 px wide horizontal bar at 2 cm/px is 0.20 m across, everywhere along it."""
    px = 0.02
    prof = _profile(px=px)
    mask = np.zeros((100, 200), np.uint8)
    mask[45:55, 20:180] = 1                       # 10 px tall => 0.20 m diameter

    import scipy.ndimage as ndi
    edt = ndi.distance_transform_edt(mask.astype(bool), sampling=(px, px))

    stem = _Stem([(49.5, c) for c in range(25, 175)])     # (row, col) along the bar
    rows = sample_profile(stem, edt, prof, spacing_m=0.5, stem_id=7)

    assert len(rows) >= 5
    # stations land every 0.5 m of GROUND, not every 0.5 px
    gaps = [b["station_m"] - a["station_m"] for a, b in zip(rows, rows[1:])]
    assert all(abs(g - 0.5) < 1e-6 for g in gaps[:-1]), gaps

    # 10 px wide -> EDT peaks at 5 px = 0.10 m -> diameter 0.20 m
    mid = [r["diameter_m"] for r in rows[1:-1]]
    assert all(abs(d - 0.20) < 0.021 for d in mid), mid
    assert all(r["stem_id"] == 7 for r in rows)


def test_world_coordinates_come_back_in_the_rasters_crs():
    """x/y must be eastings/northings, not pixel indices — the GeoPackage depends on it."""
    prof = _profile(px=0.02)
    mask = np.zeros((100, 200), np.uint8)
    mask[45:55, 20:180] = 1
    import scipy.ndimage as ndi
    edt = ndi.distance_transform_edt(mask.astype(bool), sampling=(0.02, 0.02))

    rows = sample_profile(_Stem([(49.5, c) for c in range(25, 175)]), edt, prof, 0.5)
    first = rows[0]
    # transform origin is (400000, 5800000) with a 0.02 m pixel, north-up
    assert 400000 < first["x"] < 400010, first["x"]
    assert 5799990 < first["y"] <= 5800000, first["y"]


def test_cone_volume_of_a_uniform_cylinder():
    """Truncated cones over a constant diameter reduce to pi r^2 h."""
    rows = [{"station_m": k * 0.5, "diameter_m": 0.4} for k in range(5)]
    expected = math.pi * 0.2 ** 2 * 2.0            # r=0.2, total length 2.0 m
    assert cone_volume(rows, 0.5) == pytest.approx(expected, rel=1e-9)

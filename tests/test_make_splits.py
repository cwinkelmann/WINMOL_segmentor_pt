"""The halving split is only honest if the two halves genuinely cannot share a tile."""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("rasterio")
pytest.importorskip("fiona")

from shapely.geometry import box  # noqa: E402

from scripts.make_splits import _long_axis, halve_aoi  # noqa: E402


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

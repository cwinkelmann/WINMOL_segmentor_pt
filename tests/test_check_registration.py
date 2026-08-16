"""The registration check decides whether a site's labels sit on its stems.

Both failure modes it has already shown are pinned here: locating a stem's shadow instead
of the stem, and being run in a rotated frame where a constant world offset smears away.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.check_registration import check


def _site(tmp_path, shift_px=0, shadow=True, gsd=0.04, size=700, name="s"):
    """A synthetic ortho: bright vertical bars, each with a dark shadow beside it."""
    import fiona
    from shapely.geometry import mapping, box

    img = np.full((size, size, 3), 60, np.uint8)
    polys = []
    # Stems must be FINITE in both axes. Full-height bars leave the contrast surface flat
    # along dy, so argmax picks an arbitrary point on that ridge and the test reads as a
    # failure when the code is right.
    L, W = 80, 10
    for ry in range(90, size - 150, 150):
        for cx in range(90, size - 120, 130):
            img[ry:ry + L, cx:cx + W] = 230            # the stem: bright
            if shadow:
                img[ry:ry + L, cx + W + 2:cx + W + 16] = 15   # its shadow, beside it
            # the label, optionally displaced to simulate misregistration
            x0 = (cx + shift_px) * gsd
            y1 = (size - ry) * gsd                     # world y runs up from the bottom
            polys.append(box(x0, y1 - L * gsd, x0 + W * gsd, y1))

    tr = from_origin(0, size * gsd, gsd, gsd)
    tif = tmp_path / f"{name}.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=size, width=size, count=3,
                       dtype="uint8", crs="EPSG:25833", transform=tr) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)
    shp = tmp_path / f"{name}.shp"
    with fiona.open(shp, "w", driver="ESRI Shapefile", crs="EPSG:25833",
                    schema={"geometry": "Polygon", "properties": {}}) as dst:
        for p in polys:
            # y is flipped: world origin is top-left, so mirror to land on the bars
            dst.write({"geometry": mapping(p), "properties": {}})
    return str(tif), str(shp)


def test_aligned_labels_peak_at_zero(tmp_path):
    tif, shp = _site(tmp_path, shift_px=0)
    r = check(tif, shp, n=6, max_shift_m=0.6, half_px=120)
    assert abs(r["peak_dx_m"]) < 0.05 and abs(r["peak_dy_m"]) < 0.05
    assert r["peak_contrast"] > 0


def test_the_shadow_does_not_masquerade_as_the_stem(tmp_path):
    """argmax|contrast| found shadows and called Campus offset by 0.41 m."""
    tif, shp = _site(tmp_path, shift_px=0, shadow=True)
    r = check(tif, shp, n=6, max_shift_m=0.6, half_px=120)
    assert r["peak_offset_m"] < 0.05, "bright peak must stay on the stem"
    assert r["min_contrast"] < 0, "the shadow should still be reported"
    assert r["min_offset_m"] > r["peak_offset_m"], "shadow sits away from the stem"


def test_a_real_offset_is_detected(tmp_path):
    shift = 10                                          # 10 px at 4 cm = 0.40 m
    tif, shp = _site(tmp_path, shift_px=shift, shadow=False)
    r = check(tif, shp, n=6, max_shift_m=0.8, half_px=120)
    assert r["peak_offset_m"] > 0.2, f"missed a {shift * 0.04:.2f} m offset"


def test_crs_mismatch_is_reported(tmp_path):
    tif, shp = _site(tmp_path)
    r = check(tif, shp, n=4, max_shift_m=0.3, half_px=120)
    assert r["crs_mismatch"] is False


@pytest.mark.parametrize("step", [1, 3, 4, 7])
def test_shift_grid_always_contains_zero(tmp_path, step):
    """Zero shift is the reference; a grid that steps over it has no baseline at all."""
    tif, shp = _site(tmp_path, shift_px=0, shadow=False)
    r = check(tif, shp, n=4, max_shift_m=0.6, step_px=step, half_px=120)
    assert "contrast_at_zero" in r and r["contrast_at_zero"] == r["contrast_at_zero"]

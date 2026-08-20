"""Synthetic scenes for the pipeline tests.

Everything is built in tmp_path; the tests never touch the real orthomosaics, which
are 130 gigapixels. The mask band matters more here than the pixels: it is what
`--min-valid-frac` reads, so `write_ortho` gives every scene a real internal mask
rather than relying on black pixels.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
import fiona
from shapely.geometry import box, mapping

CRS = "EPSG:32633"
GSD = 0.02
ORIGIN = (400000.0, 5800000.0)


def box_m(x0, y0, w, h):
    """Rectangle `w` x `h` metres, its top-left `x0`/`y0` metres from ORIGIN."""
    left = ORIGIN[0] + x0
    top = ORIGIN[1] - y0
    return box(left, top - h, left + w, top)


def write_ortho(path, size_m=16.0, gsd=GSD, blank_edge_m=0.0, crs=CRS, seed=0,
                with_mask=True):
    """`with_mask=False` writes a plain GeoTIFF with no mask band at all --
    `MaskFlags.all_valid` in rasterio's terms, exactly what a source orthomosaic
    with no nodata/alpha/mask information looks like, and exactly what
    `resample()`'s ratio==1.0 symlink passthrough hands straight through to
    `layout()` unchanged. `read_masks()` on such a raster returns a synthetic
    array rather than one read from the file -- and, when combined with a
    boundless window, a **bool** array rather than uint8. Every other fixture in
    this module calls `write_ortho` with the default `with_mask=True`, so this is
    the one case the rest of the suite is blind to.
    """
    n = int(round(size_m / gsd))
    rng = np.random.default_rng(seed)
    # mid-grey noise, never black: nodata rejection must fire from the mask band,
    # not from a lucky dark pixel.
    data = rng.integers(60, 200, size=(3, n, n), dtype="uint8")
    mask = np.full((n, n), 255, dtype="uint8")
    if blank_edge_m:
        k = int(round(blank_edge_m / gsd))
        data[:, :, :k] = 0
        mask[:, :k] = 0
    # Without GDAL_TIFF_INTERNAL_MASK the mask lands in a sidecar .msk file, which
    # then does not travel with the tile.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                           dtype="uint8", crs=crs,
                           transform=from_origin(ORIGIN[0], ORIGIN[1], gsd, gsd)) as dst:
            dst.write(data)
            if with_mask:
                dst.write_mask(mask)


def write_polygons(path, polys, layer, crs=CRS, props=None):
    schema = {"geometry": "Polygon",
              "properties": {"stem_id": "int", "species": "int", "old_tree": "int"}}
    # "w" on GDAL's GPKG driver adds a layer to an existing GeoPackage rather than
    # truncating the file -- verified empirically (GDAL 3.9.2 / fiona 1.10.1): three
    # sequential "w" writes with three different layer names and CRSes left all three
    # layers intact with their own feature counts. "a" mode looked like the safer
    # choice but cannot create a layer that doesn't already exist yet -- it raises
    # `fiona.errors.DriverError: NULL pointer error` (GDAL's error string is empty,
    # so fiona's fallback message gives no hint this is about a missing layer) even on
    # a brand-new file, and even when a same-named layer is opened for append it can't
    # create a *different* new layer beside it. Each layer here is written in one call,
    # so "w" is also the correct semantics: two "w" writes to the SAME layer name
    # overwrite just that layer (verified: second call left it at 1 feature, not 2)
    # without touching sibling layers.
    with fiona.open(path, "w", driver="GPKG", layer=layer, crs=crs,
                    schema=schema) as dst:
        for i, poly in enumerate(polys):
            p = (props or [{}] * len(polys))[i]
            dst.write({"geometry": mapping(poly),
                       "properties": {"stem_id": p.get("stem_id", i + 1),
                                      "species": p.get("species", 1),
                                      "old_tree": p.get("old_tree", 0)}})

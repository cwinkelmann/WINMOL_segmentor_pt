"""Synthetic scenes for the pipeline tests.

Everything is built in tmp_path; the tests never touch the real orthomosaics, which
are 130 gigapixels. The mask band matters more here than the pixels: it is what
`--min-valid-frac` reads, so `write_ortho` gives every scene a real internal mask
rather than relying on black pixels.
"""
import os

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


def write_ortho(path, size_m=16.0, gsd=GSD, blank_edge_m=0.0, crs=CRS, seed=0):
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
            dst.write_mask(mask)


def write_polygons(path, polys, layer, crs=CRS, props=None):
    schema = {"geometry": "Polygon",
              "properties": {"stem_id": "int", "species": "int", "old_tree": "int"}}
    # A second layer in an existing GeoPackage needs append mode; "w" would truncate
    # the file and silently lose the layer written before it.
    mode = "a" if os.path.exists(path) else "w"
    with fiona.open(path, mode, driver="GPKG", layer=layer, crs=crs,
                    schema=schema) as dst:
        for i, poly in enumerate(polys):
            p = (props or [{}] * len(polys))[i]
            dst.write({"geometry": mapping(poly),
                       "properties": {"stem_id": p.get("stem_id", i + 1),
                                      "species": p.get("species", 1),
                                      "old_tree": p.get("old_tree", 0)}})

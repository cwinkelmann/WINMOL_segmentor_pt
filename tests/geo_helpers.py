"""Synthetic-site builder shared by tests/conftest.py's ``geo_site`` fixture and
tests/test_geo.py.

This lives in its own module rather than in conftest.py because a test module importing
``tests.conftest`` only works under pytest's default ``importmode=prepend`` and is
discouraged by pytest itself; an ordinary sibling module imports cleanly under any import
mode.

The constants are load-bearing, not decoration. ``CRS``/``GSD``/``ORIGIN`` fix the
synthetic ortho in a real projected CRS at a realistic ground resolution, and the
``rng.integers(60, 200, ...)`` noise range keeps every background pixel mid-grey --
never black -- so that nodata-collar rejection only fires where a test deliberately
blanks an edge. ``test_nodata_collar_is_rejected`` and
``test_min_stem_frac_rejects_sparse_tiles`` assert against exactly these values.

rasterio/fiona/shapely are an optional extra (`pip install -e ".[geo]"`), so every import
of them sits inside a function body -- importing this module must not require GDAL.

The private helper repo keeps its own copy of the same builder deliberately: test suites
are not importable packages across repo boundaries.
"""
import numpy as np

CRS = "EPSG:25833"
GSD = 0.02          # 2 cm/px, in the range the real orthos sit
ORIGIN = (400000.0, 6000000.0)


def _write_ortho(path, size_m=60.0, gsd=GSD, blank_edge_m=0.0):
    import rasterio

    n = int(size_m / gsd)
    transform = rasterio.transform.from_origin(ORIGIN[0], ORIGIN[1], gsd, gsd)
    rng = np.random.default_rng(0)
    # mid-grey noise: never black, so nodata rejection only fires where we make it
    data = rng.integers(60, 200, size=(3, n, n), dtype="uint8")
    if blank_edge_m:
        k = int(blank_edge_m / gsd)
        data[:, :k, :] = 0
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3,
                       dtype="uint8", crs=CRS, transform=transform) as dst:
        dst.write(data)


def _write_polygons(path, polys, props=None):
    import fiona
    from shapely.geometry import mapping

    schema = {"geometry": "Polygon", "properties": {"id": "int", "Species": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=CRS, schema=schema) as dst:
        for i, poly in enumerate(polys):
            p = (props or [{}] * len(polys))[i]
            dst.write({"geometry": mapping(poly),
                       "properties": {"id": p.get("id", i + 1),
                                      "Species": p.get("Species", "GFI")}})


def _stem(x, y, length=3.0, width=0.4, angle=0.0):
    """A stem-shaped polygon: the corpus median is 3.0 m x 0.43 m."""
    from shapely.affinity import rotate
    from shapely.geometry import box
    return rotate(box(x - length / 2, y - width / 2, x + length / 2, y + width / 2), angle)

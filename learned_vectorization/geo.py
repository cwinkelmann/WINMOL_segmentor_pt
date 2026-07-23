"""Geo-IO between a WINMOL detected-stems gpkg and the pixel-space fields.

Bridges georeferenced stems (UTM metres) <-> a pixel raster grid, so the pure-numpy field
renderer/decoder can operate in pixels. A grid is (minx, maxy, gsd): col = (x-minx)/gsd,
row = (maxy-y)/gsd (north-up, y flipped).
"""
import json

import numpy as np


class Grid:
    def __init__(self, minx, maxy, gsd, H, W):
        self.minx, self.maxy, self.gsd, self.H, self.W = minx, maxy, gsd, H, W

    @classmethod
    def from_bounds(cls, bounds, gsd, pad=8):
        minx, miny, maxx, maxy = bounds
        W = int(np.ceil((maxx - minx) / gsd)) + 2 * pad
        H = int(np.ceil((maxy - miny) / gsd)) + 2 * pad
        return cls(minx - pad * gsd, maxy + pad * gsd, gsd, H, W)

    def world_to_px(self, xy):
        xy = np.asarray(xy, float)
        col = (xy[..., 0] - self.minx) / self.gsd
        row = (self.maxy - xy[..., 1]) / self.gsd
        return np.stack([row, col], axis=-1)

    def px_to_world(self, rc):
        rc = np.asarray(rc, float)
        x = self.minx + rc[..., 1] * self.gsd
        y = self.maxy - rc[..., 0] * self.gsd
        return np.stack([x, y], axis=-1)


def read_stems(gpkg_path):
    """Return list of {xy:(N,2) world coords, d:(N,) diameters m} from the `stems` layer.
    Per-node diameters come from d_json (falls back to the nodes layer if absent)."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path, layer="stems")
    out = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        xy = np.asarray(geom.coords, float)[:, :2]
        d = None
        dj = row.get("d_json")
        if isinstance(dj, str) and dj.strip():
            d = np.asarray(json.loads(dj), float)
        if d is None or len(d) == 0:
            d = np.full(len(xy), np.nan)
        # align diameter count to vertex count (interp if off-by-few)
        if len(d) != len(xy):
            xs = np.linspace(0, 1, len(d))
            d = np.interp(np.linspace(0, 1, len(xy)), xs, d)
        out.append({"xy": xy, "d": d})
    return out


def stems_to_pixel(stems, grid):
    """World stems -> (polylines_rc, diameters_px) for render_fields."""
    polylines, diams = [], []
    for s in stems:
        polylines.append(grid.world_to_px(s["xy"]))
        diams.append(np.asarray(s["d"], float) / grid.gsd)     # metres -> pixels
    return polylines, diams


def bounds_of(stems):
    allxy = np.concatenate([s["xy"] for s in stems], axis=0)
    return float(allxy[:, 0].min()), float(allxy[:, 1].min()), \
        float(allxy[:, 0].max()), float(allxy[:, 1].max())


def polyline_length(xy):
    xy = np.asarray(xy, float)
    return float(np.hypot(*(xy[1:] - xy[:-1]).T).sum())


def stem_volume(xy, d):
    """Frustum-integrated volume (m^3): sum of pi/12 (d_i^2 + d_i d_{i+1} + d_{i+1}^2) l_i."""
    xy, d = np.asarray(xy, float), np.asarray(d, float)
    seg = np.hypot(*(xy[1:] - xy[:-1]).T)
    d0, d1 = d[:-1], d[1:]
    return float((np.pi / 12.0 * (d0 ** 2 + d0 * d1 + d1 ** 2) * seg).sum())

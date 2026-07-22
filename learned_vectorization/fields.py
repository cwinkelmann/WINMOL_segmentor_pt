"""Render dense supervision fields from centerline polylines (Approach A targets).

Pure numpy — no geo deps. Given stem centerlines in *pixel* coords and per-vertex diameters,
produce the three learning targets:
  - heat   (H,W)      Gaussian ridge along the centerlines, in [0,1]
  - orient (2,H,W)    tangent as (sin 2θ, cos 2θ) (undirected; period π)
  - diam   (H,W)      diameter (same units as the input) sampled at the ridge

Where stems overlap, the pixel is owned by the *nearest* segment (smallest perpendicular
distance), so orientation/diameter stay consistent with the strongest ridge there.
"""
import numpy as np


def _segment_projection(pr, pc, a, b):
    """For pixels (pr,pc), project onto segment a->b (row,col). Return (t in [0,1], perp dist)."""
    ar, ac = a
    br, bc = b
    vr, vc = br - ar, bc - ac
    L2 = vr * vr + vc * vc
    if L2 < 1e-9:
        d = np.hypot(pr - ar, pc - ac)
        return np.zeros_like(pr, dtype=float), d
    t = ((pr - ar) * vr + (pc - ac) * vc) / L2
    t = np.clip(t, 0.0, 1.0)
    projr, projc = ar + t * vr, ac + t * vc
    dist = np.hypot(pr - projr, pc - projc)
    return t, dist


def render_fields(polylines, diameters, shape, sigma=1.5):
    H, W = shape
    heat = np.zeros((H, W), np.float32)
    best = np.full((H, W), np.inf, np.float32)      # nearest-segment perp distance per pixel
    orient = np.zeros((2, H, W), np.float32)
    diam = np.zeros((H, W), np.float32)
    reach = max(1, int(np.ceil(3 * sigma)))

    for line, dvec in zip(polylines, diameters):
        line = np.asarray(line, float)
        dvec = np.asarray(dvec, float)
        for i in range(len(line) - 1):
            a, b = line[i], line[i + 1]
            da, db = dvec[i], dvec[i + 1]
            r0 = int(np.floor(min(a[0], b[0]))) - reach
            r1 = int(np.ceil(max(a[0], b[0]))) + reach
            c0 = int(np.floor(min(a[1], b[1]))) - reach
            c1 = int(np.ceil(max(a[1], b[1]))) + reach
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r1, H - 1), min(c1, W - 1)
            if r1 < r0 or c1 < c0:
                continue
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            t, dist = _segment_projection(rr.astype(float), cc.astype(float), a, b)
            h = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)

            sub_heat = heat[r0:r1 + 1, c0:c1 + 1]
            np.maximum(sub_heat, h, out=sub_heat)
            heat[r0:r1 + 1, c0:c1 + 1] = sub_heat

            # own a pixel where this segment is the nearest seen so far
            sub_best = best[r0:r1 + 1, c0:c1 + 1]
            take = dist < sub_best
            sub_best[take] = dist[take]
            best[r0:r1 + 1, c0:c1 + 1] = sub_best

            theta = np.arctan2(b[0] - a[0], b[1] - a[1])      # tangent angle (row,col)
            s2, c2 = np.float32(np.sin(2 * theta)), np.float32(np.cos(2 * theta))
            dloc = (da + t * (db - da)).astype(np.float32)     # interpolated diameter

            os0 = orient[0, r0:r1 + 1, c0:c1 + 1]; os0[take] = s2; orient[0, r0:r1 + 1, c0:c1 + 1] = os0
            os1 = orient[1, r0:r1 + 1, c0:c1 + 1]; os1[take] = c2; orient[1, r0:r1 + 1, c0:c1 + 1] = os1
            ds = diam[r0:r1 + 1, c0:c1 + 1]; ds[take] = dloc[take]; diam[r0:r1 + 1, c0:c1 + 1] = ds

    return heat, orient, diam

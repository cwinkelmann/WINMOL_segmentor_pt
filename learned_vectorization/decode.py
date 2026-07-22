"""Decode dense fields (heat / orient / diam) back into stem polylines + diameters.

The learned counterpart of the heuristic's skeleton->graph->connect step, but it runs on the
*clean* centerline ridge (not the noisy binary mask), and uses the orientation field to walk
straight through junctions/crossings (the heuristic's worst case).

Pipeline: threshold the ridge -> skeletonize -> pixel graph -> split into paths at
endpoints/junctions, choosing the straightest continuation (via orient) at junctions -> sample
diameter along each path. Pure numpy + skimage.
"""
import numpy as np
from skimage.morphology import skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbors(pt, pixset):
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in _NB if (r + dr, c + dc) in pixset]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _orient_vec(orient, r, c):
    # recover a unit tangent (undirected) from (sin2t, cos2t)
    s2, c2 = float(orient[0, r, c]), float(orient[1, r, c])
    theta = 0.5 * np.arctan2(s2, c2)
    return np.array([np.sin(theta), np.cos(theta)])   # (drow, dcol)


def _walk(seed, pixset, degree, junctions, orient, visited_arm):
    """Trace from an endpoint `seed`, moving with momentum. Passes straight *through*
    junctions (shared crossing pixels), so two crossing stems each keep going instead of
    fragmenting; only arm (non-junction) pixels are consumed via `visited_arm`."""
    path = [seed]
    if seed not in junctions:
        visited_arm.add(seed)
    came, cur = None, seed
    while True:
        cand = [n for n in _neighbors(cur, pixset) if n != came
                and (n in junctions or n not in visited_arm)]
        if not cand:
            break
        if came is None:
            ref = _orient_vec(orient, *cur)
            nxt = max(cand, key=lambda n: abs(float(np.dot(_unit(np.subtract(n, cur)), ref))))
        else:
            inc = _unit(np.subtract(cur, came))          # momentum: keep going straight
            nxt = max(cand, key=lambda n: float(np.dot(_unit(np.subtract(n, cur)), inc)))
        path.append(nxt)
        if nxt not in junctions:
            visited_arm.add(nxt)
        came, cur = cur, nxt
        if degree[cur] == 1:                             # reached the far endpoint
            break
    return path


def _emit(path, diam, min_pixels, out):
    if len(path) < min_pixels:
        return
    pts = np.array(path, float)
    out.append({"line": pts, "diam": np.array([diam[r, c] for r, c in path], float)})


def decode_fields(heat, orient, diam, heat_thresh=0.3, min_pixels=4):
    skel = skeletonize(heat >= heat_thresh)
    pixset = set(map(tuple, np.argwhere(skel)))
    degree = {p: len(_neighbors(p, pixset)) for p in pixset}
    junctions = {p for p in pixset if degree[p] > 2}
    endpoints = [p for p in pixset if degree[p] == 1]

    stems, visited_arm = [], set()
    for seed in endpoints:
        if seed in visited_arm:
            continue
        _emit(_walk(seed, pixset, degree, junctions, orient, visited_arm), diam, min_pixels, stems)
    # closed loops / rings with no endpoint: trace each remaining arm component once
    for p in pixset:
        if degree[p] == 2 and p not in visited_arm:
            _emit(_walk(p, pixset, degree, junctions, orient, visited_arm), diam, min_pixels, stems)
    return stems

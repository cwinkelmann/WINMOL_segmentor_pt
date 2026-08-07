"""Bridge the occlusion gaps between fragments of one stem into a single amodal mask.

    python scripts/amodal_stems.py --stems site.shp --ortho site_ortho.tif \
        --out site_amodal.tif [--instances site_amodal_inst.tif] [--report site.json]

The WINMOL annotations trace a windthrown stem as several polygons along its length,
broken wherever branches or other stems hide it, with the `id` attribute recording which
fragments belong to the same tree. Measured on the corpus: Barnekow_5 has 1,012 polygons
over 332 trees, 235 of them fragmented into a mean 3.9 pieces, and about **10% of a
stem's end-to-end extent is missing to occlusion** (Campus and Kaufland both 9%).

`rasterize_annotations.py --instance-level tree` already gives those fragments a shared
label — correct instance segmentation, but the mask still has holes. This goes further
and reconstructs the hidden spans, producing a *modally complete* (amodal) stem: one
connected polygon covering the trunk including the parts the camera never saw.

That is a different learning target, and a harder one. A model trained on modal masks
learns "mark the bark you can see"; trained on amodal masks it learns "mark where the
trunk is", which is what a downstream vectorizer needs and currently has to infer.

## How a gap is bridged

Fragments of one id are collinear by construction (measured centroid collinearity 0.006
on Barnekow_5), so:

1. fit the stem axis by PCA over every vertex of every fragment;
2. order the fragments by their projection onto that axis;
3. for each consecutive pair, join the two closest points with a segment buffered to
   half the local stem width — the width is interpolated from the two fragments being
   joined, so a tapering trunk stays tapered;
4. union fragments and bridges.

## What is deliberately not bridged

A shared `id` is not proof of a single trunk, so a bridge is refused and reported when

  * the gap exceeds `--max-gap` (default 20 m) — beyond that a shared id is more likely
    a digitizing slip than one stem;
  * the bridge would run more than `--max-offset` half-widths off the fitted axis, which
    means the fragments are not actually collinear.

Refusals are counted in `--report`, not silently dropped: a corpus where many bridges are
refused is telling you the `id` field means something other than you assumed.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_training_tiles import _repair  # noqa: E402  (shared geometry repair)


def _axis(geoms):
    """Principal direction of a stem, from every vertex of every fragment.

    Using vertices rather than centroids means a two-fragment stem still has a
    well-defined axis — two centroids define a line but say nothing about which way the
    trunk actually runs when the fragments are stubby.
    """
    pts = []
    for g in geoms:
        for poly in getattr(g, "geoms", [g]):
            pts.extend(poly.exterior.coords)
    pts = np.asarray(pts, dtype=float)
    centre = pts.mean(axis=0)
    # the dominant eigenvector of the covariance is the long direction
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    return centre, vt[0]


def _width(geom, direction):
    """Stem width = area / length along the axis. Robust to a ragged traced outline."""
    pts = np.asarray(geom.convex_hull.exterior.coords, dtype=float)
    span = float(np.ptp(pts @ direction))
    return geom.area / span if span > 1e-9 else math.sqrt(geom.area)


def _closest_points(a, b):
    from shapely.ops import nearest_points

    p, q = nearest_points(a, b)
    return (p.x, p.y), (q.x, q.y)


def bridge_tree(fragments, max_gap_m=20.0, max_offset=2.0):
    """Return (amodal geometry, list of refusals) for one tree's fragments."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    if len(fragments) < 2:
        return unary_union(fragments), []

    centre, direction = _axis(fragments)
    order = sorted(range(len(fragments)),
                   key=lambda i: (np.asarray(fragments[i].centroid.coords[0]) - centre) @ direction)
    ordered = [fragments[i] for i in order]
    widths = [_width(g, direction) for g in ordered]

    parts, refusals = list(ordered), []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        gap = a.distance(b)
        if gap <= 0:
            continue                      # already touching: nothing to bridge
        if gap > max_gap_m:
            refusals.append({"reason": "gap_too_wide", "gap_m": round(gap, 2)})
            continue
        pa, pb = _closest_points(a, b)
        # half-width interpolated across the gap keeps a tapering trunk tapered
        half = 0.5 * (widths[i] + widths[i + 1]) / 2
        v = np.asarray(pb) - np.asarray(pa)
        # 2D cross product; np.cross on 2-vectors is deprecated in numpy 2
        off = abs(float(direction[0] * v[1] - direction[1] * v[0]))
        if off > max_offset * half * 2:
            refusals.append({"reason": "not_collinear", "offset_m": round(off, 2)})
            continue
        # Overlap the fragments slightly instead of meeting them exactly. A bridge that
        # only touches is a degenerate union: at some orientations floating point leaves
        # the result a MultiPolygon, i.e. a stem that is still broken.
        n = np.linalg.norm(v)
        if n > 1e-12:
            grow = v / n * max(half * 0.25, 1e-4)
            pa, pb = tuple(np.asarray(pa) - grow), tuple(np.asarray(pb) + grow)
        # flat caps: a round cap would bulge past the fragment ends it joins
        parts.append(LineString([pa, pb]).buffer(half, cap_style=2, join_style=2))
    return unary_union(parts), refusals


def build(stems_path, ortho_path, out_path, instances_path=None, report_path=None,
          max_gap_m=20.0, max_offset=2.0, quiet=False):
    import fiona
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    src = rasterio.open(ortho_path)
    by_id = {}
    with fiona.open(stems_path) as layer:
        crs = layer.crs
        for i, f in enumerate(layer):
            if f["geometry"] is None:
                continue
            key = (f["properties"] or {}).get("id", i)
            by_id.setdefault(key, []).append(shape(f["geometry"]))
    if crs and src.crs and str(crs) != str(src.crs):
        by_id = {k: [shape(transform_geom(crs, src.crs, g.__geo_interface__)) for g in v]
                 for k, v in by_id.items()}
    by_id = {k: _repair(v, "stems", quiet=True) for k, v in by_id.items()}

    stats = {"trees": len(by_id), "fragments": sum(len(v) for v in by_id.values()),
             "bridged_trees": 0, "bridges": 0, "refusals": [],
             "modal_area_m2": 0.0, "amodal_area_m2": 0.0}
    amodal = {}
    for key, frags in by_id.items():
        geom, refused = bridge_tree(frags, max_gap_m, max_offset)
        amodal[key] = geom
        stats["modal_area_m2"] += sum(g.area for g in frags)
        stats["amodal_area_m2"] += geom.area
        n_bridges = max(0, len(frags) - 1 - len(refused))
        if len(frags) > 1 and n_bridges:
            stats["bridged_trees"] += 1
            stats["bridges"] += n_bridges
        stats["refusals"].extend(refused)

    shape_hw = (src.height, src.width)
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    for k in ("photometric", "alpha", "interleave"):
        profile.pop(k, None)
    binary = rio_rasterize([(g, 255) for g in amodal.values()], out_shape=shape_hw,
                           transform=src.transform, fill=0, dtype="uint8")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary, 1)

    if instances_path:
        keys = sorted(amodal, key=lambda k: (k is None, k))
        if len(keys) > 65535:
            raise SystemExit(f"{len(keys)} trees exceeds uint16 instance ids")
        inst = rio_rasterize([(amodal[k], i + 1) for i, k in enumerate(keys)],
                             out_shape=shape_hw, transform=src.transform, fill=0,
                             dtype="uint16")
        iprof = profile.copy(); iprof.update(dtype="uint16")
        with rasterio.open(instances_path, "w", **iprof) as dst:
            dst.write(inst, 1)

    m, a = stats["modal_area_m2"], stats["amodal_area_m2"]
    stats["reconstructed_pct"] = round(100 * (a - m) / m, 2) if m else 0.0
    stats["modal_area_m2"] = round(m, 1)
    stats["amodal_area_m2"] = round(a, 1)
    if report_path:
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=1)
    if not quiet:
        print(f"{stats['fragments']} fragments over {stats['trees']} trees")
        print(f"  bridged {stats['bridges']} gaps across {stats['bridged_trees']} trees")
        print(f"  stem area {m:.0f} -> {a:.0f} m²  (+{stats['reconstructed_pct']:.1f}% reconstructed)")
        if stats["refusals"]:
            from collections import Counter
            why = Counter(r["reason"] for r in stats["refusals"])
            print(f"  refused {len(stats['refusals'])} bridges: {dict(why)}")
    src.close()
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stems", required=True, help="polygon stem annotations with an id field")
    p.add_argument("--ortho", required=True, help="orthomosaic defining the output grid")
    p.add_argument("--out", required=True, help="binary amodal stem mask GeoTIFF")
    p.add_argument("--instances", default=None, help="uint16 amodal instance raster")
    p.add_argument("--report", default=None, help="JSON stats, including refused bridges")
    p.add_argument("--max-gap", type=float, default=20.0,
                   help="metres; refuse to bridge a wider gap (a shared id that far apart "
                        "is more likely a digitizing slip than one stem)")
    p.add_argument("--max-offset", type=float, default=2.0,
                   help="refuse a bridge running more than this many stem widths off the "
                        "fitted axis")
    a = p.parse_args(argv)
    build(a.stems, a.ortho, a.out, a.instances, a.report, a.max_gap, a.max_offset)
    return 0


if __name__ == "__main__":
    sys.exit(main())

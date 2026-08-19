"""Run the WINMOL Analyzer's vectoriser over ground-truth masks, and emit a stem
centreline with a diameter measured every N metres.

This is the vectoriser's *ceiling*: whatever it produces here, it cannot beat when fed a
real prediction, because the segmentation is perfect by construction. It is also
supervision for a learned vectoriser — a polyline plus a diameter profile per stem,
derived from the annotations rather than from a model.

    python scripts/gt_centerlines.py \\
        --stems  .../20210706_EW_WW_Kaufland.shp \\
        --ortho  .../20210706_EW_WW_Kaufland_ortho.tif \\
        --out    out/kaufland \\
        --spacing 0.5

Needs Python >= 3.10 and the Analyzer on `--analyzer` (default
`~/hnee/WINMOL_Analyzer`); `utils/IO.py` uses `str | None` annotations, so 3.9 cannot
import it.

## What comes out

* `<out>_centerlines.gpkg` — one LineString per stem, with length, mean/min/max diameter
  and truncated-cone volume.
* `<out>_stations.csv` — one row per station: `stem_id, station_m, x, y, diameter_m`,
  at exactly `--spacing` intervals from each stem's start. This is the learning target.
* `<out>_stations.gpkg` — the same stations as points, for dropping into QGIS.

## How the diameter is measured

From the Euclidean distance transform of the mask, exactly as
`Quantification.calc_v_d_edt` does: the EDT at a medial-axis pixel is the distance to the
nearest background pixel, so `diameter = 2 * EDT`, in metres via the raster's pixel size.
Reusing the Analyzer's own `_distance_transform_m` and `_xy_to_rowcol` keeps these numbers
comparable with what the Analyzer reports for a prediction.

The Analyzer measures at whatever spacing its skeleton produced. Resampling to fixed
stations is what makes the profiles comparable between stems and usable as a fixed-length
target.
"""
import argparse
import csv
import math
import os
import sys


def _load_analyzer(analyzer_path):
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"the Analyzer needs Python >= 3.10 (utils/IO.py uses `str | None` "
            f"annotations); this is {sys.version.split()[0]}")
    analyzer_path = os.path.expanduser(analyzer_path)
    if not os.path.isdir(analyzer_path):
        raise SystemExit(f"--analyzer {analyzer_path} is not a directory")
    sys.path.insert(0, analyzer_path)
    import utils.Quantification as Quant
    import utils.Skeletonization as Skel
    import utils.Vectorization as Vec
    from classes.Config import Config
    return Skel, Vec, Quant, Config


def rasterize_gt(stems_path, ortho_path, species=None, all_touched=False, aoi_path=None,
                 margin_m=40.0):
    """Burn the annotations onto the orthomosaic grid.

    Returns (mask uint8 HW, profile, instances int32 HW). `instances` carries the
    annotation's own tree id per pixel, which is what lets a vectorised stem be traced
    back to the tree it came from.

    With `aoi_path` the grid is windowed to the AOI's bounds plus `margin_m`, keeping the
    georeferencing exact (the window's own transform is used). This is not just an
    optimisation: Campus_Oberheide's orthomosaic is 38,656 x 49,427 px — 1.9 billion
    pixels — around an AOI of 111,631 m², and skeletonising the full raster does not
    finish. The margin must exceed `max_tree_height` so `connect_stems` can still reach
    across gaps at the AOI edge.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.warp import transform_geom
    from rasterio.windows import from_bounds

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rasterize_annotations import _open_annotations

    with rasterio.open(ortho_path) as src:
        profile = src.profile.copy()
        crs, transform = src.crs, src.transform
        shape = (src.height, src.width)
        if aoi_path:
            import fiona
            from shapely.geometry import shape as to_shape
            from shapely.ops import unary_union
            with fiona.open(aoi_path) as c:
                aoi_crs = c.crs
                aoi = unary_union([to_shape(f["geometry"]).buffer(0) for f in c])
            if aoi_crs and crs and str(aoi_crs) != str(crs):
                from shapely.geometry import mapping, shape as _s
                aoi = _s(transform_geom(aoi_crs, crs, mapping(aoi)))
            minx, miny, maxx, maxy = aoi.buffer(margin_m).bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets()
            win = win.round_lengths().intersection(
                rasterio.windows.Window(0, 0, src.width, src.height))
            transform = src.window_transform(win)
            shape = (int(win.height), int(win.width))

    ann_crs, feats = _open_annotations(stems_path, species)
    if not feats:
        raise SystemExit(f"no polygons in {stems_path}")
    if ann_crs and crs and str(ann_crs) != str(crs):
        feats = [(transform_geom(ann_crs, crs, g), i) for g, i in feats]

    mask = rio_rasterize([(g, 1) for g, _ in feats], out_shape=shape,
                         transform=transform, fill=0, dtype="uint8",
                         all_touched=all_touched)
    inst = rio_rasterize([(g, int(i) + 1) for g, i in feats], out_shape=shape,
                         transform=transform, fill=0, dtype="int32",
                         all_touched=all_touched)
    profile.update(count=1, dtype="uint8", nodata=0, transform=transform, crs=crs)
    return mask, profile, inst


def padding_px(config, profile):
    """The border `find_segments` adds before skeletonising, in pixels.

    `Skeletonization.find_segments` pads the raster by `max_tree_height / px_size` on all
    four sides (so `connect_stems` can search beyond the image edge) and never removes the
    offset from the coordinates it returns — the comment at Skeletonization.py:175 claims
    it does, but `np.argwhere` hands back raw padded indices. Every coordinate downstream
    is therefore shifted by this much, silently: at Kaufland's 2.09 cm/px that is 1530 px,
    which lands well inside the raster, so nothing errors — the values are just wrong.
    """
    px = abs(profile["transform"][0])
    return int(config.max_tree_height / px) + 1


def vectorize(mask, profile, Skel, Vec, Quant, Config, config=None, quiet=False):
    """The Analyzer's own Stems pipeline, fed ground truth instead of a prediction.

    Returns stems whose `path` has been shifted back into unpadded `(row, col)` raster
    space, so it indexes the same array the mask and EDT live in.
    """
    from shapely.geometry import LineString

    config = config or Config()
    segments = Skel.find_segments(mask, config, profile)
    if isinstance(segments, tuple):
        segments = segments[0]
    stems = Vec.build_stem_parts(segments)
    if not quiet:
        print(f"  skeleton -> {len(segments)} segments -> {len(stems)} stems")
    stems = Vec.connect_stems(stems, config)

    pad = padding_px(config, profile)
    h, w = mask.shape
    kept = []
    for stem in stems:
        coords = [(a - pad, b - pad) for a, b in stem.path.coords]
        if all(0 <= r < h and 0 <= c < w for r, c in coords) and len(coords) >= 2:
            stem.path = LineString(coords)
            kept.append(stem)
    if not quiet:
        print(f"  connect_stems -> {len(stems)} stems; {len(kept)} inside the raster "
              f"after removing the {pad} px skeletonisation pad")
    return kept


def sample_profile(stem, edt_m, profile, spacing_m=0.5, stem_id=0, instances=None):
    """Walk the centreline at fixed `spacing_m` and measure a diameter at each station.

    **The Analyzer vectorises in pixel space.** `Skeletonization._build_part_from_path`
    stores raw `(row, col)` skeleton indices and `build_stem_parts` wraps those in a
    LineString, so `stem.path` carries pixel coordinates and `stem.path.length` is a pixel
    count — world coordinates are applied only when the result is written out. Two
    consequences, both of which silently produce garbage rather than an error:

    * stations must be stepped in pixels (`spacing_m / pixel_size`), not metres;
    * the EDT is indexed directly by `(row, col)`; putting these through
      `Quant._xy_to_rowcol` treats pixel indices as easting/northing, lands off the
      raster, and returns diameter 0.0 for every station.
    """
    from affine import Affine

    line = stem.path
    if line is None or line.length <= 0:
        return []

    transform = profile["transform"]
    px = abs(transform.a)                       # metres per pixel
    spacing_px = spacing_m / px
    n = max(2, int(math.floor(line.length / spacing_px)) + 1)
    h, w = edt_m.shape

    rows = []
    for k in range(n):
        d_px = min(k * spacing_px, line.length)
        p = line.interpolate(d_px)
        row, col = int(round(p.x)), int(round(p.y))     # path coords are (row, col)
        if 0 <= row < h and 0 <= col < w:
            diameter = 2.0 * float(edt_m[row, col])     # EDT already in metres
            tree = int(instances[row, col]) if instances is not None else 0
        else:
            diameter, tree = 0.0, 0
        # pixel centre -> world; rasterio's transform maps (col, row)
        wx, wy = transform * (col + 0.5, row + 0.5)
        rows.append({"stem_id": stem_id, "station_m": round(d_px * px, 3),
                     "x": wx, "y": wy, "diameter_m": round(diameter, 4),
                     "gt_tree_id": tree})
    return rows


def cone_volume(rows, spacing):
    """Truncated-cone volume between consecutive stations — the Analyzer's own model."""
    v = 0.0
    for a, b in zip(rows, rows[1:]):
        r1, r2 = a["diameter_m"] / 2, b["diameter_m"] / 2
        h = b["station_m"] - a["station_m"]
        v += math.pi * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2)
    return v


def run(stems_path, ortho_path, out_prefix, spacing=0.5, analyzer="~/hnee/WINMOL_Analyzer",
        species=None, all_touched=False, min_length_m=None, quiet=False, aoi_path=None):
    import numpy as np
    from shapely.geometry import LineString

    Skel, Vec, Quant, Config = _load_analyzer(analyzer)
    config = Config()
    if min_length_m is not None:
        config.min_length = min_length_m

    if not quiet:
        print(f"rasterising {os.path.basename(stems_path)} onto "
              f"{os.path.basename(ortho_path)}")
    mask, profile, instances = rasterize_gt(stems_path, ortho_path, species,
                                           all_touched, aoi_path)
    px, py = Quant._pixel_size(profile)
    if not quiet:
        print(f"  {mask.shape[1]}x{mask.shape[0]} px at {px*100:.2f} cm, "
              f"{int(mask.sum())} stem px ({100*mask.mean():.3f}%)")

    stems = vectorize(mask, profile, Skel, Vec, Quant, Config, config, quiet)
    edt_m = Quant._distance_transform_m(mask, profile)
    if not quiet:
        # The Analyzer already refines the skeleton to Config.measuring_point_spacing_m,
        # so its own vertices are usually the stations you want; --spacing resamples on
        # top of that only if you ask for something else.
        gaps = [math.dist(c[i], c[i + 1]) * px
                for s_ in stems for c in [list(s_.path.coords)]
                for i in range(len(c) - 1)]
        if gaps:
            print(f"  vectoriser's own station spacing: mean {sum(gaps)/len(gaps):.2f} m "
                  f"(Config.measuring_point_spacing_m = {config.measuring_point_spacing_m})")

    all_rows, lines = [], []
    for i, stem in enumerate(stems, start=1):
        rows = sample_profile(stem, edt_m, profile, spacing, i, instances)
        if len(rows) < 2:
            continue
        d = [r["diameter_m"] for r in rows]
        tree_ids = [r["gt_tree_id"] for r in rows if r["gt_tree_id"]]
        # stem.path is in pixel (row, col); rebuild it in world coordinates so the
        # GeoPackage lands on the map rather than near the origin
        world = LineString([(r["x"], r["y"]) for r in rows])
        lines.append({
            "stem_id": i,
            "length_m": round(stem.path.length * px, 3),
            "stations": len(rows),
            "d_mean_m": round(sum(d) / len(d), 4),
            "d_min_m": round(min(d), 4),
            "d_max_m": round(max(d), 4),
            "volume_m3": round(cone_volume(rows, spacing), 5),
            # the annotation id under most of the centreline: the tree this stem *is*
            "gt_tree_id": max(set(tree_ids), key=tree_ids.count) if tree_ids else 0,
            "geometry": world,
        })
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = f"{out_prefix}_stations.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["stem_id", "station_m", "x", "y",
                                           "diameter_m", "gt_tree_id"])
        wr.writeheader()
        wr.writerows(all_rows)

    written = {"stations_csv": csv_path}
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(lines, crs=profile["crs"])
        gdf.to_file(f"{out_prefix}_centerlines.gpkg", driver="GPKG")
        pts = gpd.GeoDataFrame(all_rows, crs=profile["crs"],
                               geometry=[Point(r["x"], r["y"]) for r in all_rows])
        pts.to_file(f"{out_prefix}_stations.gpkg", driver="GPKG")
        written["centerlines"] = f"{out_prefix}_centerlines.gpkg"
        written["stations_gpkg"] = f"{out_prefix}_stations.gpkg"
    except Exception as e:                       # geopandas is optional; the CSV is not
        if not quiet:
            print(f"  (no GeoPackage: {type(e).__name__}: {e})")

    if not quiet and lines:
        tot = sum(l["length_m"] for l in lines)
        dm = [l["d_mean_m"] for l in lines]
        print(f"\n{len(lines)} stems, {tot:.1f} m of centreline, "
              f"{len(all_rows)} stations at {spacing} m")
        print(f"  diameter  mean {sum(dm)/len(dm)*100:.1f} cm   "
              f"range {min(l['d_min_m'] for l in lines)*100:.1f}"
              f"-{max(l['d_max_m'] for l in lines)*100:.1f} cm")
        print(f"  volume    {sum(l['volume_m3'] for l in lines):.2f} m³")
        for k, v in written.items():
            print(f"  {k:16s} {v}")
    return {"stems": lines, "stations": all_rows, "written": written}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stems", required=True, help="stem polygon annotations")
    p.add_argument("--ortho", required=True, help="orthomosaic defining the raster grid")
    p.add_argument("--out", required=True, help="output path prefix")
    p.add_argument("--spacing", type=float, default=0.5,
                   help="metres between diameter measurements (default 0.5)")
    p.add_argument("--analyzer", default="~/hnee/WINMOL_Analyzer")
    p.add_argument("--species", nargs="*", default=None)
    p.add_argument("--all-touched", action="store_true",
                   help="burn every pixel a polygon touches, not only centre hits")
    p.add_argument("--aoi", default=None,
                   help="window the raster to this AOI's bounds + 40 m. Needed on large "
                        "orthomosaics: Oberheide is 1.9 gigapixels around a 111,631 m2 AOI")
    p.add_argument("--min-length", type=float, default=None,
                   help="override Config.min_length (m); the Analyzer default is 2.0")
    a = p.parse_args(argv)

    res = run(a.stems, a.ortho, a.out, a.spacing, a.analyzer, a.species,
              a.all_touched, a.min_length, aoi_path=a.aoi)
    return 0 if res["stems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

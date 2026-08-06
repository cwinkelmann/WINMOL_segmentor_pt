"""Sample rotated training tiles from an orthomosaic + stem annotations.

    python scripts/sample_training_tiles.py --ortho site_ortho.tif \
        --stems site.shp --aoi site_AOE.shp --out /path/to/SpecDS

This is the Python port of the original R generator kept at
`docs/reference/241202_Training_data_SpecDS.R`, which produced the SpecDS/TestDS
the R model was trained on. Reproducing its sampling matters: a plain grid over
one of these orthomosaics yields tiles that are ~99% background, because stems
cover only about 1% of a site. The R recipe, and this port, instead:

  1. draw a random point inside the **windthrow polygon** (the `*_AOE.shp` /
     `trainingscluster` layer), never the whole ortho — outside that polygon the
     stems are real but were never digitized, so tiles from there would teach the
     model that visible stems are background;
  2. shrink that polygon inward by half the footprint diagonal first, so a
     *rotated* footprint still lands entirely inside annotated ground (the R
     script's -11 m for a 15 m square is 10.61 m rounded up);
  3. cut a square `--extent` metres across at a **uniformly random rotation**,
     which is why the R-trained model saw rotated stems without rotation
     augmentation in the training loop;
  4. **reject** the tile unless stems cover at least `--min-stem-frac` of it
     (R: `SmplExtend^2/200`, i.e. 0.5%). This is the step that makes the dataset
     stem-dense;
  5. oversample heavily — R asked for `area / extent² × 100` attempts, so tiles
     overlap and the same stem is seen from many offsets and angles.

Output follows the loader convention in `training/dataset.py`: `train/train{N}.jpeg`
paired with `mask/mask{N}.gif` by the integer N.

Every tile is resampled to `--tile-px` (512), so the effective ground resolution is
`extent / tile_px` — 2.9 cm/px at the defaults. Note this is a *resampling*, not a
crop: a 1.5 cm/px ortho is downsampled to reach it and a 6.4 cm/px ortho is
upsampled. The script reports the factor per site so a mixed-resolution corpus does
not silently train on invented detail.
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def _repair(geoms, label, quiet=False):
    """Make hand-digitized polygons safe for GEOS set operations.

    Tracing stems by hand produces self-intersecting rings — Campus has them. GEOS
    raises `TopologyException: side location conflict` on the first intersection with
    one, which would abort a sampling run tens of thousands of tiles in. make_valid
    repairs the ring; it can return a collection, so keep only the polygonal parts.
    """
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon

    out, fixed, dropped = [], 0, 0
    for g in geoms:
        if g.is_valid:
            out.append(g)
            continue
        r = make_valid(g)
        parts = [p for p in getattr(r, "geoms", [r]) if isinstance(p, (Polygon, MultiPolygon))]
        parts = [p for p in parts if not p.is_empty and p.area > 0]
        if parts:
            out.append(parts[0] if len(parts) == 1 else MultiPolygon(
                [q for p in parts for q in getattr(p, "geoms", [p])]))
            fixed += 1
        else:
            dropped += 1
    if (fixed or dropped) and not quiet:
        print(f"{label}: repaired {fixed} invalid geometries"
              + (f", dropped {dropped} with no area" if dropped else ""))
    return out


def _load_geoms(path, dst_crs, species=None, label="layer", quiet=False):
    """Read a vector layer, reproject to `dst_crs`, return valid shapely geometries."""
    import fiona
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    with fiona.open(path) as src:
        crs = src.crs
        out = []
        for f in src:
            g = f["geometry"]
            if g is None:
                continue
            if species:
                sp = (f["properties"] or {}).get("Species")
                # the corpus spells beech both 'RBU' and 'rBU'
                if sp is None or str(sp).strip().upper() not in species:
                    continue
            out.append(shape(g))
    if crs and dst_crs and str(crs) != str(dst_crs):
        out = [shape(transform_geom(crs, dst_crs, g.__geo_interface__)) for g in out]
    # repair after reprojection: that step can itself produce invalid rings
    return _repair(out, label, quiet), crs


def _random_points(poly, n, rng):
    """Uniform points inside a polygon, by rejection sampling its bounding box."""
    from shapely.geometry import Point

    minx, miny, maxx, maxy = poly.bounds
    pts = []
    # the bbox hit rate bounds how many draws we need; cap the work rather than
    # spinning forever on a pathological sliver
    for _ in range(200):
        if len(pts) >= n:
            break
        k = max(64, int((n - len(pts)) * 2))
        xs = rng.uniform(minx, maxx, k)
        ys = rng.uniform(miny, maxy, k)
        pts += [(x, y) for x, y in zip(xs, ys) if poly.contains(Point(x, y))]
    return pts[:n]


def _footprint(cx, cy, extent, angle_deg):
    """The rotated square actually sampled, in world coordinates."""
    from shapely.affinity import rotate
    from shapely.geometry import box

    h = extent / 2.0
    return rotate(box(cx - h, cy - h, cx + h, cy + h), angle_deg, origin=(cx, cy))


def sample_tiles(ortho, stems, aoi, out_dir, extent_m=15.0, tile_px=512,
                 oversample=100.0, inward_buffer_m=None, min_stem_frac=1 / 200.0,
                 max_nodata_frac=0.02, seed=1, species=None, limit=None,
                 start_index=1, rotate=True, quiet=False):
    import rasterio
    from affine import Affine
    from PIL import Image
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.windows import from_bounds
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    rng = np.random.default_rng(seed)
    src = rasterio.open(ortho)
    gsd = abs(src.transform.a)

    stem_geoms, _ = _load_geoms(stems, src.crs, species, "stems", quiet)
    if not stem_geoms:
        raise SystemExit(f"no stem polygons selected from {stems}")
    tree = STRtree(stem_geoms)

    aoi_geoms, _ = _load_geoms(aoi, src.crs, None, "aoi", quiet)
    area = unary_union(aoi_geoms)
    # half the diagonal: the smallest inward buffer for which a footprint at ANY
    # rotation is still inside the annotated area
    buf = extent_m * math.sqrt(2) / 2 if inward_buffer_m is None else inward_buffer_m
    inner = area.buffer(-buf)
    if inner.is_empty:
        raise SystemExit(
            f"the AOI shrinks to nothing under a {buf:.2f} m inward buffer "
            f"(its area is {area.area:.0f} m²). Use a smaller --extent, or pass an "
            f"explicit --inward-buffer if you accept footprints crossing the AOI edge.")

    n_attempts = int(inner.area / (extent_m ** 2) * oversample)
    if limit:
        # still oversample, but do not build a 40k-point list to write 20 tiles
        n_attempts = min(n_attempts, max(limit * 20, 64))

    scale = (extent_m / tile_px) / gsd
    if not quiet:
        print(f"ortho {gsd * 100:.2f} cm/px -> tiles {extent_m / tile_px * 100:.2f} cm/px "
              f"({'downsample' if scale > 1 else 'upsample'} x{max(scale, 1 / scale):.2f})")
        print(f"AOI {area.area:.0f} m² -> {inner.area:.0f} m² after a {buf:.2f} m inward "
              f"buffer; {n_attempts} attempts at {oversample:g}x oversampling")

    img_dir = os.path.join(out_dir, "train")
    msk_dir = os.path.join(out_dir, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    # read a window big enough to hold the footprint at any rotation, rotate the
    # pixels, then centre-crop back to the footprint
    half_diag = extent_m * math.sqrt(2) / 2
    win_px = int(math.ceil(2 * half_diag / gsd))
    crop_px = int(round(extent_m / gsd))

    n = start_index
    stats = {"attempts": 0, "no_stem": 0, "too_few_stems": 0, "nodata": 0, "written": 0}
    coverages = []
    angles = rng.uniform(-179.0, 180.0, n_attempts) if rotate else np.zeros(n_attempts)

    for (cx, cy), angle in zip(_random_points(inner, n_attempts, rng), angles):
        stats["attempts"] += 1
        if limit and stats["written"] >= limit:
            break
        # Cheap pre-filter only. Use the circle that circumscribes the footprint at any
        # rotation, so this is a superset regardless of which way the crop turns — the
        # real decision is made on the finished mask below.
        disc = Point(cx, cy).buffer(half_diag)
        hits = [stem_geoms[i] for i in tree.query(disc)]
        hits = [g for g in hits if g.intersects(disc)]
        if not hits:
            stats["no_stem"] += 1
            continue

        win = from_bounds(cx - half_diag, cy - half_diag, cx + half_diag, cy + half_diag,
                          transform=src.transform)
        # boundless: an AOI touching the ortho edge still yields a full window
        arr = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0,
                       out_shape=(3, win_px, win_px))
        # the read resampled the window to win_px, so the array's transform is the
        # window's transform scaled by that ratio — the mask must be burned on the
        # same grid or it lands fractions of a metre off the imagery
        wt = src.window_transform(win) * Affine.scale(win.width / win_px,
                                                      win.height / win_px)

        rgb = Image.fromarray(np.transpose(arr, (1, 2, 0)).astype("uint8"))
        mask_arr = rio_rasterize([(g, 255) for g in hits], out_shape=(win_px, win_px),
                                 transform=wt, fill=0, dtype="uint8")
        msk = Image.fromarray(mask_arr)

        if angle:
            # image and mask rotate together, so they stay registered to each other
            rgb = rgb.rotate(angle, resample=Image.BICUBIC)
            msk = msk.rotate(angle, resample=Image.NEAREST)
        off = (win_px - crop_px) // 2
        rgb = rgb.crop((off, off, off + crop_px, off + crop_px))
        msk = msk.crop((off, off, off + crop_px, off + crop_px))

        a = np.asarray(rgb)
        if float((a.max(axis=2) == 0).mean()) > max_nodata_frac:
            # ortho collar / gap in coverage — black pixels are not forest
            stats["nodata"] += 1
            continue

        # Decide on the mask that will actually be written, not on a world-space
        # prediction of it. PIL rotates the image counter-clockwise, which samples a
        # world square rotated the *other* way; testing the world footprint instead let
        # 3.2% of tiles through below the floor, 21 of them completely empty.
        # Measuring the finished mask is exact by construction.
        cov_tile = float((np.asarray(msk) > 0).mean())
        if cov_tile < min_stem_frac:
            stats["too_few_stems"] += 1
            continue

        rgb = rgb.resize((tile_px, tile_px), Image.BICUBIC)
        msk = msk.resize((tile_px, tile_px), Image.NEAREST)
        # the loader binarizes anyway; keep the gif strictly two-valued
        msk = Image.fromarray(((np.asarray(msk) > 127) * 255).astype("uint8"))

        rgb.save(os.path.join(img_dir, f"train{n}.jpeg"), quality=95)
        msk.save(os.path.join(msk_dir, f"mask{n}.gif"))
        coverages.append(float((np.asarray(msk) > 0).mean()))
        stats["written"] += 1
        n += 1

    src.close()
    stats["mean_stem_coverage"] = round(float(np.mean(coverages)), 5) if coverages else 0.0
    stats["next_index"] = n
    if not quiet:
        print(f"wrote {stats['written']} tiles to {out_dir} "
              f"(mean stem coverage {100 * stats['mean_stem_coverage']:.2f}%)")
        print(f"rejected: {stats['no_stem']} empty, {stats['too_few_stems']} below "
              f"{100 * min_stem_frac:.2f}% stem, {stats['nodata']} over nodata")
        if stats["written"] == 0:
            print("no tiles were written — check that the stems, AOI and ortho are the "
                  "same site, and that the AOI is not tiny relative to --extent",
                  file=sys.stderr)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ortho", required=True)
    p.add_argument("--stems", required=True, help="polygon stem annotations (.shp)")
    p.add_argument("--aoi", required=True,
                   help="windthrow / training-cluster polygon (*_AOE.shp); sampling "
                        "never leaves it, because stems outside it are undigitized")
    p.add_argument("--out", required=True, help="dataset root; gets train/ and mask/")
    p.add_argument("--extent", type=float, default=15.0, help="footprint side in metres")
    p.add_argument("--tile-px", type=int, default=512)
    p.add_argument("--oversample", type=float, default=100.0,
                   help="attempts per footprint-sized cell of the AOI (R used 100)")
    p.add_argument("--inward-buffer", type=float, default=None,
                   help="metres; default is half the footprint diagonal")
    p.add_argument("--min-stem-frac", type=float, default=1 / 200.0,
                   help="reject tiles with less stem than this (R used 1/200)")
    p.add_argument("--max-nodata-frac", type=float, default=0.02)
    p.add_argument("--species", nargs="*", default=None,
                   help="keep only these species codes, case-insensitive")
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="stop after N tiles")
    p.add_argument("--start-index", type=int, default=1,
                   help="first N in train{N}.jpeg; use to append sites into one dataset")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stats-json", default=None)
    a = p.parse_args(argv)

    species = {s.strip().upper() for s in a.species} if a.species else None
    stats = sample_tiles(a.ortho, a.stems, a.aoi, a.out, a.extent, a.tile_px,
                         a.oversample, a.inward_buffer, a.min_stem_frac,
                         a.max_nodata_frac, a.seed, species, a.limit, a.start_index,
                         not a.no_rotate)
    if a.stats_json:
        with open(a.stats_json, "w") as f:
            json.dump(stats, f, indent=1)
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    sys.exit(main())

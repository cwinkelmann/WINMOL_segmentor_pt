"""Faithful port of the original R SpecDS generator.

    python scripts/r_sample_tiles.py --ortho site_ortho.tif --stems site.shp \
        --aoi site_AOE.shp --out /path/to/SpecDS

This reproduces `docs/reference/241202_Training_data_SpecDS.R` as closely as Python
allows, so datasets built here can be compared against the R model's training data on
equal terms. For new work prefer `sample_training_tiles.py`, which fixes several things
the R script gets wrong or leaves implicit — the differences are listed at the bottom.

## The R semantics reproduced here

| R                                             | here |
|-----------------------------------------------|------|
| `SmplExtend <- 15`                            | `--extent 15` |
| `SmplGeom <- 512`                             | `--tile-px 512` |
| `st_buffer(STORM_AREA, -11)`                  | `--inward-buffer 11` (literal, not derived) |
| `smplNbr <- floor(area/(SmplExtend^2)*100)`   | `--oversample 100` |
| `runif(n=1, min=-179, max=180)`               | uniform rotation over the same range |
| `spatial.select(SpecData, footprint)`         | polygons intersecting the footprint |
| `sum(area(mask_shp)) > SmplExtend^2/200`      | `--min-stem-area` on the **whole** area of those polygons |

That last row is the subtle one. R sums the *full* area of every selected polygon, not the
part lying inside the footprint, so a tile qualifies on the strength of a stem that mostly
falls outside it. It is reproduced here because reproducing it is the point; it is also
one reason to prefer the other sampler.

## Deliberate deviations

* **Output naming.** R writes `train_{i}_{pid}_{host}{time}.jpeg`. The loader in
  `training/dataset.py` pairs on an integer, so tiles are written `train{N}.jpeg` /
  `mask{N}.gif`. The R names encode the parallel worker that produced them and carry no
  information a reader needs.
* **Rotation.** R rotates the cropped image with ImageMagick and trims twice to recover
  the axis-aligned content. Reading an oversized window, rotating, and centre-cropping is
  the same geometry without depending on trim finding the right edges.
* **The `background=75` trick.** R rasterizes non-stem interior as 75 and outside-footprint
  as 0 so `image_trim` can tell them apart, then dithers back to binary. With an explicit
  window transform that distinction is free, so the mask is binary throughout and never
  passes through a dither.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_training_tiles import _load_geoms, _random_points, _footprint  # noqa: E402


def sample_r(ortho, stems, aoi, out_dir, extent_m=15.0, tile_px=512, oversample=100.0,
             inward_buffer_m=11.0, min_stem_divisor=200.0, seed=1, limit=None,
             start_index=1, quiet=False):
    import rasterio
    from affine import Affine
    from PIL import Image
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.windows import from_bounds
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    rng = np.random.default_rng(seed)
    src = rasterio.open(ortho)
    gsd = abs(src.transform.a)

    stem_geoms, _ = _load_geoms(stems, src.crs, None, "stems", quiet)
    tree = STRtree(stem_geoms)
    aoi_geoms, _ = _load_geoms(aoi, src.crs, None, "aoi", quiet)
    area = unary_union(aoi_geoms)

    # R: STORM_AREA <- st_buffer(st_as_sf(STORM_AREA), -11)  — a literal 11 m, not
    # derived from the footprint. For a 15 m square that is just over the 10.61 m
    # half-diagonal, so it happens to be sufficient; at any other --extent it is not.
    inner = area.buffer(-inward_buffer_m)
    if inner.is_empty:
        raise SystemExit(
            f"the AOI is empty after the {inward_buffer_m:g} m inward buffer "
            f"(area {area.area:.0f} m²)")

    # R: smplNbr <- floor((area(STORM_AREA)/(SmplExtend^2)*100))
    n_attempts = int(math.floor(inner.area / (extent_m ** 2) * oversample))
    min_stem_area = extent_m ** 2 / min_stem_divisor

    img_dir = os.path.join(out_dir, "train")
    msk_dir = os.path.join(out_dir, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    half_diag = extent_m * math.sqrt(2) / 2
    win_px = int(math.ceil(2 * half_diag / gsd))
    crop_px = int(round(extent_m / gsd))

    if not quiet:
        print(f"R port: extent {extent_m:g} m, buffer {inward_buffer_m:g} m, "
              f"{n_attempts} attempts, min stem area {min_stem_area:.3f} m²")

    n = start_index
    stats = {"attempts": 0, "no_stem": 0, "too_few_stems": 0, "written": 0}
    angles = rng.uniform(-179.0, 180.0, n_attempts)      # R: runif(1, -179, 180)

    for (cx, cy), angle in zip(_random_points(inner, n_attempts, rng), angles):
        stats["attempts"] += 1
        if limit and stats["written"] >= limit:
            break
        fp = _footprint(cx, cy, extent_m, angle)

        # R: mask_shp_sf = spatial.select(SpecData, sample_footprint); the test is on
        # length(mask_shp_sf$id) > 0, i.e. any polygon intersecting the footprint
        hits = [stem_geoms[i] for i in tree.query(fp)]
        hits = [g for g in hits if g.intersects(fp)]
        if not hits:
            stats["no_stem"] += 1
            continue
        # R: sum(area(mask_shp)) — the WHOLE area of the selected polygons, not the part
        # inside the footprint. Reproduced deliberately; see the module docstring.
        if sum(g.area for g in hits) <= min_stem_area:
            stats["too_few_stems"] += 1
            continue

        win = from_bounds(cx - half_diag, cy - half_diag, cx + half_diag, cy + half_diag,
                          transform=src.transform)
        arr = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0,
                       out_shape=(3, win_px, win_px))
        wt = src.window_transform(win) * Affine.scale(win.width / win_px,
                                                      win.height / win_px)
        rgb = Image.fromarray(np.transpose(arr, (1, 2, 0)).astype("uint8"))
        mask_arr = rio_rasterize([(g, 255) for g in hits], out_shape=(win_px, win_px),
                                 transform=wt, fill=0, dtype="uint8")
        msk = Image.fromarray(mask_arr)

        if angle:
            rgb = rgb.rotate(angle, resample=Image.BICUBIC)
            msk = msk.rotate(angle, resample=Image.NEAREST)
        off = (win_px - crop_px) // 2
        rgb = rgb.crop((off, off, off + crop_px, off + crop_px)).resize(
            (tile_px, tile_px), Image.BICUBIC)
        msk = msk.crop((off, off, off + crop_px, off + crop_px)).resize(
            (tile_px, tile_px), Image.NEAREST)
        msk = Image.fromarray(((np.asarray(msk) > 127) * 255).astype("uint8"))

        rgb.save(os.path.join(img_dir, f"train{n}.jpeg"), quality=95)
        msk.save(os.path.join(msk_dir, f"mask{n}.gif"))
        stats["written"] += 1
        n += 1

    src.close()
    stats["next_index"] = n
    if not quiet:
        print(f"wrote {stats['written']} tiles to {out_dir}")
        print(f"rejected: {stats['no_stem']} with no stem, "
              f"{stats['too_few_stems']} below {min_stem_area:.3f} m²")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ortho", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--aoi", required=True, help="the R script's `windthrow_area`")
    p.add_argument("--out", required=True)
    p.add_argument("--extent", type=float, default=15.0, help="R: SmplExtend")
    p.add_argument("--tile-px", type=int, default=512, help="R: SmplGeom")
    p.add_argument("--oversample", type=float, default=100.0, help="R: the 100 in smplNbr")
    p.add_argument("--inward-buffer", type=float, default=11.0, help="R: st_buffer(-11)")
    p.add_argument("--min-stem-divisor", type=float, default=200.0,
                   help="R: SmplExtend^2/200")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stats-json", default=None)
    a = p.parse_args(argv)
    stats = sample_r(a.ortho, a.stems, a.aoi, a.out, a.extent, a.tile_px, a.oversample,
                     a.inward_buffer, a.min_stem_divisor, a.seed, a.limit, a.start_index)
    if a.stats_json:
        with open(a.stats_json, "w") as f:
            json.dump(stats, f, indent=1)
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    sys.exit(main())

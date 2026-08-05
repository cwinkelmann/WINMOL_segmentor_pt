"""Extract georeferenced (stem mask -> ortho tile) pairs for ControlNet pretraining.

    python scripts/build_ortho_pairs.py --ortho <ortho.tif> --stem-map <stems.tif> \
        --out <DIR> [--dem <dem.tif>] [--tile 512] [--stride 512]

The orthomosaics are the largest untapped source of appearance data: Barnekow
alone is ~9600x8700 px. The analyzer's stem maps supply masks for them. Those
masks are *predictions*, not ground truth, which rules them out for training a
segmenter — but ControlNet only needs plausible geometry paired with real
appearance, so pseudo-labels are acceptable here in a way they would not be
downstream.

Ortho and stem map generally sit on different grids (different size, origin and
sometimes resolution), so tiles are cut by **world coordinates** via each
raster's affine transform, never by pixel index. Cutting by pixel index silently
pairs a tile with the wrong piece of mask — the failure mode this script exists
to avoid.

With --dem the same world window is also cut from a DEM/DSM/CHM, giving
depth/depth{N}.png for --rgbd training. Because every raster is read by world
coordinates, the DEM may differ from the ortho in size, origin and resolution —
which it usually does — and the tiles still line up. Heights are written as
uint16 over a fixed range (--depth-vmin/--depth-vmax, metres) so a grey level
means the same height in every tile of the dataset; per-tile scaling would
discard exactly the absolute height that makes real depth useful.

Intended use: build a loader-ready RGB or RGBD dataset from one site. For a
leak-free multi-site experiment drive this through build_site_splits.py, which
keeps whole orthomosaics on one side of the split.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def _open(path):
    try:
        import rasterio
    except ImportError:  # pragma: no cover - environment-dependent
        print("error: this script needs rasterio (pip install rasterio)", file=sys.stderr)
        raise SystemExit(2)
    return rasterio.open(path)


def _read_window_world(src, left, top, right, bottom, size, resample_nearest):
    """Read the world-coordinate box from `src`, resampled to size x size."""
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    win = from_bounds(left, bottom, right, top, transform=src.transform)
    if win.width < 1 or win.height < 1:
        return None
    data = src.read(
        indexes=list(range(1, min(src.count, 3) + 1)),
        window=win, out_shape=(min(src.count, 3), size, size), boundless=True,
        fill_value=0,
        resampling=Resampling.nearest if resample_nearest else Resampling.bilinear,
    )
    return np.transpose(data, (1, 2, 0))


def extract(ortho_path, stem_path, out_dir, tile, stride, min_frac, max_frac, size, limit,
            dem_path=None, depth_vmin=None, depth_vmax=None, dem_nodata=None):
    ortho, stems = _open(ortho_path), _open(stem_path)
    dem = _open(dem_path) if dem_path else None
    if dem is not None and dem.crs != ortho.crs:
        print(f"error: DEM CRS {dem.crs} != ortho CRS {ortho.crs}; reproject first",
              file=sys.stderr)
        return None
    if ortho.crs != stems.crs:
        print(f"error: CRS mismatch — ortho {ortho.crs} vs stem map {stems.crs}; "
              "reproject one of them first", file=sys.stderr)
        return None

    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
    if dem is not None:
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

    # walk the intersection of both footprints in world coordinates
    left = max(ortho.bounds.left, stems.bounds.left)
    right = min(ortho.bounds.right, stems.bounds.right)
    bottom = max(ortho.bounds.bottom, stems.bounds.bottom)
    top = min(ortho.bounds.top, stems.bounds.top)
    if left >= right or bottom >= top:
        print("error: ortho and stem map do not overlap", file=sys.stderr)
        return None

    px = abs(ortho.transform.a)                      # ortho ground resolution
    step_m, tile_m = stride * px, tile * px
    written, skipped_empty, skipped_nodata = 0, 0, 0

    y = top
    while y - tile_m > bottom and (limit is None or written < limit):
        x = left
        while x + tile_m < right and (limit is None or written < limit):
            img = _read_window_world(ortho, x, y, x + tile_m, y - tile_m, size, False)
            msk = _read_window_world(stems, x, y, x + tile_m, y - tile_m, size, True)
            x += step_m
            if img is None or msk is None:
                continue
            rgb = img[..., :3].astype(np.uint8)
            if rgb.mean() < 20 or rgb.std() < 10:    # nodata / featureless
                skipped_nodata += 1
                continue
            m = (msk[..., 0] > 0)
            frac = float(m.mean())
            if not min_frac <= frac <= max_frac:
                skipped_empty += 1
                continue
            n = written + 1
            if dem is not None:
                z = _read_window_world(dem, x, y, x + tile_m, y - tile_m, size, False)
                if z is None:
                    skipped_nodata += 1
                    continue
                z = z[..., 0].astype(np.float32)
                valid = np.isfinite(z)
                if dem_nodata is not None:
                    valid &= z != dem_nodata
                if valid.mean() < 0.5:          # mostly holes: not worth training on
                    skipped_nodata += 1
                    continue
                lo = float(np.nanmin(z[valid])) if depth_vmin is None else float(depth_vmin)
                hi = float(np.nanmax(z[valid])) if depth_vmax is None else float(depth_vmax)
                zz = np.zeros_like(z)
                if hi > lo:
                    zz[valid] = np.clip((z[valid] - lo) / (hi - lo), 0.0, 1.0)
                Image.fromarray((zz * 65535).astype(np.uint16), mode="I;16").save(
                    os.path.join(out_dir, "depth", f"depth{n}.png"))
            Image.fromarray(rgb).save(
                os.path.join(out_dir, "train", f"train{n}.jpeg"), quality=95)
            Image.fromarray((m * 255).astype(np.uint8), mode="L").convert("P").save(
                os.path.join(out_dir, "mask", f"mask{n}.gif"))
            written += 1
        y -= step_m

    print(f"wrote {written} pairs to {out_dir} "
          f"(skipped {skipped_empty} out-of-range stem fraction, {skipped_nodata} nodata)")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ortho", required=True)
    p.add_argument("--stem-map", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=512, help="tile size in ortho pixels")
    p.add_argument("--stride", type=int, default=512, help="step in ortho pixels")
    p.add_argument("--size", type=int, default=512, help="output tile size")
    p.add_argument("--min-stem-fraction", type=float, default=0.005)
    p.add_argument("--max-stem-fraction", type=float, default=0.6)
    p.add_argument("--dem", default=None,
                   help="DEM/DSM/CHM raster; cut by world coords into depth/depth{N}.png")
    p.add_argument("--depth-vmin", type=float, default=None,
                   help="fixed height range low end in metres — set this so a grey level "
                        "means the same height across the whole dataset")
    p.add_argument("--depth-vmax", type=float, default=None)
    p.add_argument("--dem-nodata", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    ok = extract(a.ortho, a.stem_map, a.out, a.tile, a.stride,
                 a.min_stem_fraction, a.max_stem_fraction, a.size, a.limit,
                 dem_path=a.dem, depth_vmin=a.depth_vmin, depth_vmax=a.depth_vmax,
                 dem_nodata=a.dem_nodata)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

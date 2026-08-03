"""Extract georeferenced (stem mask -> ortho tile) pairs for ControlNet pretraining.

    python scripts/build_ortho_pairs.py --ortho <ortho.tif> --stem-map <stems.tif> \
        --out <DIR> [--tile 512] [--stride 512] [--min-stem-fraction 0.005]

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

Intended use: pretrain ControlNet on these pairs for broad appearance, then
fine-tune on hand-labelled pairs (build_controlnet_dataset.py) for mask
adherence.
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


def extract(ortho_path, stem_path, out_dir, tile, stride, min_frac, max_frac, size, limit):
    ortho, stems = _open(ortho_path), _open(stem_path)
    if ortho.crs != stems.crs:
        print(f"error: CRS mismatch — ortho {ortho.crs} vs stem map {stems.crs}; "
              "reproject one of them first", file=sys.stderr)
        return None

    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)

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
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    ok = extract(a.ortho, a.stem_map, a.out, a.tile, a.stride,
                 a.min_stem_fraction, a.max_stem_fraction, a.size, a.limit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Trace a training tile back to the ground it came from.

    python scripts/locate_tile.py --data-dir <DS>/test --tile 182
    python scripts/locate_tile.py --data-dir <DS>/test --tile 182 --crop audit.png

A tile is a rotated crop of one orthomosaic, resampled to 512 px. When a label looks
wrong — a stem marked where none is visible, or a visible stem left unmarked — the
question is always "where is this, and what does the source imagery look like there?".
Without the manifest that means guessing the site from the tile index.

`--crop` re-cuts the same footprint from the original orthomosaic at its **native**
resolution and writes it beside the training tile, which is the view that settles most
label questions: the training tile for Campus is upsampled 3.2x from 6.4 cm imagery and
simply has no detail to judge by.
"""
import argparse
import json
import os
import sys


def load_manifest(data_dir):
    path = os.path.join(data_dir, "tiles.jsonl")
    if not os.path.exists(path):
        raise SystemExit(
            f"no tiles.jsonl in {data_dir}. It is written by sample_training_tiles.py; "
            f"datasets built before provenance was added do not have one and would need "
            f"regenerating (same --seed and --split-seed reproduce the tiles exactly).")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find(data_dir, n):
    for rec in load_manifest(data_dir):
        if rec["n"] == n:
            return rec
    raise SystemExit(f"tile {n} is not in {data_dir}/tiles.jsonl")


def native_crop(rec, out_path, pad_m=2.0):
    """Re-cut the tile's footprint from the source ortho at native resolution."""
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.windows import from_bounds

    if not os.path.exists(rec["ortho"]):
        raise SystemExit(f"source orthomosaic not found: {rec['ortho']}")
    half = rec["extent_m"] / 2 + pad_m
    with rasterio.open(rec["ortho"]) as src:
        win = from_bounds(rec["centre_x"] - half, rec["centre_y"] - half,
                          rec["centre_x"] + half, rec["centre_y"] + half,
                          transform=src.transform)
        arr = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0)
        gsd = abs(src.transform.a)
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)).astype("uint8"))
    img.save(out_path)
    return {"path": out_path, "size": img.size, "native_gsd_cm": round(gsd * 100, 2)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="tile dir containing tiles.jsonl")
    p.add_argument("--tile", type=int, required=True)
    p.add_argument("--crop", default=None,
                   help="write the same footprint re-cut from the ortho at native "
                        "resolution, which is what the training tile was resampled from")
    a = p.parse_args(argv)

    rec = find(a.data_dir, a.tile)
    print(f"tile {rec['n']}  —  site {rec['site']}  ({rec.get('split') or 'no split'}"
          + (", amodal" if rec.get("amodal") else "") + ")")
    print(f"  centre   {rec['centre_x']:.2f}, {rec['centre_y']:.2f}  [{rec['crs']}]")
    print(f"  footprint {rec['extent_m']:g} m at {rec['angle_deg']:+.1f}°, "
          f"{rec['tile_px']} px -> {100 * rec['gsd_m_per_px']:.2f} cm/px")
    print(f"  stem      {100 * rec['stem_frac']:.2f}% of the tile")
    print(f"  ortho     {rec['ortho']}")
    if a.crop:
        info = native_crop(rec, a.crop)
        print(f"  native crop -> {info['path']} {info['size']} at {info['native_gsd_cm']} cm/px"
              f"  (the training tile is resampled "
              f"{rec['gsd_m_per_px'] * 100 / info['native_gsd_cm']:.2f}x from this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

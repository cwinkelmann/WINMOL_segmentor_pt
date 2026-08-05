"""Build leak-free train/val/test datasets from whole orthomosaics.

    python scripts/build_site_splits.py --config sites.json --out <ROOT> [--rgbd]

Splitting *tiles* randomly leaks: neighbouring tiles from one flight share
illumination, stand structure, sensor and season, so a model can score well by
recognising the site rather than the stems. This assigns whole **orthomosaics**
to a split, so nothing from a training site can appear in val or test.

Config is a JSON list, one entry per site:

    [
      {"name": "barnekow",    "split": "train",
       "ortho": "/path/20220212_Barnekow_4.tiff",
       "stem_map": "/path/barnekow_stems.tif",
       "dem": "/path/barnekow_dsm.tif"},
      {"name": "bremerhagen", "split": "val",   "ortho": ..., "stem_map": ...},
      {"name": "kraking",     "split": "test",  "ortho": ..., "stem_map": ...}
    ]

`dem` is optional per site, but with --rgbd every site must have one, or the
split it belongs to would silently train on fewer tiles than expected.

Writes <ROOT>/{train,val,test}/ in loader format, plus splits.json recording
which site produced which tiles — so the provenance of any tile is recoverable
and the leak-free claim is auditable rather than asserted.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ortho_pairs import extract            # noqa: E402

SPLITS = ("train", "val", "test")


def _renumber_into(src, dst, start, rgbd):
    """Append src's tiles to dst, renumbering from `start`. Returns how many moved."""
    subs = ["train", "mask"] + (["depth"] if rgbd else [])
    for s in subs:
        os.makedirs(os.path.join(dst, s), exist_ok=True)
    ids = sorted(int(f[len("train"):-len(".jpeg")])
                 for f in os.listdir(os.path.join(src, "train"))
                 if f.startswith("train") and f.endswith(".jpeg"))
    moved = 0
    for i, n in enumerate(ids):
        t = start + i
        shutil.copy2(os.path.join(src, "train", f"train{n}.jpeg"),
                     os.path.join(dst, "train", f"train{t}.jpeg"))
        shutil.copy2(os.path.join(src, "mask", f"mask{n}.gif"),
                     os.path.join(dst, "mask", f"mask{t}.gif"))
        if rgbd:
            dp = os.path.join(src, "depth", f"depth{n}.png")
            if not os.path.exists(dp):
                raise FileNotFoundError(
                    f"--rgbd requested but {dp} is missing; the split would silently "
                    f"train on fewer tiles than the images suggest")
            shutil.copy2(dp, os.path.join(dst, "depth", f"depth{t}.png"))
        moved += 1
    return moved


def build(config_path, out_root, rgbd, tile, stride, size, min_frac, max_frac,
          depth_vmin, depth_vmax, dem_nodata, limit_per_site):
    with open(config_path) as f:
        sites = json.load(f)

    for s in sites:
        if s.get("split") not in SPLITS:
            raise SystemExit(f"site {s.get('name')!r}: split must be one of {SPLITS}")
        if rgbd and not s.get("dem"):
            raise SystemExit(
                f"site {s['name']!r}: --rgbd requires a 'dem' for every site, else the "
                f"{s['split']} split silently shrinks")
        for key in ("ortho", "stem_map"):
            if not os.path.exists(s[key]):
                raise SystemExit(f"site {s['name']!r}: {key} not found: {s[key]}")

    by_split = {k: [] for k in SPLITS}
    for s in sites:
        by_split[s["split"]].append(s["name"])
    overlap = set(by_split["train"]) & (set(by_split["val"]) | set(by_split["test"]))
    if overlap:
        raise SystemExit(f"site(s) {sorted(overlap)} appear in train and in val/test")
    for k in SPLITS:
        if not by_split[k]:
            print(f"warning: split {k!r} has no sites", file=sys.stderr)

    manifest = {"splits": by_split, "sites": [], "rgbd": bool(rgbd)}
    counters = {k: 1 for k in SPLITS}
    staging = os.path.join(out_root, "_staging")

    for s in sites:
        site_dir = os.path.join(staging, s["name"])
        shutil.rmtree(site_dir, ignore_errors=True)
        print(f"[{s['split']}] {s['name']}: tiling…", flush=True)
        n = extract(s["ortho"], s["stem_map"], site_dir, tile, stride,
                    min_frac, max_frac, size, limit_per_site,
                    dem_path=s.get("dem"), depth_vmin=depth_vmin,
                    depth_vmax=depth_vmax, dem_nodata=dem_nodata)
        if not n:
            print(f"  no usable tiles from {s['name']}", file=sys.stderr)
            manifest["sites"].append({**{k: s[k] for k in ("name", "split")}, "tiles": 0})
            continue
        dst = os.path.join(out_root, s["split"])
        first = counters[s["split"]]
        moved = _renumber_into(site_dir, dst, first, rgbd)
        counters[s["split"]] += moved
        manifest["sites"].append({"name": s["name"], "split": s["split"],
                                  "ortho": s["ortho"], "dem": s.get("dem"),
                                  "tiles": moved, "ids": [first, first + moved - 1]})
        print(f"  {moved} tiles -> {s['split']}/ (ids {first}..{first + moved - 1})")

    shutil.rmtree(staging, ignore_errors=True)
    with open(os.path.join(out_root, "splits.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("\nsummary")
    for k in SPLITS:
        n = counters[k] - 1
        print(f"  {k:5s} {n:6d} tiles from {len(by_split[k])} site(s): "
              f"{', '.join(by_split[k]) or '—'}")
    print(f"\nmanifest: {os.path.join(out_root, 'splits.json')}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="JSON list of sites (see module docstring)")
    p.add_argument("--out", required=True, help="root dir; train/ val/ test/ are created")
    p.add_argument("--rgbd", action="store_true", help="require a DEM per site, emit depth/")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--min-stem-fraction", type=float, default=0.005)
    p.add_argument("--max-stem-fraction", type=float, default=0.6)
    p.add_argument("--depth-vmin", type=float, default=None,
                   help="fixed height range (metres) shared by every tile — strongly "
                        "recommended, otherwise each tile is scaled alone")
    p.add_argument("--depth-vmax", type=float, default=None)
    p.add_argument("--dem-nodata", type=float, default=None)
    p.add_argument("--limit-per-site", type=int, default=None)
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    build(a.config, a.out, a.rgbd, a.tile, a.stride, a.size, a.min_stem_fraction,
          a.max_stem_fraction, a.depth_vmin, a.depth_vmax, a.dem_nodata, a.limit_per_site)
    return 0


if __name__ == "__main__":
    sys.exit(main())

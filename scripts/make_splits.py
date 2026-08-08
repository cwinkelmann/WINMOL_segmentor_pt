"""Build leak-free train/val/test splits, by halving a site or by holding sites out.

    python scripts/make_splits.py --config sites.json --out /path/to/DS --strategy halve
    python scripts/make_splits.py --config sites.json --out /path/to/DS --strategy sites

`sites.json` lists what to sample:

    {"sites": [{"name": "Kaufland", "ortho": "...tif", "stems": "...shp", "aoi": "..._AOE.shp",
                "split": "train"}, ...],
     "extent_m": 10.24, "caps": {"train": 2000, "val": 400, "test": 400}}

`split` is only read by `--strategy sites`; `halve` derives its own.

## halve — one site, cut in two

Each AOI is divided by a straight line and the halves go to different splits, with a
buffer of the footprint's half-diagonal held out on *both* sides of the cut so no tile can
contain pixels from the other half.

The cut runs perpendicular to the AOI's long axis by default, which is the direction that
splits the area most evenly. `--cut-axis ns` or `ew` forces a compass direction.

This is what to use when the corpus is small. The beech sites number three, so holding one
out removes an entire acquisition — its phenology, colour cast and ground resolution — and
the resulting score measures domain transfer rather than segmentation. A halved site keeps
both sides in the same conditions.

Its weakness is the mirror image: train and test share a site, so the score says nothing
about transfer to a *new* site. It is a measure of how well the model reads this forest,
not any forest.

## sites — whole orthomosaics per split

The stronger claim, and the honest one when enough sites exist: train on two orthomosaics,
test on a third the model has never seen. Nothing is shared, so nothing can leak.

It needs sites to spare. With three, one per split leaves no redundancy and the test score
becomes a measure of one acquisition. Prefer it from about six sites upward.

Both strategies write `splits.json` recording exactly which ground went where, and both
refuse loudly rather than leaving a split empty.

## Pipeline order

This runs **fix -> sample -> split** in that order, per site, because it has to: 50 of the
corpus's polygons have self-intersecting rings and GEOS aborts on the first set operation
that touches one, killing a sampling run partway through. Step 1 writes cleaned shapefiles
and their repair reports under `<out>/_clean/`, and every later step reads those rather
than the originals. `--skip-fix` is available when the input is already repaired.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_geometries import fix as fix_geometries  # noqa: E402
from sample_training_tiles import _load_geoms, sample_tiles  # noqa: E402


def _long_axis(geom):
    """Direction of the AOI's greatest extent, from its convex hull vertices."""
    import numpy as np

    pts = np.asarray(geom.convex_hull.exterior.coords, dtype=float)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    return centre, vt[0]


def halve_aoi(geom, buffer_m, cut_axis="auto"):
    """Split an AOI in two along a line, holding `buffer_m` out on each side of the cut.

    Returns (side_a, side_b). The gap is what makes the split leak-free: a tile centred
    anywhere in side_a cannot reach across it into side_b at any rotation.
    """
    import numpy as np
    from shapely.geometry import Polygon

    centre, direction = _long_axis(geom)
    if cut_axis == "ns":
        direction = np.array([0.0, 1.0])
    elif cut_axis == "ew":
        direction = np.array([1.0, 0.0])

    # project the AOI onto its long axis and cut at the midpoint of that projection,
    # which halves the extent rather than the bounding box
    pts = np.asarray(geom.convex_hull.exterior.coords, dtype=float)
    proj = (pts - centre) @ direction
    mid = float((proj.min() + proj.max()) / 2)
    normal = np.array([-direction[1], direction[0]])

    span = float(np.abs((pts - centre) @ normal).max()) + geom.length
    out = []
    for lo, hi in ((proj.min() - 1.0, mid - buffer_m), (mid + buffer_m, proj.max() + 1.0)):
        corners = [centre + direction * lo + normal * span,
                   centre + direction * hi + normal * span,
                   centre + direction * hi - normal * span,
                   centre + direction * lo - normal * span]
        out.append(geom.intersection(Polygon(corners)))
    return out[0], out[1]


def _write_aoi(geom, path, crs):
    import fiona
    from shapely.geometry import mapping

    schema = {"geometry": "Polygon", "properties": {"part": "str"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as dst:
        for g in getattr(geom, "geoms", [geom]):
            dst.write({"geometry": mapping(g), "properties": {"part": "half"}})


def _clean_stems(site, out_dir, quiet):
    """Step 1 of the pipeline: repair before anything reads the geometry.

    50 of the corpus's polygons have self-intersecting rings, and GEOS aborts on the
    first set operation that touches one -- a sampling run dies tens of thousands of
    tiles in. Doing the repair here, rather than trusting the caller to have run
    fix_geometries.py first, makes the pipeline order structural: fix, then sample,
    then split.
    """
    work = os.path.join(out_dir, "_clean")
    os.makedirs(work, exist_ok=True)
    cleaned = os.path.join(work, f"{site['name']}.shp")
    report = fix_geometries(site["stems"], cleaned,
                            os.path.join(work, f"{site['name']}_geom.json"), quiet=quiet)
    return cleaned, {"polygons": report["polygons"], "repaired": report["repaired"],
                     "dropped": report["dropped"],
                     "area_delta_m2": report["area_delta_m2"]}


def run(config_path, out_dir, strategy, cut_axis="auto", seed=1, skip_fix=False,
        quiet=False):
    import rasterio

    cfg = json.load(open(config_path))
    extent = cfg.get("extent_m", 10.24)
    caps = cfg.get("caps", {"train": 2000, "val": 400, "test": 400})
    buffer_m = extent * math.sqrt(2) / 2
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"strategy": strategy, "extent_m": extent, "buffer_m": round(buffer_m, 3),
                "sites": []}
    counters = {s: 1 for s in ("train", "val", "test")}

    for site in cfg["sites"]:
        name = site["name"]
        # STEP 1 — fix. Everything downstream reads the cleaned geometry.
        if skip_fix:
            stems, geom_report = site["stems"], None
        else:
            stems, geom_report = _clean_stems(site, out_dir, quiet)

        if strategy == "sites":
            split = site.get("split")
            if split not in counters:
                raise SystemExit(f"site {name!r} needs a \"split\" of train/val/test "
                                 f"for --strategy sites (got {split!r})")
            # STEP 2 — sample, STEP 3 — the split is the destination directory
            stats = sample_tiles(site["ortho"], stems, site["aoi"],
                                 os.path.join(out_dir, split), extent_m=extent,
                                 limit=caps.get(split), start_index=counters[split],
                                 seed=seed, quiet=quiet)
            counters[split] = stats["next_index"]
            manifest["sites"].append({"name": name, "split": split,
                                      "tiles": stats["written"], "geometry": geom_report})
            continue

        # halve: derive two AOIs from this site and give them to different splits
        with rasterio.open(site["ortho"]) as src:
            crs = src.crs
        aoi_geoms, _ = _load_geoms(site["aoi"], crs, None, "aoi", quiet=True)
        from shapely.ops import unary_union
        whole = unary_union(aoi_geoms)
        a, b = halve_aoi(whole, buffer_m, cut_axis)
        if a.is_empty or b.is_empty:
            raise SystemExit(
                f"site {name!r}: halving leaves one side empty. Its AOI is "
                f"{whole.area:.0f} m² and the cut holds {buffer_m:.2f} m out on each side; "
                f"a smaller --extent or a site with more ground is needed.")

        parts = site.get("halves", ["train", "test"])
        entry = {"name": name, "split": "halved", "halves": {},
                 "geometry": geom_report}
        for geom, split in zip((a, b), parts):
            tmp = os.path.join(out_dir, f"_aoi_{name}_{split}.shp")
            _write_aoi(geom, tmp, crs)
            stats = sample_tiles(site["ortho"], stems, tmp,
                                 os.path.join(out_dir, split), extent_m=extent,
                                 limit=caps.get(split), start_index=counters[split],
                                 seed=seed, quiet=quiet)
            counters[split] = stats["next_index"]
            entry["halves"][split] = {"aoi_m2": round(geom.area, 1),
                                      "tiles": stats["written"]}
        manifest["sites"].append(entry)

    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if not quiet:
        print(f"\n{strategy} split -> {out_dir}")
        for s in ("train", "val", "test"):
            d = os.path.join(out_dir, s, "train")
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {s:5s}: {n} tiles")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", choices=("halve", "sites"), required=True)
    p.add_argument("--cut-axis", choices=("auto", "ns", "ew"), default="auto",
                   help="halve only; 'auto' cuts across the AOI's long axis")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-fix", action="store_true",
                   help="the stems are already repaired; skip step 1")
    a = p.parse_args(argv)
    run(a.config, a.out, a.strategy, a.cut_axis, a.seed, a.skip_fix)
    return 0


if __name__ == "__main__":
    sys.exit(main())

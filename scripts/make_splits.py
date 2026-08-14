"""Build leak-free train/val/test splits: by block, by halving a site, or by holding sites out.

    python scripts/make_splits.py --config sites.json --out /path/to/DS --strategy halve
    python scripts/make_splits.py --config sites.json --out /path/to/DS --strategy sites

`sites.json` lists what to sample:

    {"sites": [{"name": "Kaufland", "ortho": "...tif", "stems": "...shp", "aoi": "..._AOE.shp",
                "split": "train"}, ...],
     "extent_m": 10.24, "caps": {"train": 2000, "val": 400, "test": 400}}

`split` is only read by `--strategy sites`; `halve` and `blocks` derive their own.

## blocks — every site in every split

The AOI of each site is cut into a `block_size_m` grid, whole blocks are dealt to
train/val/test, and each block is then shrunk by the footprint half-diagonal, so two tiles
on opposite sides of a shared edge cannot overlap. All sites feed all three splits.

This is what to use when the deliverable is a *model* rather than a transfer measurement.
Holding a whole site out answers "does this generalise to a new forest"; pooling blocks
answers "how well can we read the forests we have", and keeps every acquisition's
phenology and resolution in training — which matters when one site returns F1 0.0000 as a
holdout, as Bachsee_north does.

Its test score is therefore optimistic about a genuinely new site. Report it alongside a
`sites` run, never instead of one.

Size the grid so at least ~12 blocks fit, or a split will be dealt none:
`block_size_m ~ sqrt(aoi_m2 / 12)`.

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
    native_px = cfg.get("native_px")
    caps = cfg.get("caps", {"train": 2000, "val": 400, "test": 400})
    # 'auto' keeps PIL bicubic, which filters hard on downsampling; 'on'/'off' switch to
    # skimage order=3 with anti_aliasing set accordingly. Measured at a x1.40 downsample,
    # on-vs-off differ by 0.006 grey levels (skimage's sigma is scale-derived and near
    # zero there) while auto-vs-skimage differ by 3.16 -- so this knob really selects the
    # resampler, not the anti-aliasing.
    antialias = cfg.get("antialias", "auto")
    # Tiles larger than the model input exist so the loader can crop at varying scale
    # (RandomSizedCrop -> img_size). Effective GSD is then native_gsd * crop_px / img_size,
    # which is what makes a model robust to the analyzer's user-set tile_size.
    tile_px = cfg.get("tile_px", 512)
    # In native mode the footprint is a pixel count, so its size in metres — and therefore
    # the buffer that keeps splits apart — differs per site with the ortho's resolution.
    buffer_m = None if native_px else extent * math.sqrt(2) / 2
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"strategy": strategy, "antialias": cfg.get("antialias", "auto"),
                "tile_px": cfg.get("tile_px", 512),
                "extent_m": None if native_px else extent,
                "native_px": native_px,
                "buffer_m": None if native_px else round(buffer_m, 3),
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
                                 seed=seed, quiet=quiet, native_px=native_px,
                                 antialias=antialias, tile_px=tile_px)
            counters[split] = stats["next_index"]
            manifest["sites"].append({"name": name, "split": split,
                                      "tiles": stats["written"], "geometry": geom_report})
            continue

        if strategy == "blocks" and site.get("whole_split"):
            # A site too small to be cut up goes entirely into one split. Kaufland's AOI
            # is 5,393 m² and a 15 m footprint holds 10.61 m out from every edge, leaving
            # 2,190 m² — no grid leaves more than one usable block. Putting it wholly in
            # train keeps the corpus's finest orthomosaic (2.09 cm/px) in the model rather
            # than dropping the resolution extreme; it contributes nothing to val/test,
            # which is the honest trade and is recorded in splits.json.
            split = site["whole_split"]
            if split not in counters:
                raise SystemExit(f"site {name!r}: whole_split must be train/val/test, "
                                 f"got {split!r}")
            n_sites = len(cfg["sites"])
            cap = caps.get(split)
            stats = sample_tiles(site["ortho"], stems, site["aoi"],
                                 os.path.join(out_dir, split), extent_m=extent,
                                 limit=math.ceil(cap / n_sites) if cap else None,
                                 start_index=counters[split], seed=seed, quiet=quiet,
                                 native_px=native_px, antialias=antialias, tile_px=tile_px)
            counters[split] = stats["next_index"]
            manifest["sites"].append({"name": name, "split": f"whole -> {split}",
                                      "tiles": {split: stats["written"]},
                                      "geometry": geom_report})
            continue

        if strategy == "blocks":
            # Every site contributes to every split. The AOI is cut into a grid, whole
            # blocks are assigned, and each block is then shrunk by the footprint's
            # half-diagonal — so two tiles either side of a shared block edge are at
            # least `extent * sqrt(2)` apart, the distance below which rotated footprints
            # can share area. Leak-free without removing an acquisition from training.
            block = site.get("block_size_m", cfg.get("block_size_m"))
            if not block:
                raise SystemExit(
                    f"site {name!r}: --strategy blocks needs \"block_size_m\", per site or "
                    f"at the top level. Aim for 12+ blocks across the AOI, or a split ends "
                    f"up with none: block_size ~ sqrt(aoi_m2 / 12).")
            fracs = cfg.get("split_fractions", {"train": 0.7, "val": 0.15, "test": 0.15})
            entry = {"name": name, "split": "blocks", "block_size_m": block,
                     "geometry": geom_report, "tiles": {}}
            # `caps` is the TOTAL per split, shared equally between sites. Applying it
            # per site instead lets the largest AOI exhaust the budget alone: Campus is
            # 118,808 m² against Kaufland's 5,393, so an unshared cap of 800 val tiles
            # was filled entirely by Campus and the other three sites reached validation
            # with nothing. Equal shares over-represent the small sites relative to their
            # ground, which is the intended trade — site identity has dominated every
            # comparison measured here, so balance beats proportionality.
            n_sites = len(cfg["sites"])
            share = {s: (math.ceil(c / n_sites) if c else None) for s, c in caps.items()}
            # `block_splits` narrows which splits this site is dealt blocks for. A
            # leave-one-site-out fold needs its training sites to yield train and val
            # only — the test set is the held-out site, and dealing test blocks here
            # would mix a second site into the yardstick. Splits left out get no cells,
            # so their fraction must be 0 or the blocks simply go unused.
            wanted = site.get("block_splits", cfg.get("block_splits",
                                                      ("train", "val", "test")))
            unknown = [s for s in wanted if s not in ("train", "val", "test")]
            if unknown:
                raise SystemExit(f"site {name!r}: block_splits has {unknown}, "
                                 f"expected any of train/val/test")
            for split in wanted:
                stats = sample_tiles(site["ortho"], stems, site["aoi"],
                                     os.path.join(out_dir, split), extent_m=extent,
                                     limit=share.get(split), start_index=counters[split],
                                     seed=seed, quiet=quiet, native_px=native_px, antialias=antialias,
                                     tile_px=tile_px, block_size_m=block, split=split,
                                     split_fractions=fracs,
                                     split_seed=cfg.get("split_seed", 1))
                counters[split] = stats["next_index"]
                entry["tiles"][split] = stats["written"]
            manifest["sites"].append(entry)
            continue

        # halve: derive two AOIs from this site and give them to different splits
        with rasterio.open(site["ortho"]) as src:
            crs = src.crs
            gsd = abs(src.transform.a)
        site_extent = native_px * gsd if native_px else extent
        site_buffer = site_extent * math.sqrt(2) / 2
        aoi_geoms, _ = _load_geoms(site["aoi"], crs, None, "aoi", quiet=True)
        from shapely.ops import unary_union
        whole = unary_union(aoi_geoms)
        a, b = halve_aoi(whole, site_buffer, cut_axis)
        if a.is_empty or b.is_empty:
            raise SystemExit(
                f"site {name!r}: halving leaves one side empty. Its AOI is "
                f"{whole.area:.0f} m² and the cut holds {site_buffer:.2f} m out on each "
                f"side (footprint {site_extent:.1f} m at {100*gsd:.2f} cm/px); a smaller "
                f"footprint or a site with more ground is needed.")

        parts = site.get("halves", ["train", "test"])
        entry = {"name": name, "split": "halved", "halves": {},
                 "geometry": geom_report}
        for geom, split in zip((a, b), parts):
            tmp = os.path.join(out_dir, f"_aoi_{name}_{split}.shp")
            _write_aoi(geom, tmp, crs)
            stats = sample_tiles(site["ortho"], stems, tmp,
                                 os.path.join(out_dir, split), extent_m=extent,
                                 limit=caps.get(split), start_index=counters[split],
                                 seed=seed, quiet=quiet, native_px=native_px,
                                 antialias=antialias, tile_px=tile_px)
            counters[split] = stats["next_index"]
            entry["halves"][split] = {"aoi_m2": round(geom.area, 1),
                                      "tiles": stats["written"],
                                      "extent_m": round(site_extent, 2),
                                      "gsd_cm": round(100 * gsd, 2)}
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
    p.add_argument("--strategy", choices=("halve", "sites", "blocks"), required=True)
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

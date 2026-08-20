"""`prepare.py` — orthomosaic in, training tiles out.

One command over four modes, because they are stages of the same job and sharing a CLI
keeps their vocabulary (extent, native-px, split, AOI) identical:

    # one site
    prepare.py --ortho site.tif --stems site.shp --aoi site_AOE.shp --out DS

    # several sites, leak-free splits (this is the usual one)
    prepare.py --config sites.json --out DS --strategy blocks [--jobs 4]

    # leave-one-site-out folds from per-site tile sets, by symlink
    prepare.py --compose-folds --site-all ALL --site-tv TV \
               --fold-sites A B C --splittable A B --out FOLDS

    # just the label raster, no tiling
    prepare.py --rasterize --stems site.shp --ortho site.tif --out stem_map.tif

Starting from data that is already raster pairs rather than from an orthomosaic — these
need no GDAL, so they work without the [geo] extra:

    prepare.py --from-folder --src raw/ --out data/ready       # pair, renumber, binarise
    prepare.py --from-coco --coco-json a.json --images-dir img/ --out data/ready
    prepare.py --split --src data/ready --out data/split       # fixed train/val split

Two things worth knowing before choosing flags:

**`--extent-m` is a scale knob, not a speed knob.** A tile covers `extent_m` of ground and
is resized to 512 px, so the model's effective ground resolution is `extent_m / 512` —
2.93 cm/px at the 15 m default, regardless of the orthomosaic's own resolution. The
analyzer applies the same arithmetic to its `tile_size`, and matching the two was worth
+14.6 F1. Use `--native-px` instead to cut at the ortho's own resolution, which is what
the multi-scale recipes (`--recipe jitter`) train on.

**Splits must be leak-free by construction.** Sampling oversamples heavily and tiles are
cut at random rotations, so tiles overlap and a random split of the tile list leaks
between train and test. `--strategy blocks|halve|sites` partitions the *ground* first.
Every run writes `tiles.jsonl` (source ortho, world centre, rotation, GSD, stem fraction)
so the result can be audited for leakage afterwards rather than taken on trust.
"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="prepare.py", description="Build training tiles from an orthomosaic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Two things worth knowing")[0].strip())

    mode = p.add_argument_group("mode (pick one; single-site is the default)")
    mode.add_argument("--config", help="JSON listing sites; enables the multi-site path")
    mode.add_argument("--compose-folds", action="store_true",
                      help="compose leave-one-site-out folds from per-site tile sets")
    mode.add_argument("--rasterize", action="store_true",
                      help="burn annotations to a label raster and stop (no tiling)")
    mode.add_argument("--ingest", action="store_true",
                      help="normalise annotations onto the ortho's CRS and stop")
    mode.add_argument("--resample", action="store_true",
                      help="write one full-ortho COG per --gsd and stop")
    mode.add_argument("--layout", action="store_true",
                      help="write tile footprints without cutting any pixels")
    mode.add_argument("--from-folder", action="store_true",
                      help="convert an existing image/mask folder to the loader convention")
    mode.add_argument("--from-coco", action="store_true",
                      help="rasterise COCO polygon annotations into a loader dataset")
    mode.add_argument("--split", action="store_true",
                      help="materialise a fixed train/val split of a loader dataset")

    src = p.add_argument_group("single-site inputs")
    src.add_argument("--ortho", help="orthomosaic GeoTIFF")
    src.add_argument("--stems", help="digitised stem polygons (shapefile/gpkg)")
    src.add_argument("--aoi", help="windthrow polygon; sampling never leaves it")
    src.add_argument("--out", required=True, help="output dataset dir (or file for --rasterize)")

    tile = p.add_argument_group("tiling")
    tile.add_argument("--extent-m", type=float, default=None,
                      help="tile footprint in metres. Effective GSD is extent_m/512 "
                           "(default 15 m -> 2.93 cm/px). Match the analyzer's tile_size. "
                           "Mutually exclusive with --tile-px (--layout).")
    tile.add_argument("--native-px", type=int, default=None,
                      help="cut NATIVE_PX square at the ortho's own resolution instead, "
                           "for the multi-scale recipes. Overrides --extent-m.")
    tile.add_argument("--tile-px", type=int, default=None,
                      help="output tile side (default 512). Mutually exclusive with "
                           "--extent-m (--layout).")
    tile.add_argument("--limit", type=int, default=None, help="stop after N tiles")
    tile.add_argument("--min-stem-frac", type=float, default=1 / 200.0,
                      help="reject tiles below this stem fraction (default 0.5%%); a plain "
                           "grid over these sites is ~99%% background")
    tile.add_argument("--species", nargs="*", default=None, help="filter by Species attribute")
    tile.add_argument("--seed", type=int, default=1)

    sp = p.add_argument_group("splitting (multi-site)")
    sp.add_argument("--strategy", choices=("blocks", "halve", "sites"), default="blocks",
                    help="how to partition the GROUND. blocks: spatial blocks within each "
                         "site. halve: cut each site in two. sites: hold whole sites out.")
    sp.add_argument("--cut-axis", choices=("auto", "ns", "ew"), default="auto")
    sp.add_argument("--jobs", type=int, default=1,
                    help="sites to extract concurrently; >1 uses the parallel driver")
    sp.add_argument("--skip-fix", action="store_true",
                    help="skip stem-geometry repair (only if the input is already clean; "
                         "50 of the corpus's 3,842 polygons self-intersect and GEOS "
                         "raises on the first set operation that touches one)")

    fo = p.add_argument_group("--compose-folds options")
    fo.add_argument("--site-all", help="dir of <site>/ whole-site tiles")
    fo.add_argument("--site-tv", help="dir of <site>/{train,val} block tiles")
    fo.add_argument("--fold-sites", nargs="+", default=None,
                    help="one fold per named site (folds.main calls this --folds)")
    fo.add_argument("--splittable", nargs="+", default=None,
                    help="sites large enough to block-split (contribute train+val)")
    fo.add_argument("--copy", action="store_true", help="copy instead of symlink")

    cv = p.add_argument_group("--from-folder / --from-coco / --split options")
    cv.add_argument("--src", help="source dataset dir (--from-folder, --split)")
    cv.add_argument("--coco-json", help="COCO annotations JSON (--from-coco)")
    cv.add_argument("--images-dir", help="directory of source images (--from-coco)")
    cv.add_argument("--val-fraction", type=float, default=0.2,
                    help="--split: fraction held out for validation (default 0.2). Matches "
                         "the loader's own seeded split for the same fraction and seed.")

    ra = p.add_argument_group("--rasterize options")
    ra.add_argument("--instances", default=None, help="also write an instance-id raster here")
    ra.add_argument("--all-touched", action="store_true")

    pl = p.add_argument_group("staged pipeline (--ingest / --resample / --layout / --cut)")
    pl.add_argument("--stems-layer", default=None,
                    help="layer holding the stem polygons (--ingest)")
    pl.add_argument("--aoi-layer", default=None,
                    help="layer holding the AOI polygons (--ingest)")
    pl.add_argument("--aoi-ids", nargs="*", type=int, default=None,
                    help="keep only these AOIs, by position in the AOI layer")
    pl.add_argument("--gsd", nargs="*", type=float, default=None,
                    help="target ground sample distances in metres (--resample)")
    pl.add_argument("--jpeg-quality", type=int, default=95,
                    help="stage-2 JPEG quality; the source is already lossy")
    pl.add_argument("--mode", choices=("grid", "random"), default="grid",
                    help="tile placement (--layout)")
    pl.add_argument("--stride-frac", type=float, default=1.0,
                    help="grid step as a fraction of the tile extent")
    pl.add_argument("--min-valid-frac", type=float, default=0.5,
                    help="reject tiles whose footprint is less than this fraction valid")
    pl.add_argument("--n-tiles", type=int, default=None,
                    help="how many tiles to draw (--mode random)")

    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_folder:
        from winmol_unet.data.build import build_dataset
        if not args.src:
            parser.error("--from-folder needs --src")
        print(build_dataset(args.src, args.out))
        return 0

    if args.from_coco:
        from winmol_unet.data.coco import coco_to_dataset
        if not (args.coco_json and args.images_dir):
            parser.error("--from-coco needs --coco-json and --images-dir")
        print(coco_to_dataset(args.coco_json, args.images_dir, args.out,
                              limit=args.limit or 100, seed=args.seed))
        return 0

    if args.split:
        from winmol_unet.data.split import split_dataset
        if not args.src:
            parser.error("--split needs --src")
        print(split_dataset(args.src, args.out, val_fraction=args.val_fraction,
                            seed=args.seed))
        return 0

    if args.ingest:
        from winmol_unet.pipeline.ingest import ingest
        if not (args.stems and args.ortho and args.stems_layer):
            parser.error("--ingest needs --stems, --ortho and --stems-layer")
        # Upper-cased and stripped here so the case-insensitive comparison ingest()
        # already performs on the stem's own value actually matches end to end.
        species = {s.strip().upper() for s in args.species} if args.species else None
        ingest(args.stems, args.ortho, args.out, stems_layer=args.stems_layer,
               aoi_layer=args.aoi_layer, aoi_ids=args.aoi_ids,
               species=species, quiet=args.quiet)
        return 0

    if args.resample:
        from winmol_unet.pipeline.resample import resample
        if not (args.ortho and args.gsd):
            parser.error("--resample needs --ortho and --gsd")
        for g in args.gsd:
            resample(args.ortho, args.out, g, jpeg_quality=args.jpeg_quality,
                     quiet=args.quiet)
        return 0

    if args.layout:
        from winmol_unet.pipeline.layout import layout
        if not args.ortho:
            parser.error("--layout needs --ortho (a stage-2 GSD copy)")
        layout(args.ortho, args.aoi, args.stems, args.out, mode=args.mode,
               extent_m=args.extent_m, tile_px=args.tile_px,
               stride_frac=args.stride_frac, min_valid_frac=args.min_valid_frac,
               min_stem_frac=args.min_stem_frac, n_tiles=args.n_tiles,
               seed=args.seed, quiet=args.quiet)
        return 0

    if args.rasterize:
        from winmol_unet.geo.rasterize import rasterize
        if not (args.stems and args.ortho):
            parser.error("--rasterize needs --stems and --ortho")
        rasterize(args.stems, args.ortho, args.out, instances_path=args.instances,
                  species=args.species, all_touched=args.all_touched)
        print(f"wrote {args.out}")
        return 0

    if args.compose_folds:
        from winmol_unet.geo import folds
        missing = [n for n, v in (("--site-all", args.site_all), ("--site-tv", args.site_tv),
                                  ("--fold-sites", args.fold_sites),
                                  ("--splittable", args.splittable)) if not v]
        if missing:
            parser.error(f"--compose-folds needs {', '.join(missing)}")
        forwarded = ["--site-all", args.site_all, "--site-tv", args.site_tv,
                     "--out", args.out, "--folds", *args.fold_sites,
                     "--splittable", *args.splittable]
        if args.copy:
            forwarded.append("--copy")
        return folds.main(forwarded)

    if args.config:
        if args.jobs > 1:
            from winmol_unet.geo import parallel
            # Forward everything that affects WHAT is sampled, not just how fast: --jobs
            # must not change the result.
            forwarded = ["--config", args.config, "--out", args.out,
                         "--strategy", args.strategy, "--jobs", str(args.jobs),
                         "--seed", str(args.seed), "--cut-axis", args.cut_axis]
            if args.skip_fix:
                forwarded.append("--skip-fix")
            return parallel.main(forwarded)
        from winmol_unet.geo.splits import run
        run(args.config, args.out, args.strategy, cut_axis=args.cut_axis,
            seed=args.seed, skip_fix=args.skip_fix, quiet=args.quiet)
        return 0

    # single site
    if not (args.ortho and args.stems and args.aoi):
        parser.error("single-site mode needs --ortho, --stems and --aoi "
                     "(or pass --config for the multi-site path)")
    from winmol_unet.geo.sample import sample_tiles
    stats = sample_tiles(args.ortho, args.stems, args.aoi, args.out,
                         extent_m=15.0 if args.extent_m is None else args.extent_m,
                         tile_px=512 if args.tile_px is None else args.tile_px,
                         native_px=args.native_px, min_stem_frac=args.min_stem_frac,
                         seed=args.seed, species=args.species, limit=args.limit,
                         quiet=args.quiet)
    print(stats)
    return 0

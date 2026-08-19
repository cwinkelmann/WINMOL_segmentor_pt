"""`evaluate.py` — score a model, on tiles or on the ground.

Three modes, one metric implementation. Everything routes through
`winmol_unet.training.evaluate`, which is what produced every `test_results.md` in this
repo, so a number from here drops straight into those tables.

    # tile mode: anything prepare.py produced. Works on .onnx or .pt
    evaluate.py --model model.onnx --data-dir DS/test

    # geospatial mode: predict over the AOI in world space, score there
    evaluate.py --model model.onnx --ortho site.tif --aoi site_AOE.shp --stems site.shp

    # score an Analyzer stem map on the same basis
    evaluate.py --stem-map out.tif --aoi site_AOE.shp --stems site.shp

**Which mode to use.** Tile F1 measures the model *and the exam*: a tile cut at 1.2 cm/px
and one at 2.93 cm/px are different exams, because a stem is ~2.4x thicker in pixels on
the first. Two models trained at different ground resolutions therefore cannot be compared
on their own tile sets — use geospatial mode, which predicts over the same ground and
scores both on one reference grid.

**The AOI mask is not optional in geospatial mode.** Outside the windthrow polygon stems
are real but were never digitised, so scoring the whole raster counts correct detections
as false positives — and penalises the better model hardest. `--edge-buffer-m` shrinks the
AOI further, so tiles straddling its boundary do not score against absent labels.
"""
import argparse
import json
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="evaluate.py", description="Score a model on tiles or on the ground.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="contract-conformant .onnx, or a .pt checkpoint")
    p.add_argument("--data-dir", default=None,
                   help="tile mode: a dataset dir with train/ and mask/")
    p.add_argument("--ortho", default=None, help="geospatial mode: orthomosaic")
    p.add_argument("--stem-map", default=None,
                   help="score an existing stem-map raster (e.g. an Analyzer output)")
    p.add_argument("--aoi", default=None, help="windthrow polygon; required for the ground modes")
    p.add_argument("--stems", default=None, help="digitised stems, the ground truth")
    p.add_argument("--extent-m", type=float, default=15.0,
                   help="geospatial mode: tile footprint; effective GSD is extent_m/512")
    p.add_argument("--native-px", type=int, default=None)
    p.add_argument("--ref-gsd-cm", type=float, default=2.9297)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--edge-buffer-m", type=float, default=2.0,
                   help="shrink the AOI before scoring, so tiles straddling its edge do "
                        "not score against undigitised ground")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--arch", default="unet", help="tile mode with a .pt: architecture to rebuild")
    p.add_argument("--encoder", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--label", default=None, help="name for this run in the output")
    p.add_argument("--json-out", default=None, help="also write the metrics as JSON here")
    p.add_argument("--quiet", action="store_true")
    return p


def _emit(metrics, args):
    if args.label:
        metrics = {"label": args.label, **metrics}
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- score an existing raster -------------------------------------------------
    if args.stem_map:
        if not (args.aoi and args.stems):
            parser.error("--stem-map needs --aoi and --stems")
        from winmol_unet.geo.score import score_map
        return _emit(score_map(args.stem_map, args.aoi, args.stems,
                               ref_gsd=args.ref_gsd_cm / 100.0,
                               edge_buffer_m=args.edge_buffer_m,
                               threshold=args.threshold), args)

    if not args.model:
        parser.error("provide --model (tile or geospatial mode) or --stem-map")

    # --- geospatial: predict over the AOI, score on the reference grid -------------
    if args.ortho:
        if not (args.aoi and args.stems):
            parser.error("geospatial mode needs --aoi and --stems. The AOI is not "
                         "optional: outside it stems are real but undigitised, so "
                         "scoring the full raster counts correct detections as false "
                         "positives and penalises the better model hardest.")
        from winmol_unet.geo.predict import predict_plot, score
        prob, tf, W, H, area, crs, _meta = predict_plot(
            args.ortho, args.aoi, args.model,
            extent_m=None if args.native_px else args.extent_m,
            native_px=args.native_px, ref_gsd=args.ref_gsd_cm / 100.0,
            overlap=args.overlap, quiet=args.quiet)
        return _emit(score(prob, tf, W, H, area, crs, args.stems,
                           threshold=args.threshold,
                           edge_buffer_m=args.edge_buffer_m), args)

    # --- tile mode ----------------------------------------------------------------
    if not args.data_dir:
        parser.error("provide --data-dir (tile mode) or --ortho (geospatial mode)")

    if args.model.endswith(".onnx"):
        from winmol_unet.training.score_onnx import score as score_onnx
        return _emit(score_onnx(args.model, args.data_dir, batch_size=args.batch_size,
                                threshold=args.threshold), args)

    from winmol_unet.training.score_checkpoint import score as score_checkpoint
    return _emit(score_checkpoint(args.model, args.arch, args.data_dir,
                                  encoder=args.encoder, batch_size=args.batch_size,
                                  device=args.device), args)

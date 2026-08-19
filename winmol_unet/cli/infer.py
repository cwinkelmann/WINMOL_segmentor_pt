"""`infer.py` — run a model over an orthomosaic and write a georeferenced stem map.

    infer.py --model model.onnx --ortho site.tif --out pred.tif
    infer.py --model model.onnx --ortho site.tif --aoi site_AOE.shp --out pred.tif \\
             --extent-m 15 --overlap 0.5 --threshold 0.5

Writes a single-band GeoTIFF of stem probability on a common reference grid, and with
`--threshold` also a thresholded binary mask alongside it (`<out stem>_mask.tif`).

**`--extent-m` is a scale knob, not a speed knob.** A tile covers `extent_m` of ground and
is resized to the model's 512 px input, so effective ground resolution is `extent_m / 512`
— 2.93 cm/px at the 15 m default, independent of the orthomosaic's own resolution. A
fixed-scale model loses 5.2 F1 across a +/-30% zoom, so if results look inconsistent
between surveys, check this before blaming the model. It must match the scale the model
was trained at, and it is the same arithmetic the WINMOL Analyzer applies to its
`tile_size`.

Tiles overlap by `--overlap` and their probabilities are averaged, because a single pass
leaves seams exactly where a stem crosses a tile edge.

This produces a raster. **Vectorising it into stem polylines is the WINMOL Analyzer's
job**, not this repo's — hand the Analyzer the same ONNX and let it do the tracing.
"""
import argparse
import os
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="infer.py",
        description="Run a model over an orthomosaic; write a georeferenced stem map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Vectorising the raster into stem polylines is the WINMOL Analyzer's job.")
    p.add_argument("--model", required=True, help="contract-conformant .onnx")
    p.add_argument("--ortho", required=True, help="orthomosaic GeoTIFF")
    p.add_argument("--out", required=True, help="output probability GeoTIFF")
    p.add_argument("--aoi", default=None,
                   help="restrict inference to this polygon. Strongly recommended: "
                        "outside the windthrow polygon stems are real but undigitised.")
    p.add_argument("--extent-m", type=float, default=15.0,
                   help="tile footprint in metres; effective GSD is extent_m/512. "
                        "MUST match the scale the model was trained at.")
    p.add_argument("--native-px", type=int, default=None,
                   help="tile at the ortho's own resolution instead of a fixed footprint")
    p.add_argument("--ref-gsd-cm", type=float, default=2.9297,
                   help="reference grid resolution for the output raster")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="tile overlap fraction; probabilities are averaged where tiles "
                        "overlap, which is what removes seams at stem crossings")
    p.add_argument("--threshold", type=float, default=None,
                   help="also write a thresholded binary mask next to the probability")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    import numpy as np
    import rasterio

    from winmol_unet.geo.predict import predict_plot

    prob, tf, W, H, meta = predict_plot(
        args.ortho, args.aoi, args.model,
        extent_m=None if args.native_px else args.extent_m,
        native_px=args.native_px, ref_gsd=args.ref_gsd_cm / 100.0,
        overlap=args.overlap, batch=args.batch, threads=args.threads, quiet=args.quiet)

    crs = meta.get("crs") if isinstance(meta, dict) else None
    profile = dict(driver="GTiff", width=W, height=H, count=1, dtype="float32",
                   crs=crs, transform=tf, compress="deflate")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(prob.astype("float32"), 1)
    if not args.quiet:
        print(f"wrote {args.out}  ({W}x{H} @ {args.ref_gsd_cm:.4f} cm/px)")

    if args.threshold is not None:
        stem, ext = os.path.splitext(args.out)
        mask_path = f"{stem}_mask{ext or '.tif'}"
        profile.update(dtype="uint8")
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write((prob >= args.threshold).astype("uint8"), 1)
        if not args.quiet:
            print(f"wrote {mask_path}  (threshold {args.threshold})")
    return 0

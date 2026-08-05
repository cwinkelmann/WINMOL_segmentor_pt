"""Run a trained ONNX segmenter over an orthomosaic -> georeferenced stem map.

    python scripts/predict_stem_map.py --ortho <ortho.tif> --model <model.onnx> \
        --out <stem_map.tif> [--dem <dem.tif> --depth-vmin M --depth-vmax M] \
        [--tile 512] [--overlap 64] [--threshold 0.5]

Used to supply conditioning masks where the analyzer's own stem map is missing
or empty. The result is a *prediction*, so it inherits the segmenter's errors —
acceptable for ControlNet, which needs plausible geometry paired with real
appearance, but never as evaluation ground truth. Anything trained downstream
must still be scored against hand-labelled data, or the circularity (our
segmenter teaches a generator that teaches our segmenter) goes unmeasured.

Tiles overlap and are blended by taking the maximum probability, so stems
crossing a tile seam are not clipped at the boundary.

With --dem the fourth channel is read from a DEM/DSM/CHM in the same world
window as each ortho tile, so a 4-channel (--rgbd) model can be run over a whole
site. The channel count is taken from the ONNX graph, not from a flag: a
3-channel model with --dem, or a 4-channel model without, is a mistake that
would otherwise surface as a shape error deep in onnxruntime or, worse, as a
silently wrong map.

--depth-vmin/--depth-vmax MUST match what the model was trained with. Depth
normalized over a different range is a different input distribution, and the
model will quietly underperform rather than fail.
"""
import argparse
import sys

import numpy as np


def _dem_window(dem, src, x0, y0, w, h, tile, vmin, vmax, nodata):
    """Read the DEM over the same world box as an ortho window, at tile resolution."""
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    left, top = src.xy(y0, x0, offset="ul")
    right, bottom = src.xy(y0 + h, x0 + w, offset="ul")
    win = from_bounds(left, bottom, right, top, transform=dem.transform)
    z = dem.read(1, window=win, out_shape=(tile, tile), boundless=True,
                 fill_value=np.nan, resampling=Resampling.bilinear).astype(np.float32)
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= z != nodata
    out = np.zeros_like(z)
    if valid.any():
        lo = float(z[valid].min()) if vmin is None else float(vmin)
        hi = float(z[valid].max()) if vmax is None else float(vmax)
        if hi > lo:
            out[valid] = np.clip((z[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def predict(ortho_path, model_path, out_path, tile, overlap, threshold,
            dem_path=None, depth_vmin=None, depth_vmax=None, dem_nodata=None):
    import rasterio
    from winmol_unet.runtime import OnnxSegmenter

    seg = OnnxSegmenter(model_path)
    # the graph is the authority on channel count; a mismatch here is a silent
    # wrong-answer bug, so fail before writing anything
    if seg.in_channels == 4 and dem_path is None:
        raise SystemExit("model expects 4 channels (RGBD): pass --dem")
    if seg.in_channels == 3 and dem_path is not None:
        raise SystemExit("model expects 3 channels (RGB): drop --dem")
    if seg.in_channels == 4 and (depth_vmin is None or depth_vmax is None):
        print("warning: no --depth-vmin/--depth-vmax; depth is scaled per tile, which "
              "almost certainly differs from training", file=sys.stderr)

    src = rasterio.open(ortho_path)
    dem = rasterio.open(dem_path) if dem_path else None
    if dem is not None and dem.crs != src.crs:
        raise SystemExit(f"DEM CRS {dem.crs} != ortho CRS {src.crs}; reproject first")
    h, w = src.height, src.width
    prob = np.zeros((h, w), np.float32)
    step = tile - overlap
    n_tiles = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            y0, x0 = min(y, max(0, h - tile)), min(x, max(0, w - tile))
            win = rasterio.windows.Window(x0, y0, min(tile, w - x0), min(tile, h - y0))
            a = src.read(indexes=[1, 2, 3], window=win)
            a = np.transpose(a, (1, 2, 0)).astype(np.float32) / 255.0
            if a.shape[0] != tile or a.shape[1] != tile:      # edge tile
                pad = np.zeros((tile, tile, 3), np.float32)
                pad[: a.shape[0], : a.shape[1]] = a
                a = pad
            if a.mean() < 0.05:                               # nodata; skip
                continue
            if dem is not None:
                z = _dem_window(dem, src, x0, y0, int(win.width), int(win.height),
                                tile, depth_vmin, depth_vmax, dem_nodata)
                a = np.concatenate([a, z[..., None]], axis=2)   # HWC -> HW4
            p = seg.predict_on_batch(a[None])[0, :, :, 0]
            ph, pw = min(tile, h - y0), min(tile, w - x0)
            # max-blend the overlap so a stem is never cut at a seam
            prob[y0:y0 + ph, x0:x0 + pw] = np.maximum(
                prob[y0:y0 + ph, x0:x0 + pw], p[:ph, :pw])
            n_tiles += 1
            if n_tiles % 25 == 0:
                print(f"  {n_tiles} tiles inferred", flush=True)

    mask = (prob >= threshold).astype(np.uint8) * 255
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    for k in ("photometric", "alpha"):
        profile.pop(k, None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"{n_tiles} tiles inferred; stem coverage {100 * (mask > 0).mean():.2f}% "
          f"-> {out_path}")
    return mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ortho", required=True)
    p.add_argument("--model", required=True, help="contract-conformant ONNX segmenter")
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dem", default=None,
                   help="DEM/DSM/CHM for the 4th channel; required for an --rgbd model")
    p.add_argument("--depth-vmin", type=float, default=None,
                   help="must match the range the model was trained with")
    p.add_argument("--depth-vmax", type=float, default=None)
    p.add_argument("--dem-nodata", type=float, default=None)
    a = p.parse_args(argv)
    predict(a.ortho, a.model, a.out, a.tile, a.overlap, a.threshold,
            dem_path=a.dem, depth_vmin=a.depth_vmin, depth_vmax=a.depth_vmax,
            dem_nodata=a.dem_nodata)
    return 0


if __name__ == "__main__":
    sys.exit(main())

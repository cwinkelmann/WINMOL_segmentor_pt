"""Run a trained ONNX segmenter over an orthomosaic -> georeferenced stem map.

    python scripts/predict_stem_map.py --ortho <ortho.tif> --model <model.onnx> \
        --out <stem_map.tif> [--tile 512] [--overlap 64] [--threshold 0.5]

Used to supply conditioning masks where the analyzer's own stem map is missing
or empty. The result is a *prediction*, so it inherits the segmenter's errors —
acceptable for ControlNet, which needs plausible geometry paired with real
appearance, but never as evaluation ground truth. Anything trained downstream
must still be scored against hand-labelled data, or the circularity (our
segmenter teaches a generator that teaches our segmenter) goes unmeasured.

Tiles overlap and are blended by taking the maximum probability, so stems
crossing a tile seam are not clipped at the boundary.
"""
import argparse
import sys

import numpy as np


def predict(ortho_path, model_path, out_path, tile, overlap, threshold):
    import rasterio
    from winmol_unet.runtime import OnnxSegmenter

    seg = OnnxSegmenter(model_path)
    src = rasterio.open(ortho_path)
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
            p = seg.predict_on_batch(a[None])[0, :, :, 0]
            ph, pw = min(tile, h - y0), min(tile, w - x0)
            # max-blend the overlap so a stem is never cut at a seam
            prob[y0:y0 + ph, x0:x0 + pw] = np.maximum(
                prob[y0:y0 + ph, x0:x0 + pw], p[:ph, :pw])
            n_tiles += 1

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
    a = p.parse_args(argv)
    predict(a.ortho, a.model, a.out, a.tile, a.overlap, a.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())

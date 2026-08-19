"""Run a model over a plot's AOI in world space and score it on a common reference grid.

**Why this exists.** Two models trained at different ground resolutions cannot be compared
on their own tile sets: a 1.2 cm/px tile and a 2.93 cm/px tile are different exams, because
a stem is ~2.4x thicker in pixels on the first. Tile-level F1 then measures the exam, not
the model. The only fair comparison is the one the deployment actually performs — predict
over the *same ground*, write both predictions into the *same* world-space grid, and score
against the same ground truth there.

Each model keeps its own internal tiling (that is the thing being compared): a tile covers
`extent_m` of ground, is resized to the model's 512 px input, and its probability map is
resampled back onto the reference grid and averaged where tiles overlap. Overlap matters —
a single pass leaves seams exactly where a stem crosses a tile edge.

Scoring is masked to the AOI shrunk by `--edge-buffer-m`, because a tile centred near the
boundary sees unannotated ground outside it, and stems there are real but undigitised.

    python infer.py --ortho x.tif --aoi aoi.gpkg --stems stems.gpkg \\
        --model m.onnx --extent-m 19.512 --ref-gsd-cm 2.9297
"""
import argparse
import json
import math
import os
import sys

import numpy as np



def _grid(bounds, ref_gsd):
    """Reference raster geometry covering `bounds` at `ref_gsd` m/px."""
    from affine import Affine
    minx, miny, maxx, maxy = bounds
    w = int(math.ceil((maxx - minx) / ref_gsd))
    h = int(math.ceil((maxy - miny) / ref_gsd))
    return Affine(ref_gsd, 0, minx, 0, -ref_gsd, maxy), w, h


def predict_plot(ortho, aoi, model, extent_m=None, native_px=None, ref_gsd=0.029297,
                 img_size=512, overlap=0.5, threshold=0.5, batch=4, threads=None,
                 quiet=False):
    """Probability raster over the AOI on a common reference grid.

    Returns (prob, tf, W, H, area, crs, meta) -- seven values. `crs` is the source
    ortho's CRS; `meta` carries tile counts and the three GSDs, and deliberately does
    NOT repeat crs or area.
    """
    import onnxruntime as ort
    import rasterio
    from PIL import Image
    from rasterio.windows import from_bounds
    from shapely.ops import unary_union

    from .sample import _load_geoms

    src = rasterio.open(ortho)
    gsd = abs(src.transform.a)
    if native_px:                      # native mode: footprint is a pixel count
        extent_m = native_px * gsd
    if not extent_m:
        raise SystemExit("give --extent-m or --native-px")

    area = unary_union(_load_geoms(aoi, src.crs, None, "aoi", quiet=True)[0])
    tf, W, H = _grid(area.bounds, ref_gsd)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    nhwc = len(inp.shape) == 4 and inp.shape[-1] in (3, "3")

    step = extent_m * (1.0 - overlap)
    minx, miny, maxx, maxy = area.bounds
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    todo = []
    for y in ys:
        for x in xs:
            from shapely.geometry import box as sbox
            if area.intersects(sbox(x, y, x + extent_m, y + extent_m)):
                todo.append((x, y))

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        imgs, boxes = [], []
        for x, y in chunk:
            win = from_bounds(x, y, x + extent_m, y + extent_m, src.transform)
            a = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0,
                         out_shape=(3, img_size, img_size))
            imgs.append(np.transpose(a, (1, 2, 0)).astype(np.float32) / 255.0)
            boxes.append((x, y))
        x_in = np.stack(imgs)
        if not nhwc:
            x_in = np.transpose(x_in, (0, 3, 1, 2))
        out = sess.run(None, {inp.name: x_in.astype(np.float32)})[0]
        out = np.squeeze(out, axis=1) if out.ndim == 4 else out
        for (x, y), p in zip(boxes, np.atleast_3d(out).reshape(-1, img_size, img_size)):
            # place this tile's probabilities into the reference grid
            c0 = int(round((x - minx) / ref_gsd))
            r1 = int(round((maxy - y) / ref_gsd))
            n = int(round(extent_m / ref_gsd))
            r0, c1 = r1 - n, c0 + n
            if n < 1:
                continue
            tile = np.asarray(Image.fromarray(p).resize((n, n), Image.BILINEAR), np.float32)
            rr0, rr1 = max(r0, 0), min(r1, H)
            cc0, cc1 = max(c0, 0), min(c1, W)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            acc[rr0:rr1, cc0:cc1] += tile[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
            cnt[rr0:rr1, cc0:cc1] += 1.0
    prob = np.where(cnt > 0, acc / np.maximum(cnt, 1e-6), 0.0)
    meta = {"tiles": len(todo), "extent_m": extent_m, "ortho_gsd_cm": gsd * 100,
            "model_gsd_cm": extent_m / img_size * 100, "ref_gsd_cm": ref_gsd * 100,
            "grid": [H, W], "covered_frac": float((cnt > 0).mean())}
    return prob, tf, W, H, area, src.crs, meta


def score(prob, tf, W, H, area, crs, stems, threshold=0.5, edge_buffer_m=0.0,
          thresholds=None):
    """F1/precision/recall inside the AOI, on the reference grid."""
    from rasterio.features import rasterize as rio_rasterize

    from .sample import _load_geoms

    geoms, _ = _load_geoms(stems, crs, None, "stems", quiet=True)
    gt = rio_rasterize([(g, 1) for g in geoms], out_shape=(H, W), transform=tf,
                       fill=0).astype(bool)
    inner = area.buffer(-edge_buffer_m) if edge_buffer_m else area
    if inner.is_empty:
        raise SystemExit(f"--edge-buffer-m {edge_buffer_m} leaves no AOI")
    valid = rio_rasterize([(inner, 1)], out_shape=(H, W), transform=tf,
                          fill=0).astype(bool)
    out = []
    for t in (thresholds or [threshold]):
        pred = (prob > t) & valid
        g = gt & valid
        tp = int((pred & g).sum())
        fp = int((pred & ~g).sum())
        fn = int((~pred & g).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        out.append({"threshold": t, "f1": 2 * p * r / max(p + r, 1e-9),
                    "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
                    "gt_px": int(g.sum()), "valid_px": int(valid.sum())})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ortho", required=True)
    p.add_argument("--aoi", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--extent-m", type=float, default=None)
    p.add_argument("--native-px", type=int, default=None)
    p.add_argument("--ref-gsd-cm", type=float, default=2.9297)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--edge-buffer-m", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sweep-thresholds", action="store_true",
                   help="also report F1 across thresholds (calibration check)")
    p.add_argument("--batch", type=int, default=4,
                   help="tiles per ONNX call; raise it when reads dominate")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    prob, tf, W, H, area, crs, meta = predict_plot(
        a.ortho, a.aoi, a.model, a.extent_m, a.native_px, a.ref_gsd_cm / 100.0,
        overlap=a.overlap, threads=a.threads, batch=a.batch)
    ths = [round(x, 2) for x in np.arange(0.2, 0.81, 0.05)] if a.sweep_thresholds else None
    rows = score(prob, tf, W, H, area, crs, a.stems, a.threshold, a.edge_buffer_m, ths)
    at = [r for r in rows if abs(r["threshold"] - a.threshold) < 1e-9] or rows
    best = max(rows, key=lambda r: r["f1"])
    label = a.label or os.path.basename(a.model)
    print(f"{label}   {meta['tiles']} tiles of {meta['extent_m']:.2f} m "
          f"(model {meta['model_gsd_cm']:.2f} cm/px) -> ref {meta['ref_gsd_cm']:.2f} cm/px, "
          f"{meta['covered_frac']*100:.1f}% covered")
    print(f"  @{at[0]['threshold']:.2f}  F1 {at[0]['f1']:.4f}  P {at[0]['precision']:.4f}  "
          f"R {at[0]['recall']:.4f}")
    if ths:
        print(f"  best  @{best['threshold']:.2f}  F1 {best['f1']:.4f}  "
              f"P {best['precision']:.4f}  R {best['recall']:.4f}")
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"label": label, "model": a.model, "meta": meta, "rows": rows}, f,
                      indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

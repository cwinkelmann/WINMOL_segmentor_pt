"""Generate distillation labels by running the REAL WINMOL_Analyzer heuristic over the
segmentor's training tiles.

This replaces the single hand-uploaded plot (and the synthetic `corrupt_mask`) with a fleet of
genuine teacher outputs, and -- crucially -- lets the teacher run on the **UNet's own predicted
masks**, which is the distribution the analyzer actually sees in deployment.

Two phases, each resumable (existing outputs are skipped):

  1. predict   train/*.jpeg --(UNet ONNX, batched)--> mask PNG      [--source pred]
               or just copy the ground-truth mask/*.gif             [--source gt]
  2. vectorize mask --(analyzer Skeletonization+Vectorization+Quantification)--> per-tile .gpkg

    python learned_vectorization/build_teacher_dataset.py \
        --data-dir datasets/SpecDS_ready --out work/teacher_specds \
        --source pred --model .../twostage_lrfix/unet.onnx --workers 8

Run inside `winmol-vec` with the analyzer repo mounted at /analyzer (it carries fiona/pyogrio/
matplotlib/onnxruntime for exactly this). The analyzer is imported lazily inside the workers so
the pure helpers here stay unit-testable without it.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

# The analyzer's target ground resolution: Config.tile_size / Config.img_width = 15/512 m/px.
# The vectorizer's thresholds (min_length 2.0 m, max_distance 8 m, measuring_point_spacing 0.5 m)
# are all in METRES, so this constant decides whether its output is sane. Do not guess it.
ANALYZER_GSD = 15.0 / 512.0

IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".tif", ".tiff")


def tile_key(filename):
    """Map `train_100_10.jpeg` / `mask_100_10.gif` -> `100_10`, and `train1`/`mask1` -> `1`.

    The two datasets on disk disagree (GenDS10 uses `train_<a>_<b>`, SpecDS_ready uses `train<N>`),
    and StemDataset's integer pairing can't handle the former -- so pair on the stem-name suffix.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    for prefix in ("train", "mask", "image", "img"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return base.lstrip("_")


def pair_tiles(data_dir):
    """[(key, image_path, mask_path)] for a loader-format dataset dir (train/ + mask/)."""
    img_dir, mask_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "mask")
    images = {}
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(IMAGE_EXTS):
            images[tile_key(f)] = os.path.join(img_dir, f)
    masks = {}
    if os.path.isdir(mask_dir):
        for f in sorted(os.listdir(mask_dir)):
            masks[tile_key(f)] = os.path.join(mask_dir, f)
    return [(k, images[k], masks.get(k)) for k in sorted(images) if k in masks]


def tile_profile(height, width, gsd=ANALYZER_GSD, minx=0.0, maxy=0.0, epsg=25833):
    """A minimal north-up rasterio profile so the analyzer can work in metres.

    The tiles are plain JPEGs with no georeferencing; the heuristic only needs a consistent
    scale, so we place every tile at a synthetic origin at the analyzer's own resolution.
    """
    import rasterio
    from rasterio.transform import from_origin
    return {
        "driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "uint8",
        "crs": rasterio.crs.CRS.from_epsg(epsg),
        "transform": from_origin(minx, maxy, gsd, gsd),
    }


def load_binary_mask(path, thresh=127):
    from PIL import Image
    return (np.asarray(Image.open(path).convert("L")) > thresh).astype(np.uint8)


def _load_batch(paths, size=512):
    from PIL import Image
    batch = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        batch.append(np.asarray(im, np.float32) / 255.0)
    return np.stack(batch).transpose(0, 3, 1, 2)           # NHWC -> NCHW, the contract input


def predict_masks(predict_fn, tiles, out_dir, batch_size=8, thresh=0.5, size=512):
    """Phase 1: run the UNet over the tiles, writing a binary mask PNG per tile.

    `predict_fn(NCHW float32 in [0,1]) -> (N,1,H,W) probabilities`.
    """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    todo = [(k, p) for k, p, _ in tiles if not os.path.exists(os.path.join(out_dir, k + ".png"))]
    print(f"predict: {len(todo)} tiles to do ({len(tiles) - len(todo)} already done)", flush=True)

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        y = predict_fn(_load_batch([p for _, p in chunk], size))
        for (k, _), prob in zip(chunk, y):
            m = (prob[0] >= thresh).astype(np.uint8) * 255
            Image.fromarray(m).save(os.path.join(out_dir, k + ".png"))
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)


def make_torch_predictor(ckpt_path, device="cuda"):
    """UNet state-dict -> predict_fn on the GPU.

    Preferred over ONNX here: the vec image ships plain `onnxruntime` (CPU-only), which turned
    out to be ~5 s/tile -- hours for the sweep -- while torch in the same image has CUDA.
    The .pt is a bare state dict for winmol_unet.model.UNet; sigmoid is NOT baked in (that is
    added at ONNX export), so apply it here.
    """
    import torch
    from winmol_unet.model import UNet

    net = UNet()
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    net.eval().to(device)

    def predict_fn(x):
        with torch.no_grad():
            t = torch.from_numpy(x).to(device)
            return torch.sigmoid(net(t)).cpu().numpy()
    return predict_fn


def make_onnx_predictor(model_path):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(model_path, providers=providers)
    print(f"onnx {os.path.basename(model_path)} on {sess.get_providers()[0]}", flush=True)
    iname = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {iname: x})[0]         # sigmoid baked in at export


def vectorize_one(job):
    """Phase 2 worker: one binary mask -> one .gpkg via the analyzer's heuristic.

    Returns a small dict (never the analyzer's objects) so it survives the process boundary.
    """
    key, mask_path, out_dir, analyzer_dir, quiet = job
    out_prefix = os.path.join(out_dir, key)
    if os.path.exists(out_prefix + ".gpkg"):
        return {"key": key, "status": "cached"}

    if analyzer_dir not in sys.path:
        sys.path.insert(0, analyzer_dir)
    from utils.VectorTilePipeline import process_prediction_array_to_gpkg

    mask = load_binary_mask(mask_path)
    if mask.sum() == 0:
        return {"key": key, "status": "empty", "stems": 0}
    prof = tile_profile(*mask.shape)
    cfg = {"vector_summary_log": not quiet, "prediction_tile_log": not quiet}
    try:
        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
        with ctx:
            res = process_prediction_array_to_gpkg(mask, prof, cfg, "Stems", out_prefix)
    except Exception as exc:                     # one bad tile must not kill the sweep
        return {"key": key, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if not res or not res.get("gpkg_path"):
        return {"key": key, "status": "no_output", "stems": 0}
    return {"key": key, "status": "ok", "stems": int(res.get("stem_count", 0)),
            "fg": int(res.get("fg_count", 0)), "secs": float(res.get("timings", {}).get("total_s", 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="loader-format dataset (train/ + mask/)")
    ap.add_argument("--out", required=True, help="output dir (masks/ + gpkg/ + manifest.json)")
    ap.add_argument("--source", choices=["pred", "gt"], default="pred",
                    help="pred = UNet output (deployment distribution); gt = ground-truth masks")
    ap.add_argument("--model", help="UNet .pt (torch, GPU) or .onnx; required for --source pred")
    ap.add_argument("--device", default="cuda", help="device for a .pt model")
    ap.add_argument("--limit", type=int, help="only the first N tiles (smoke runs)")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--analyzer-dir", default="/analyzer")
    ap.add_argument("--verbose", action="store_true", help="let the analyzer print per tile")
    args = ap.parse_args()

    tiles = pair_tiles(args.data_dir)
    if args.limit:
        tiles = tiles[:args.limit]
    print(f"{len(tiles)} paired tiles in {args.data_dir}", flush=True)

    mask_dir = os.path.join(args.out, "masks")
    gpkg_dir = os.path.join(args.out, "gpkg")
    os.makedirs(gpkg_dir, exist_ok=True)

    if args.source == "pred":
        if not args.model:
            ap.error("--source pred needs --model")
        if args.model.endswith(".pt"):
            print(f"torch {os.path.basename(args.model)} on {args.device}", flush=True)
            predict_fn = make_torch_predictor(args.model, args.device)
        else:
            predict_fn = make_onnx_predictor(args.model)
        predict_masks(predict_fn, tiles, mask_dir, batch_size=args.batch_size)
        mask_for = lambda k: os.path.join(mask_dir, k + ".png")
    else:
        mask_for = lambda k: dict((t[0], t[2]) for t in tiles)[k]

    jobs = [(k, mask_for(k), gpkg_dir, args.analyzer_dir, not args.verbose)
            for k, _, _ in tiles]

    from multiprocessing import Pool
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(vectorize_one, jobs, chunksize=4), 1):
            results.append(r)
            if i % 100 == 0 or i == len(jobs):
                ok = sum(1 for x in results if x["status"] in ("ok", "cached"))
                stems = sum(x.get("stems", 0) for x in results)
                print(f"  vectorized {i}/{len(jobs)} | ok {ok} | stems so far {stems}", flush=True)

    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    manifest = {"data_dir": args.data_dir, "source": args.source, "model": args.model,
                "gsd": ANALYZER_GSD, "n_tiles": len(tiles), "status": by_status,
                "total_stems": sum(r.get("stems", 0) for r in results),
                "results": sorted(results, key=lambda r: r["key"])}
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nstatus {by_status} | total stems {manifest['total_stems']} -> {args.out}", flush=True)
    errs = [r for r in results if r["status"] == "error"][:5]
    for e in errs:
        print(f"  e.g. {e['key']}: {e['error']}", flush=True)


if __name__ == "__main__":
    main()

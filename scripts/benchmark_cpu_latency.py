"""Benchmark ONNX UNet **CPU** inference: latency (batch 1 / batch 4) + held-out F1.

The reusable measuring stick for the CPU-inference-speedup study
(docs/superpowers/specs/2026-07-21-cpu-inference-speedup-design.md).

Latency = pure `session.run` time on the CPU EP at a pinned intra-op thread count
(warmup + timed runs, min/median/p90/mean). Accuracy = beech-TestDS F1 using the SAME
definition as scripts/benchmark_architectures.py::_onnx_test_metrics (StemDataset ->
predict_on_batch -> 0.5 threshold -> micro-averaged TP/FP/FN), so numbers are comparable
to every earlier report.

Runs CPU-only (forces CPUExecutionProvider). Intended for the winmol-test docker image.
"""
import argparse
import json
import os
import time

import numpy as np

from winmol_unet.contract import IMG_SIZE, INPUT_NAME, OUTPUT_NAME


def latency_stats(samples_ms):
    """Summarize a list of per-run latencies (ms) -> min/median/p90/mean + count."""
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        "runs": int(a.size),
        "min_ms": float(a.min()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "mean_ms": float(a.mean()),
    }


def build_session(onnx_path, threads):
    """A CPU-EP InferenceSession with full graph opt and a pinned intra-op thread count."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def measure_latency(onnx_path, threads=4, batch=1, warmup=5, runs=30, tile=None):
    """Time session.run on a fixed NCHW input; return latency_stats + config."""
    sess = build_session(onnx_path, threads)
    if tile is not None:
        x = np.ascontiguousarray(np.broadcast_to(tile, (batch, 3, IMG_SIZE, IMG_SIZE)),
                                 dtype=np.float32)
    else:
        x = np.zeros((batch, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    feed = {INPUT_NAME: x}
    for _ in range(warmup):
        sess.run([OUTPUT_NAME], feed)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run([OUTPUT_NAME], feed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"threads": int(threads), "batch": int(batch), **latency_stats(samples)}


def testds_f1(onnx_path, test_data_dir, img_size=IMG_SIZE, batch=4, max_tiles=None):
    """Held-out F1 on a StemDataset directory via the CPU EP (exact fp32/quantized)."""
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
    from training.dataset import StemDataset
    from winmol_unet.runtime import OnnxSegmenter

    ds = StemDataset(os.path.join(test_data_dir, "train"),
                     os.path.join(test_data_dir, "mask"),
                     img_size, transform=None, cache=False)
    seg = OnnxSegmenter(onnx_path)
    n = len(ds) if max_tiles is None else min(len(ds), max_tiles)
    tp = fp = fn = 0
    for start in range(0, n, batch):
        idx = list(range(start, min(start + batch, n)))
        xs = np.stack([ds[i][0].permute(1, 2, 0).numpy() for i in idx])
        preds = np.asarray(seg.predict_on_batch(xs))[..., 0]
        for k, i in enumerate(idx):
            p, g = preds[k] >= 0.5, ds[i][1][0].numpy() >= 0.5
            tp += int((p & g).sum()); fp += int((p & ~g).sum()); fn += int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "n": n}


def _real_tile(test_data_dir, img_size):
    """First test tile as an NCHW [0,1] float32 array (representative latency input)."""
    from training.dataset import StemDataset
    ds = StemDataset(os.path.join(test_data_dir, "train"),
                     os.path.join(test_data_dir, "mask"),
                     img_size, transform=None, cache=False)
    return ds[0][0].numpy()  # CHW [0,1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="label=path.onnx or just path.onnx")
    ap.add_argument("--test-data-dir", default=None, help="StemDataset dir for F1 (train/ + mask/)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thread-sweep", default=None, help="comma list, e.g. 1,2,4,8 (batch-1 latency)")
    ap.add_argument("--batches", default="1,4", help="comma list of batch sizes")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b]
    tile = None
    if args.test_data_dir:
        tile = _real_tile(args.test_data_dir, IMG_SIZE)[None]  # 1CHW for broadcast

    rows = []
    for spec in args.models:
        label, path = (spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec))
        size_mb = os.path.getsize(path) / 1e6
        row = {"label": label, "path": path, "size_mb": round(size_mb, 1), "latency": {}}
        for b in batches:
            row["latency"][b] = measure_latency(path, args.threads, b, args.warmup, args.runs, tile)
        if args.thread_sweep:
            row["thread_sweep"] = {
                int(t): measure_latency(path, int(t), 1, args.warmup, args.runs, tile)
                for t in args.thread_sweep.split(",") if t
            }
        if args.test_data_dir:
            row["accuracy"] = testds_f1(path, args.test_data_dir)
        rows.append(row)
        b1 = row["latency"].get(1, {})
        acc = row.get("accuracy", {})
        print(f"{label}: b1 median {b1.get('median_ms', float('nan')):.1f}ms "
              f"p90 {b1.get('p90_ms', float('nan')):.1f}ms | "
              f"F1 {acc.get('f1', float('nan')):.4f} | {size_mb:.0f}MB")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"threads": args.threads, "rows": rows}, f, indent=2)
    if args.md_out:
        base = rows[0]["latency"][1]["median_ms"] if rows else 1.0
        lines = [f"# CPU latency + accuracy ({args.threads} intra-op threads)", "",
                 "| Model | b1 median (ms) | b1 p90 | speedup | b4 median | F1 | ΔF1 | MB |",
                 "|---|--:|--:|--:|--:|--:|--:|--:|"]
        base_f1 = rows[0].get("accuracy", {}).get("f1") if rows else None
        for r in rows:
            b1, b4 = r["latency"][1], r["latency"].get(4, {})
            f1 = r.get("accuracy", {}).get("f1")
            df1 = (f1 - base_f1) if (f1 is not None and base_f1 is not None) else None
            lines.append(
                f"| {r['label']} | {b1['median_ms']:.1f} | {b1['p90_ms']:.1f} | "
                f"{base / b1['median_ms']:.2f}× | {b4.get('median_ms', float('nan')):.1f} | "
                f"{'' if f1 is None else f'{f1:.4f}'} | "
                f"{'' if df1 is None else f'{df1:+.4f}'} | {r['size_mb']:.0f} |")
        with open(args.md_out, "w") as f:
            f.write("\n".join(lines) + "\n")

    return rows


if __name__ == "__main__":
    main()

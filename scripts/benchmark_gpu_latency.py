"""Benchmark UNet **GPU** inference latency: width scaling x precision (fp32/fp16) x batch.

Companion to benchmark_cpu_latency.py. ONNX Runtime in this image is CPU-only, so GPU latency
is measured with PyTorch eager on CUDA -- the actual on-GPU compute, and a fair proxy for an
onnxruntime-gpu / TensorRT deployment (which is typically no slower than eager). Timing wraps
torch.cuda.synchronize() so async kernel launches are fully accounted for.

fp16 (model.half()) is the GPU analogue of the CPU int8 lever: Tensor-Core acceleration,
near-lossless. Reports latency (median/p90) and throughput (img/s) per (width, dtype, batch),
plus speedup vs the fp32 full-width model at the same batch.
"""
import argparse
import time

import torch

from winmol_unet.model import UNet
from benchmark_cpu_latency import latency_stats


def _bench(model, x, warmup, runs):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", default="1.0,0.5,0.25")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    widths = [float(w) for w in args.widths.split(",")]
    dtypes = args.dtypes.split(",")
    batches = [int(b) for b in args.batches.split(",")]

    rows = []
    base = {}  # (batch) -> fp32 full-width median, for speedup
    for w in widths:
        for dt in dtypes:
            model = UNet(width_mult=w).eval().to(dev)
            if dt == "fp16":
                model = model.half()
            tdt = torch.float16 if dt == "fp16" else torch.float32
            for b in batches:
                x = torch.zeros(b, 3, args.img_size, args.img_size, dtype=tdt, device=dev)
                s = _bench(model, x, args.warmup, args.runs)
                s.update(width=w, dtype=dt, batch=b,
                         throughput=b / (s["median_ms"] / 1000.0))
                if w == 1.0 and dt == "fp32":
                    base[b] = s["median_ms"]
                rows.append(s)
                print(f"w{w} {dt} b{b}: median {s['median_ms']:.1f}ms "
                      f"p90 {s['p90_ms']:.1f}ms | {s['throughput']:.1f} img/s")
            del model
            torch.cuda.empty_cache()

    lines = [f"# GPU latency — {torch.cuda.get_device_name(0)} (PyTorch eager, CUDA)", "",
             "Latency = median of timed forwards (cuda.synchronize'd). Speedup vs fp32 width-1.0 "
             "at the same batch. fp16 = Tensor-Core half precision (near-lossless).", "",
             "| width | dtype | batch | median (ms) | p90 | img/s | speedup |",
             "|------:|:-----:|------:|------------:|----:|------:|--------:|"]
    for r in rows:
        sp = base.get(r["batch"], r["median_ms"]) / r["median_ms"]
        lines.append(f"| {r['width']} | {r['dtype']} | {r['batch']} | {r['median_ms']:.1f} | "
                     f"{r['p90_ms']:.1f} | {r['throughput']:.1f} | {sp:.2f}× |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()

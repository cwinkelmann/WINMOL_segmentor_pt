#!/usr/bin/env python
"""Benchmark all segmentation architectures the SAME way and write markdown summaries.

Trains `unet`, `deeplabv3plus`, `hrnet` with identical two-stage (GenDS -> SpecDS)
settings (same data, epochs, augmentation, seed), then for each records: config,
parameter count, per-epoch val-F1 curves, final species-val metrics, wall-clock, and the
ONNX-served F1 (OnnxSegmenter on the CPU EP for an exact, machine-independent number).

Outputs, under --out-dir:
  <arch>.md    per-architecture detail (one per arch)
  SUMMARY.md   comparison table with a placeholder row to fill in from the R-style run

Usage:
  python scripts/benchmark_architectures.py \
    --gen-data-dir  /Users/christian/data/Winmol/data/SpecDS \
    --spec-data-dir /Users/christian/data/Winmol/data/spruce/SpecDS_ready \
    --out-dir results --epochs 20 --device mps --no-cache-dataset --num-workers 4
"""
import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np

# make the dev-only `training` package importable when run as `python scripts/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainConfig
from training.dataset import train_val_split
from training.model_factory import build_model
from training.run_train import run_two_stage
from winmol_unet.runtime import OnnxSegmenter

ARCHS = ["unet", "deeplabv3plus", "hrnet"]


def _param_count(arch, cfg):
    m = build_model(arch, dropout=cfg.dropout, encoder=cfg.encoder, encoder_weights=None)
    return sum(p.numel() for p in m.parameters())


def _val_f1_curve(log_dir, stage):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(os.path.join(log_dir, stage, "events*")))
    if not fs:
        return []
    ea = EventAccumulator(fs[-1]); ea.Reload()
    if "val/f1" not in ea.Tags().get("scalars", []):
        return []
    return [round(e.value, 4) for e in ea.Scalars("val/f1")]


def _onnx_served_f1(onnx_path, spec_data_dir, cfg, max_tiles=60):
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"     # exact fp32, machine-independent
    _, val = train_val_split(os.path.join(spec_data_dir, "train"),
                             os.path.join(spec_data_dir, "mask"),
                             cfg.val_fraction, cfg.seed, cfg.img_size, cache=False)
    seg = OnnxSegmenter(onnx_path)
    n = min(max_tiles, len(val))
    xs = np.stack([val[i][0].permute(1, 2, 0).numpy() for i in range(n)])
    preds = np.asarray(seg.predict_on_batch(xs))[..., 0]
    tp = fp = fn = 0
    for i in range(n):
        p, g = preds[i] >= 0.5, val[i][1][0].numpy() >= 0.5
        tp += int((p & g).sum()); fp += int((p & ~g).sum()); fn += int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return f1, n


def _write_arch_md(path, arch, cfg, r):
    m = r["metrics"]
    md = f"""# {arch} — two-stage benchmark result

**Date:** {r['date']}
**Regime:** two-stage GenDS -> SpecDS (fine-tune)
**Config:** encoder={cfg.encoder}, encoder_weights={cfg.encoder_weights}, epochs={cfg.epochs}, \
batch_size={cfg.batch_size}, lr={cfg.lr}, device={cfg.device}, seed={cfg.seed}, \
patience={cfg.patience_stage1}/{cfg.patience_stage2}
**Augmentation:** hflip={cfg.aug_hflip_p} vflip={cfg.aug_vflip_p} rotate={cfg.aug_rotate_p} \
bc={cfg.aug_bc_p} hsv={cfg.aug_hsv_p}
**Data:** GenDS=`{cfg.gen_data_dir}`, SpecDS=`{cfg.spec_data_dir}`
**Parameters:** {r['params']:,}
**Train time:** {r['minutes']:.1f} min

## Final species-val metrics

| metric | value |
|--------|------:|
| F1 | {m['f1']:.4f} |
| precision | {m['precision']:.4f} |
| recall | {m['recall']:.4f} |
| loss | {m['loss']:.4f} |

**ONNX-served F1** (OnnxSegmenter, CPU EP, {r['onnx_tiles']} val tiles): **{r['onnx_f1']:.4f}**

## Per-epoch val F1

- Stage 1 (GenDS): {r['stage1_f1']}
- Stage 2 (SpecDS): {r['stage2_f1']}

## Artifacts

- `{r['onnx_out']}`
- `{r['pt_out']}`
"""
    with open(path, "w") as f:
        f.write(md)


def _write_summary_md(path, cfg, results):
    rows = [
        "# Architecture benchmark — PyTorch (to compare vs R-style UNet)",
        "",
        f"**Regime:** two-stage GenDS -> SpecDS | **epochs:** {cfg.epochs} | "
        f"**encoder:** {cfg.encoder}/{cfg.encoder_weights} | **seed:** {cfg.seed}",
        f"**GenDS:** `{cfg.gen_data_dir}` | **SpecDS:** `{cfg.spec_data_dir}`",
        "",
        "| Architecture | Params | Best val F1 | Final F1 | Precision | Recall | ONNX F1 | Train (min) |",
        "|--------------|-------:|------------:|---------:|----------:|-------:|--------:|------------:|",
    ]
    for arch in ARCHS:
        r = results.get(arch)
        if r is None:
            rows.append(f"| {arch} | — | (failed) | — | — | — | — | — |")
            continue
        m = r["metrics"]
        best = max(r["stage2_f1"]) if r["stage2_f1"] else m["f1"]
        rows.append(
            f"| {arch} | {r['params'] / 1e6:.1f}M | {best:.4f} | {m['f1']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {r['onnx_f1']:.4f} | {r['minutes']:.1f} |")
    rows += [
        "| **R UNet (reference)** | _TBD_ | _fill from R run_ | _fill_ | _fill_ | _fill_ | — | _fill_ |",
        "",
        "Per-architecture detail: the individual `<arch>.md` files in this directory.",
        "The R reference row is filled in later from the R-style two-stage UNet run on the "
        "same GenDS -> SpecDS data.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def benchmark(cfg_base, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for arch in ARCHS:
        run_dir = os.path.join(out_dir, arch)
        cfg = replace(cfg_base, arch=arch,
                      checkpoint_dir=os.path.join(run_dir, "checkpoints"),
                      log_dir=os.path.join(run_dir, "logs"),
                      pt_out=os.path.join(run_dir, "model.pt"),
                      onnx_out=os.path.join(run_dir, "model.onnx"),
                      hdf5_out=os.path.join(run_dir, "model.hdf5"),
                      keras_out=os.path.join(run_dir, "model.keras"))
        print(f"=== {arch} ===", flush=True)
        t0 = time.time()
        try:
            metrics = run_two_stage(cfg)
        except Exception as e:                       # keep going; a failed arch -> (failed) row
            print(f"{arch} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        onnx_f1, tiles = _onnx_served_f1(cfg.onnx_out, cfg.spec_data_dir, cfg)
        results[arch] = {
            "metrics": metrics, "minutes": (time.time() - t0) / 60,
            "params": _param_count(arch, cfg),
            "stage1_f1": _val_f1_curve(cfg.log_dir, "stage1"),
            "stage2_f1": _val_f1_curve(cfg.log_dir, "stage2"),
            "onnx_f1": onnx_f1, "onnx_tiles": tiles,
            "onnx_out": cfg.onnx_out, "pt_out": cfg.pt_out,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        _write_arch_md(os.path.join(out_dir, f"{arch}.md"), arch, cfg, results[arch])
        print(f"{arch}: val F1 {metrics['f1']:.3f}, ONNX F1 {onnx_f1:.3f}, "
              f"{results[arch]['minutes']:.1f} min", flush=True)
    _write_summary_md(os.path.join(out_dir, "SUMMARY.md"), cfg_base, results)
    print(f"wrote {out_dir}/SUMMARY.md")
    return results


def _none_or_str(v):
    return None if str(v).lower() in ("none", "") else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-data-dir", required=True)
    p.add_argument("--spec-data-dir", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false", default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    cfg_base = TrainConfig(
        data_dir="", gen_data_dir=a.gen_data_dir, spec_data_dir=a.spec_data_dir,
        checkpoint_dir=a.out_dir, log_dir=a.out_dir, hdf5_out="", onnx_out="",
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        encoder=a.encoder, encoder_weights=_none_or_str(a.encoder_weights),
        cache_dataset=a.cache_dataset, num_workers=a.num_workers, seed=a.seed,
    )
    benchmark(cfg_base, a.out_dir)


if __name__ == "__main__":
    main()

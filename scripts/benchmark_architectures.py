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
import csv
import glob
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

# make the dev-only `training` package importable when run as `python scripts/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainConfig
from training.dataset import StemDataset, train_val_split
from training.model_factory import build_model
from training.run_train import run_training, run_two_stage
from winmol_unet.runtime import OnnxSegmenter

ARCHS = ["unet", "deeplabv3plus", "hrnet"]


def _param_count(arch, cfg):
    m = build_model(arch, dropout=cfg.dropout, encoder=cfg.encoder, encoder_weights=None)
    return sum(p.numel() for p in m.parameters())


def _onnx_served_f1(onnx_path, spec_data_dir, cfg, max_tiles=60, batch=4):
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"     # exact fp32, machine-independent
    _, val = train_val_split(os.path.join(spec_data_dir, "train"),
                             os.path.join(spec_data_dir, "mask"),
                             cfg.val_fraction, cfg.seed, cfg.img_size, cache=False)
    seg = OnnxSegmenter(onnx_path)
    n = min(max_tiles, len(val))
    tp = fp = fn = 0
    # Chunk the ONNX inference: a UNet at 512^2 with a 60-tile batch peaks at ~37 GB and
    # OOM-kills the host — process a few tiles at a time to bound memory.
    for start in range(0, n, batch):
        idx = list(range(start, min(start + batch, n)))
        xs = np.stack([val[i][0].permute(1, 2, 0).numpy() for i in idx])
        preds = np.asarray(seg.predict_on_batch(xs))[..., 0]
        for k, i in enumerate(idx):
            p, g = preds[k] >= 0.5, val[i][1][0].numpy() >= 0.5
            tp += int((p & g).sum()); fp += int((p & ~g).sum()); fn += int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return f1, n


def _epoch_metrics(log_dir, stage):
    """Full per-epoch scalars for a stage from TensorBoard: train_loss, val_loss,
    val_precision/recall/f1, lr (aligned by step). Mirrors the R metrics.csv columns."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(os.path.join(log_dir, stage, "events*")))
    if not fs:
        return []
    ea = EventAccumulator(fs[-1]); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    wanted = {"train/loss": "train_loss", "val/loss": "val_loss",
              "val/precision": "val_precision", "val/recall": "val_recall",
              "val/f1": "val_f1", "lr": "lr"}
    by_step = {}
    for tag, col in wanted.items():
        if tag not in tags:
            continue
        for e in ea.Scalars(tag):
            by_step.setdefault(e.step, {})[col] = round(e.value, 5)
    return [{"epoch": i + 1, **by_step[s]} for i, s in enumerate(sorted(by_step))]


def _write_metrics_csv(path, stage1, stage2):
    cols = ["stage", "epoch", "train_loss", "val_loss", "val_precision", "val_recall",
            "val_f1", "lr"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for stage, rows in ((1, stage1), (2, stage2)):
            for r in rows:
                w.writerow({"stage": stage, **{k: r.get(k, "") for k in cols if k != "stage"}})


def _onnx_test_metrics(onnx_path, test_data_dir, cfg, batch=4):
    """Held-out TestDS metrics via the exported ONNX (OnnxSegmenter, CPU EP, exact fp32) over
    ALL test tiles — the machine-independent held-out number mirroring R's cost_eval(TestDS)."""
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
    ds = StemDataset(os.path.join(test_data_dir, "train"), os.path.join(test_data_dir, "mask"),
                     cfg.img_size, transform=None, cache=False)
    seg = OnnxSegmenter(onnx_path)
    n = len(ds)
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

**ONNX-served F1** (OnnxSegmenter, CPU EP, {r['onnx_tiles']} SpecDS-val tiles): **{r['onnx_f1']:.4f}**
"""
    t = r.get("test")
    if t:
        md += f"""
## Held-out test metrics (TestDS: `{cfg.test_data_dir}`, ONNX CPU EP, {t['n']} tiles)

| metric | value |
|--------|------:|
| F1 | {t['f1']:.4f} |
| precision | {t['precision']:.4f} |
| recall | {t['recall']:.4f} |
"""
    md += f"""
## Per-epoch val F1

- Stage 1 (GenDS): {r['stage1_f1']}
- Stage 2 (SpecDS): {r['stage2_f1']}

Full per-epoch metrics (train_loss/val_*/lr, both stages): `{arch}/metrics.csv`

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
        f"**GenDS:** `{cfg.gen_data_dir}` | **SpecDS:** `{cfg.spec_data_dir}` | "
        f"**TestDS:** `{cfg.test_data_dir}`",
        "",
        "| Architecture | Params | Best val F1 | Final val F1 | TestDS F1 | TestDS P | TestDS R | Train (min) |",
        "|--------------|-------:|------------:|-------------:|----------:|---------:|---------:|------------:|",
    ]
    for arch in ARCHS:
        r = results.get(arch)
        if r is None:
            rows.append(f"| {arch} | — | (failed) | — | — | — | — | — |")
            continue
        m = r["metrics"]
        best = max(r["stage2_f1"]) if r["stage2_f1"] else m["f1"]
        t = r.get("test") or {}
        tf1 = f"{t['f1']:.4f}" if t else "—"
        tp = f"{t['precision']:.4f}" if t else "—"
        tr = f"{t['recall']:.4f}" if t else "—"
        rows.append(
            f"| {arch} | {r['params'] / 1e6:.1f}M | {best:.4f} | {m['f1']:.4f} | "
            f"{tf1} | {tp} | {tr} | {r['minutes']:.1f} |")
    rows += [
        "| **R UNet (reference)** | _TBD_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |",
        "",
        "TestDS columns are the held-out ONNX-served metrics (CPU EP). Per-architecture detail:"
        " the individual `<arch>.md` files + `<arch>/metrics.csv`.",
        "The R reference row is filled by `write_comparison_summary.py` from the R run.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def benchmark(cfg_base, out_dir, skip_stage1=False):
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
        if skip_stage1:   # single-stage ablation: train directly on SpecDS, no GenDS10 pretrain
            cfg = replace(cfg, data_dir=cfg.spec_data_dir, patience=cfg.patience_stage1)
        print(f"=== {arch}{' (single-stage)' if skip_stage1 else ''} ===", flush=True)
        t0 = time.time()
        try:
            metrics = run_training(cfg) if skip_stage1 else run_two_stage(cfg)
        except Exception as e:                       # keep going; a failed arch -> (failed) row
            print(f"{arch} FAILED (train): {type(e).__name__}: {e}", flush=True)
            continue
        # Post-processing is wrapped per-step so a helper bug can NEVER discard the (expensive)
        # training result or abort the remaining architectures — each falls back gracefully.
        def _try(label, fn, default):
            try:
                return fn()
            except Exception as e:
                print(f"{arch} {label} failed: {type(e).__name__}: {e}", flush=True)
                return default

        onnx_f1, tiles = _try("onnx-val-F1",
                              lambda: _onnx_served_f1(cfg.onnx_out, cfg.spec_data_dir, cfg),
                              (float("nan"), 0))
        # single-stage logs straight to log_dir (no stage1/stage2 subdirs); its curve is s2.
        s1 = [] if skip_stage1 else _try("stage1-epochs",
                                         lambda: _epoch_metrics(cfg.log_dir, "stage1"), [])
        s2 = _try("stage-epochs",
                  lambda: _epoch_metrics(cfg.log_dir, "" if skip_stage1 else "stage2"), [])
        _try("metrics.csv", lambda: _write_metrics_csv(os.path.join(run_dir, "metrics.csv"), s1, s2), None)
        test_m = _try("TestDS-onnx",
                      lambda: _onnx_test_metrics(cfg.onnx_out, cfg.test_data_dir, cfg)
                      if cfg.test_data_dir else None, None)
        results[arch] = {
            "metrics": metrics, "minutes": (time.time() - t0) / 60,
            "params": _try("param-count", lambda: _param_count(arch, cfg), 0),
            "stage1_f1": [round(r["val_f1"], 4) for r in s1 if "val_f1" in r],
            "stage2_f1": [round(r["val_f1"], 4) for r in s2 if "val_f1" in r],
            "onnx_f1": onnx_f1, "onnx_tiles": tiles, "test": test_m,
            "onnx_out": cfg.onnx_out, "pt_out": cfg.pt_out,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        _try("arch.md", lambda: _write_arch_md(os.path.join(out_dir, f"{arch}.md"), arch, cfg,
                                               results[arch]), None)
        # Dump JSON incrementally so a later arch's failure can't lose completed archs.
        _try("json", lambda: json.dump(results, open(os.path.join(out_dir,
             "pytorch_results.json"), "w"), indent=2), None)
        tf1 = f", TestDS F1 {test_m['f1']:.3f}" if test_m else ""
        print(f"{arch}: val F1 {metrics['f1']:.3f}, ONNX F1 {onnx_f1:.3f}{tf1}, "
              f"{results[arch]['minutes']:.1f} min", flush=True)
    _write_summary_md(os.path.join(out_dir, "SUMMARY.md"), cfg_base, results)
    with open(os.path.join(out_dir, "pytorch_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir}/SUMMARY.md + pytorch_results.json")
    return results


def _none_or_str(v):
    return None if str(v).lower() in ("none", "") else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-data-dir", required=True)
    p.add_argument("--spec-data-dir", required=True)
    p.add_argument("--test-data-dir", default=None,
                   help="held-out test set (TestDS); ONNX-served metrics reported per arch")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false", default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-stage1", action="store_true",
                   help="single-stage ablation: train on SpecDS only (no GenDS10 pretraining)")
    a = p.parse_args()
    cfg_base = TrainConfig(
        data_dir="", gen_data_dir=a.gen_data_dir, spec_data_dir=a.spec_data_dir,
        test_data_dir=a.test_data_dir,
        checkpoint_dir=a.out_dir, log_dir=a.out_dir, hdf5_out="", onnx_out="",
        epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        encoder=a.encoder, encoder_weights=_none_or_str(a.encoder_weights),
        cache_dataset=a.cache_dataset, num_workers=a.num_workers, seed=a.seed,
        # Mirror the R input_pipeline.R augmentation: 50% up/down + left/right flips,
        # no rotation, and mild always-on photometric jitter (random_brightness 0.1,
        # random_contrast 0.95-1.05, random_saturation 0.95-1.05, random_hue 0.1).
        aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=0.0,
        aug_bc_p=1.0, aug_brightness_limit=0.1, aug_contrast_limit=0.05,
        aug_hsv_p=1.0, aug_hue_shift=18, aug_sat_shift=13, aug_val_shift=0,
    )
    benchmark(cfg_base, a.out_dir, skip_stage1=a.skip_stage1)


if __name__ == "__main__":
    main()

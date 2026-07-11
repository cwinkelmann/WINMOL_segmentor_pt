#!/usr/bin/env python
"""Resolution ablation: two-stage UNet at a chosen --img-size (default 256, matching the R
pipeline's downsample) with everything else identical to the 512 benchmark (GenDS10->SpecDS
two-stage, carried-over optimizer/LR, R-matched aug, from scratch). Isolates the resolution
effect vs the 512 run and R.

Torch-only eval (no ONNX export — the ONNX contract pins 512); the held-out TestDS metric is
computed on the torch model, directly comparable to R's Keras `evaluate` and the 512 run's
torch test_results.md.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.run_train import _build_loaders
from training.augment import build_augmentation
from training.model_factory import build_model
from training.train import train_one_run
from training.evaluate import evaluate
from training.dataset import StemDataset
from training.device import resolve_device


def _epoch_metrics(log_dir, stage):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(os.path.join(log_dir, stage, "events*")))
    if not fs:
        return []
    ea = EventAccumulator(fs[-1]); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    wanted = {"train/loss": "train_loss", "val/loss": "val_loss", "val/precision": "val_precision",
              "val/recall": "val_recall", "val/f1": "val_f1", "lr": "lr"}
    by_step = {}
    for tag, col in wanted.items():
        if tag not in tags:
            continue
        for e in ea.Scalars(tag):
            by_step.setdefault(e.step, {})[col] = round(e.value, 5)
    return [{"epoch": i + 1, **by_step[s]} for i, s in enumerate(sorted(by_step))]


def _write_csv(path, s1, s2):
    cols = ["stage", "epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1", "lr"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for st, rows in ((1, s1), (2, s2)):
            for r in rows:
                w.writerow({"stage": st, **{k: r.get(k, "") for k in cols if k != "stage"}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-data-dir", required=True)
    p.add_argument("--spec-data-dir", required=True)
    p.add_argument("--test-data-dir", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    run_dir = os.path.join(a.out_dir, f"unet_{a.img_size}")
    cfg = TrainConfig(
        data_dir="", gen_data_dir=a.gen_data_dir, spec_data_dir=a.spec_data_dir,
        test_data_dir=a.test_data_dir,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"), log_dir=os.path.join(run_dir, "logs"),
        hdf5_out="", onnx_out="", pt_out=os.path.join(run_dir, "model.pt"),
        img_size=a.img_size, epochs=a.epochs, batch_size=a.batch_size, device=a.device,
        num_workers=a.num_workers, cache_dataset=False, seed=a.seed, arch="unet",
        encoder_weights=None,
        aug_hflip_p=0.5, aug_vflip_p=0.5, aug_rotate_p=0.0,
        aug_bc_p=1.0, aug_brightness_limit=0.1, aug_contrast_limit=0.05,
        aug_hsv_p=1.0, aug_hue_shift=18, aug_sat_shift=13, aug_val_shift=0)

    print(f"=== unet @ {a.img_size}px (resolution ablation) ===", flush=True)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    transform = build_augmentation(cfg)
    model = build_model("unet", dropout=cfg.dropout)
    model.to(resolve_device(cfg.device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)   # shared across stages (match R)

    gen_tr, gen_va = _build_loaders(cfg.gen_image_dir, cfg.gen_mask_dir, cfg, transform)
    train_one_run(model, gen_tr, gen_va, cfg, patience=cfg.patience_stage1,
                  ckpt_name="best_stage1.pt", log_dir=os.path.join(cfg.log_dir, "stage1"),
                  optimizer=opt)
    spec_tr, spec_va = _build_loaders(cfg.spec_image_dir, cfg.spec_mask_dir, cfg, transform)
    train_one_run(model, spec_tr, spec_va, cfg, patience=cfg.patience_stage2,
                  ckpt_name="best_stage2.pt", log_dir=os.path.join(cfg.log_dir, "stage2"),
                  optimizer=opt)

    val_m = evaluate(model, spec_va)
    test_ds = StemDataset(os.path.join(a.test_data_dir, "train"),
                          os.path.join(a.test_data_dir, "mask"), cfg.img_size,
                          transform=None, cache=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_m = evaluate(model, test_loader)

    s1 = _epoch_metrics(cfg.log_dir, "stage1")
    s2 = _epoch_metrics(cfg.log_dir, "stage2")
    _write_csv(os.path.join(run_dir, "metrics.csv"), s1, s2)
    params = sum(q.numel() for q in model.parameters())
    res = {"img_size": a.img_size, "params": params, "minutes": (time.time() - t0) / 60,
           "val": val_m, "test": test_m,
           "stage1_f1": [round(r["val_f1"], 4) for r in s1 if "val_f1" in r],
           "stage2_f1": [round(r["val_f1"], 4) for r in s2 if "val_f1" in r]}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "ablation_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out_dir, f"unet_{a.img_size}.md"), "w") as f:
        f.write(f"# UNet @ {a.img_size}px — resolution ablation\n\n"
                f"Params {params/1e6:.1f}M | {res['minutes']:.1f} min | two-stage GenDS10->SpecDS, "
                f"carry-LR, from scratch, torch eval\n\n"
                f"**Held-out TestDS (torch):** F1 {test_m['f1']:.4f} | P {test_m['precision']:.4f} "
                f"| R {test_m['recall']:.4f} | loss {test_m['loss']:.4f}\n\n"
                f"**SpecDS val:** F1 {val_m['f1']:.4f}\n\n"
                f"Stage-2 val F1 curve: {res['stage2_f1']}\n")
    print(f"unet@{a.img_size}: val F1 {val_m['f1']:.3f}, TestDS F1 {test_m['f1']:.3f}, "
          f"{res['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

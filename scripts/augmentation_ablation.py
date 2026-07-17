#!/usr/bin/env python
"""Augmentation ablation: does each augmentation actually make the model better?

For every (architecture, augmentation-arm) pair, train single-stage on the same
train/val/test split and record held-out TestDS F1 (the R cost_eval number). Arms are
grouped so comparisons stay apples-to-apples:

  RESIZE group      — classic pipeline (whole tile resized to img_size); every arm sees the
                      SAME resized val/test, so 'none' vs 'flip' vs 'rotate' vs 'photometric'
                      are directly comparable. Answers "do flips / rotation / photometric
                      jitter help?".
  MULTISCALE group  — the native-fidelity multi-scale pipeline (cfg.multiscale: rotate the
                      full native tile -> RandomSizedCrop -> flips -> photometric), evaluated
                      on deterministic native-res grid tiling (cfg.eval_tiling). Tests whether
                      the native multi-scale dataloader beats plain resize. NOT comparable to
                      the RESIZE group (different eval pipeline) — reported in its own block.

Usage (single-stage on a 3-way split):
  python scripts/augmentation_ablation.py \
    --train-dir <DS>/train --val-dir <DS>/val --test-dir <DS>/test \
    --out-dir results/aug_ablation --epochs 100 --device cuda \
    --no-cache-dataset --num-workers 8 --encoder resnet34 --encoder-weights imagenet
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainConfig
from training.model_factory import build_model
from training.run_train import run_training

ARCHS = ["unet", "deeplabv3plus", "hrnet"]

# Each arm is a set of aug-config overrides on top of the base cfg. `group` keeps
# resize-pipeline arms separate from native-crop arms in the report (different eval scale).
# Photometric magnitudes: 'mild' mirrors the R input_pipeline.R ranges (the classic
# baseline); 'strong' uses the larger albumentations defaults (the "new" photometric).
_MILD = dict(aug_bc_p=1.0, aug_brightness_limit=0.1, aug_contrast_limit=0.05,
             aug_hsv_p=1.0, aug_hue_shift=18, aug_sat_shift=13, aug_val_shift=0)
_STRONG = dict(aug_bc_p=1.0, aug_brightness_limit=0.2, aug_contrast_limit=0.2,
               aug_hsv_p=1.0, aug_hue_shift=20, aug_sat_shift=30, aug_val_shift=20)
_FLIP = dict(aug_hflip_p=0.5, aug_vflip_p=0.5)
_ROTATE = dict(aug_rotate_p=0.5, aug_rotate_limit=15.0)
_OFF = dict(aug_hflip_p=0.0, aug_vflip_p=0.0, aug_rotate_p=0.0, aug_bc_p=0.0, aug_hsv_p=0.0,
            multiscale=False, eval_tiling=False)

ARMS = [
    # RESIZE group: whole tile -> img_size; identical resize-to-img_size eval -> comparable.
    ("none",         "resize",     {**_OFF}),
    ("flip",         "resize",     {**_OFF, **_FLIP}),
    ("classic",      "resize",     {**_OFF, **_FLIP, **_MILD}),                    # R-matched
    ("rotate",       "resize",     {**_OFF, **_FLIP, **_MILD, **_ROTATE}),
    ("photometric+", "resize",     {**_OFF, **_FLIP, **_STRONG}),
    ("all_resize",   "resize",     {**_OFF, **_FLIP, **_STRONG, **_ROTATE}),
    # MULTISCALE group: native-res multi-scale RandomSizedCrop (rotate the full tile ->
    # RandomSizedCrop -> flips -> photometric), evaluated on deterministic native-res grid
    # tiling. Different eval from the RESIZE group -> compare to the RESIZE arms only with
    # that caveat. Tests whether the native multi-scale pipeline beats plain resize.
    ("multiscale",   "multiscale", {**_OFF, **_FLIP, **_STRONG, **_ROTATE,
                                    "multiscale": True, "eval_tiling": True}),
]


def _param_count(arch, cfg):
    m = build_model(arch, dropout=cfg.dropout, encoder=cfg.encoder, encoder_weights=None)
    return sum(p.numel() for p in m.parameters())


def _read_test_md(path):
    """Parse the test_results.md that run_training writes (its held-out TestDS eval, applied
    with the arm's own pipeline incl. crop). Returns {f1,precision,recall,loss} or {}."""
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    out = {}
    for k in ("f1", "precision", "recall", "loss"):
        m = re.search(rf"\|\s*{k}\s*\|\s*([0-9.]+)\s*\|", txt)
        if m:
            out[k] = float(m.group(1))
    return out


def _run_one(arch, arm_name, overrides, cfg_base, out_dir):
    run_dir = os.path.join(out_dir, arch, arm_name)
    cfg = replace(cfg_base, arch=arch,
                  checkpoint_dir=os.path.join(run_dir, "checkpoints"),
                  log_dir=os.path.join(run_dir, "logs"),
                  pt_out=os.path.join(run_dir, "model.pt"),
                  onnx_out=os.path.join(run_dir, "model.onnx"),
                  hdf5_out=os.path.join(run_dir, "model.hdf5"),
                  keras_out=os.path.join(run_dir, "model.keras"),
                  **overrides)
    print(f"=== {arch} / {arm_name} ===", flush=True)
    t0 = time.time()
    val = run_training(cfg)                          # trains, held-out test eval, ONNX export
    test = _read_test_md(os.path.join(run_dir, "test_results.md"))
    minutes = (time.time() - t0) / 60
    print(f"{arch}/{arm_name}: val F1 {val['f1']:.3f}, TestDS F1 "
          f"{test.get('f1', float('nan')):.3f}, {minutes:.1f} min", flush=True)
    return {"arch": arch, "arm": arm_name, "group": None, "val": val, "test": test,
            "minutes": minutes}


def _write_summary(path, cfg, results):
    by = {(r["arch"], r["arm"]): r for r in results}
    lines = [
        "# Augmentation ablation — does each augmentation improve held-out F1?",
        "",
        f"**Regime:** single-stage | **epochs:** {cfg.epochs} | **encoder:** "
        f"{cfg.encoder}/{cfg.encoder_weights} | **seed:** {cfg.seed}",
        f"**train:** `{cfg.data_dir}` | **val:** `{cfg.val_data_dir}` | "
        f"**test:** `{cfg.test_data_dir}`",
        "",
        "TestDS F1 is the held-out number, evaluated with each arm's own pipeline (RESIZE "
        "arms on tiles resized to img_size; the MULTISCALE arm on native-res grid tiles). "
        "Compare **within** a group only — the groups use different eval pipelines.",
    ]
    for group, title in (("resize", "RESIZE group (resize-to-img_size eval — mutually comparable)"),
                         ("multiscale", "MULTISCALE group (native grid-tiling eval)")):
        arms = [a for a, g, _ in ARMS if g == group]
        lines += ["", f"## {title}", "",
                  "| Arm | " + " | ".join(ARCHS) + " | (best) |",
                  "|-----|" + "|".join(["------:"] * (len(ARCHS) + 1)) + "|"]
        for arm in arms:
            cells, best = [], None
            for arch in ARCHS:
                r = by.get((arch, arm))
                f1 = r["test"].get("f1") if r and r.get("test") else None
                cells.append(f"{f1:.4f}" if f1 is not None else "—")
                if f1 is not None:
                    best = f1 if best is None else max(best, f1)
            best_cell = f"{best:.4f}" if best is not None else "—"
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {best_cell} |")
    lines += ["", "Per-run detail: `<arch>/<arm>/test_results.md` + TensorBoard logs.",
              "R WINMOL_segmentor single-stage baseline is reported separately."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True, help="single-stage training set (train/ + mask/)")
    p.add_argument("--val-dir", required=True, help="fixed validation split")
    p.add_argument("--test-dir", required=True, help="held-out test set (TestDS)")
    p.add_argument("--out-dir", default="results/aug_ablation")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false", default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--img-size", type=int, default=512,
                   help="model input size (RESIZE arms resize the tile to this; the MULTISCALE "
                        "arm's RandomSizedCrop/tiling produces this)")
    p.add_argument("--crop-min-px", type=int, default=400,
                   help="MULTISCALE arm: min crop side sampled from the native tile")
    p.add_argument("--crop-max-px", type=int, default=1024,
                   help="MULTISCALE arm: max crop side (>= native tile -> full tile)")
    p.add_argument("--archs", default=",".join(ARCHS),
                   help="comma-separated subset of unet,deeplabv3plus,hrnet")
    p.add_argument("--arms", default="", help="comma-separated arm subset (default: all)")
    a = p.parse_args()
    archs = [x for x in a.archs.split(",") if x]
    arms = [arm for arm in ARMS if not a.arms or arm[0] in a.arms.split(",")]
    ew = None if str(a.encoder_weights).lower() in ("none", "") else a.encoder_weights
    # each of --train-dir/--val-dir/--test-dir is a split dir holding train/ + mask/ subdirs
    # (config.image_dir = data_dir/train, mask_dir = data_dir/mask).
    cfg_base = TrainConfig(
        data_dir=a.train_dir,
        val_data_dir=a.val_dir, test_data_dir=a.test_dir,
        checkpoint_dir=a.out_dir, log_dir=a.out_dir, hdf5_out="", onnx_out="",
        epochs=a.epochs, batch_size=a.batch_size, patience=a.patience, device=a.device,
        encoder=a.encoder, encoder_weights=ew, cache_dataset=a.cache_dataset,
        num_workers=a.num_workers, seed=a.seed, img_size=a.img_size,
        crop_min_px=a.crop_min_px, crop_max_px=a.crop_max_px)
    os.makedirs(a.out_dir, exist_ok=True)
    results = []
    for arch in archs:
        for arm_name, group, overrides in arms:
            try:
                r = _run_one(arch, arm_name, overrides, cfg_base, a.out_dir)
                r["group"] = group
                r["params"] = _param_count(arch, cfg_base)
                results.append(r)
            except Exception as e:                   # keep going; one failed arm != lost run
                print(f"{arch}/{arm_name} FAILED: {type(e).__name__}: {e}", flush=True)
            json.dump(results, open(os.path.join(a.out_dir, "ablation_results.json"), "w"),
                      indent=2)
            _write_summary(os.path.join(a.out_dir, "SUMMARY.md"), cfg_base, results)
    print(f"wrote {a.out_dir}/SUMMARY.md + ablation_results.json", flush=True)


if __name__ == "__main__":
    main()

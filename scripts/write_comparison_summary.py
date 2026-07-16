#!/usr/bin/env python3
"""Assemble the R-vs-PyTorch comparison SUMMARY.md from the R + PyTorch run artifacts.

Reads <base>/r/metrics.csv + <base>/r/final.json (from parse_r_run.py) and
<base>/pytorch_results.json (from benchmark_architectures.py). Stdlib only.

Usage: python scripts/write_comparison_summary.py <base_dir>
"""
import csv
import json
import os
import sys

ARCHS = ["unet", "deeplabv3plus", "hrnet"]


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _r_stage_curve(csv_path, stage):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        return [round(float(r["val_f1"]), 4) for r in csv.DictReader(f)
                if int(r["stage"]) == stage and r["val_f1"]]


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    base = sys.argv[1]
    regime = sys.argv[2] if len(sys.argv) > 2 else (
        "two-stage GenDS10 (beech, stage 1) -> SpecDS (stage 2 fine-tune), then a held-out "
        "**TestDS** evaluation — the canonical `main_training.R` pipeline")
    r_final = _load_json(os.path.join(base, "r", "final.json"))
    pt = _load_json(os.path.join(base, "pytorch_results.json")) or {}
    r_curve2 = _r_stage_curve(os.path.join(base, "r", "metrics.csv"), 2)
    r_curve1 = _r_stage_curve(os.path.join(base, "r", "metrics.csv"), 1)

    L = [
        "# R (Keras) vs PyTorch — WINMOL tree-stem segmentation benchmark",
        "",
        f"**Regime:** {regime}, applied identically to the R U-Net and each PyTorch "
        "architecture.",
        "**Protocol:** 80/20 train/val split per stage, early-stop on val_loss (patience 3/5), "
        "Adam 1e-3, loss BCE+(1-F1), ReduceLROnPlateau(0.1, patience 2), batch 4, 100 "
        "epochs/stage cap. PyTorch trains from scratch (encoder_weights=None) at 512x512; R "
        "U-Net at 256x256.",
        "",
        "## Held-out comparison (primary result)",
        "",
        "| Model | Params | Best val F1 | Final val F1 | **TestDS F1** | TestDS P | TestDS R | Train (min) |",
        "|-------|-------:|------------:|-------------:|--------------:|---------:|---------:|------------:|",
    ]

    # R reference row
    if r_final:
        bv = (r_final.get("best_val") or {}).get("val_f1")
        fv = r_curve2[-1] if r_curve2 else (r_curve1[-1] if r_curve1 else None)
        t = r_final.get("test") or {}
        L.append(
            f"| **R U-Net (Keras)** | {r_final.get('params', 0)/1e6:.1f}M | {_fmt(bv)} | "
            f"{_fmt(fv)} | **{_fmt(t.get('f1'))}** | {_fmt(t.get('precision'))} | "
            f"{_fmt(t.get('recall'))} | — |")
    else:
        L.append("| **R U-Net (Keras)** | — | (no R run) | — | — | — | — | — |")

    # PyTorch rows
    for arch in ARCHS:
        r = pt.get(arch)
        if not r:
            L.append(f"| {arch} (PyTorch) | — | (failed/absent) | — | — | — | — | — |")
            continue
        best = max(r["stage2_f1"]) if r.get("stage2_f1") else r["metrics"]["f1"]
        t = r.get("test") or {}
        L.append(
            f"| {arch} (PyTorch) | {r['params']/1e6:.1f}M | {_fmt(best)} | "
            f"{_fmt(r['metrics']['f1'])} | **{_fmt(t.get('f1'))}** | {_fmt(t.get('precision'))} | "
            f"{_fmt(t.get('recall'))} | {r['minutes']:.1f} |")

    L += ["", "TestDS = held-out ONNX-served metrics (CPU EP, exact fp32) for PyTorch; R uses "
          "Keras `evaluate` on the same TestDS. Best/Final val F1 are the stage-2 (fine-tune) "
          "validation split.", ""]

    # Per-epoch stage-2 val-F1 curves
    L += ["## Stage-2 (SpecDS fine-tune) per-epoch val F1", ""]
    if r_final:
        L.append(f"- **R U-Net:** {r_curve2 or r_curve1}")
    for arch in ARCHS:
        r = pt.get(arch)
        if r:
            L.append(f"- **{arch}:** {r.get('stage2_f1', [])}")
    L += ["", "Full per-epoch metrics (both stages, train_loss/val_*/lr): `r/metrics.csv` and "
          "`<arch>/metrics.csv`. Per-architecture detail: `<arch>.md`.", ""]

    # Resolution ablation (256 vs 512), if present
    abl = _load_json(os.path.join(base, "ablation_results.json"))
    if abl:
        u = pt.get("unet") or {}
        u512 = (u.get("test") or {}).get("f1")
        rt = (r_final or {}).get("test") or {}
        L += [
            "## Resolution ablation — isolating 256 vs 512 (UNet, held-out TestDS F1)",
            "",
            "Same UNet, same two-stage protocol/aug/carry-LR/from-scratch — only the input "
            "resolution changes. This is the controlled test for whether the PyTorch gain is "
            "resolution.",
            "",
            "| UNet variant | Resolution | TestDS F1 | TestDS P | TestDS R |",
            "|--------------|-----------:|----------:|---------:|---------:|",
            f"| PyTorch (512) | 512x512 | {_fmt(u512)} | "
            f"{_fmt((u.get('test') or {}).get('precision'))} | "
            f"{_fmt((u.get('test') or {}).get('recall'))} |",
            f"| PyTorch (256, ablation) | 256x256 | {_fmt(abl['test']['f1'])} | "
            f"{_fmt(abl['test']['precision'])} | {_fmt(abl['test']['recall'])} |",
            f"| R U-Net (256) | 256x256 | {_fmt(rt.get('f1'))} | {_fmt(rt.get('precision'))} | "
            f"{_fmt(rt.get('recall'))} |",
            "",
            "Read: PyTorch-512 vs PyTorch-256 isolates resolution (only that differs); "
            "PyTorch-256 vs R-256 isolates framework/architecture/LR-trajectory at matched "
            "resolution. Note: the 512 figure is ONNX-served, the 256 ablation and R are "
            "native-framework eval (fp32; the ONNX-vs-torch difference is negligible).",
            "",
        ]

    # Caveats
    L += [
        "## Caveats",
        "- **TestDS is the beech/Zenodo test set** (byte-identical to `beech/TestDS`), so the "
        "held-out metric measures generalization to that set, not in-distribution spruce. The "
        "same TestDS is used for all models, so the comparison is fair.",
        "- R U-Net runs at 256x256; PyTorch at 512x512 (ONNX contract). PyTorch is from scratch "
        "(no ImageNet); the R U-Net is also from scratch.",
        "- R logs train Precision/Recall/F1 per epoch; PyTorch logs train loss only (val "
        "P/R/F1 both). GenDS10->SpecDS leakage was not fully excluded (see the R repo's "
        "`runs/*/leakage_*` checks).",
    ]
    out = os.path.join(base, "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

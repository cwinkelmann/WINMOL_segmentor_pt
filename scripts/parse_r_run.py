#!/usr/bin/env python3
"""Parse an R/Keras two-stage training stdout log into metrics.csv + final.json.

Handles the R container (`docker/run_training.R`) output: `[stage1]`/`[stage2]` markers,
per-epoch Keras lines (`... loss: X - Precision: X - Recall: X - F1Score: X - val_loss: X
- val_Precision: X - val_Recall: X - val_F1Score: X [- lr: X]`), and the final
`Evaluation on <TEST>:` named-vector score. Stdlib only.

Usage: python scripts/parse_r_run.py <train_stdout.log> <out_dir>
  writes <out_dir>/metrics.csv (cols: stage,epoch,loss,precision,recall,f1,val_loss,
  val_precision,val_recall,val_f1,lr) and <out_dir>/final.json.
"""
import csv
import json
import os
import re
import sys

EPOCH = re.compile(
    r"loss: ([\d.]+) - Precision: ([\d.]+) - Recall: ([\d.]+) - F1Score: ([\d.]+)"
    r" - val_loss: ([\d.]+) - val_Precision: ([\d.]+) - val_Recall: ([\d.]+)"
    r" - val_F1Score: ([\d.]+)(?: - lr: ([\d.eE+-]+))?")


def parse(log_path):
    rows, stage, per_stage_epoch = [], 0, {}
    eval_metrics, in_eval, eval_tokens = None, False, []
    with open(log_path, errors="ignore") as f:
        for line in f:
            if "[stage1]" in line:
                stage = 1
            elif "[stage2]" in line:
                stage = 2
            if "Evaluation on" in line:
                in_eval = True
                continue
            if in_eval:
                # R prints a named numeric vector: a names row then a values row.
                eval_tokens += line.split()
                floats = [t for t in eval_tokens if re.fullmatch(r"[\d.]+(?:[eE][+-]?\d+)?", t)]
                names = [t for t in eval_tokens if t in ("loss", "Precision", "Recall", "F1Score")]
                if len(floats) >= 4 and len(names) >= 4:
                    eval_metrics = dict(loss=float(floats[0]), precision=float(floats[1]),
                                        recall=float(floats[2]), f1=float(floats[3]))
                    in_eval = False
                continue
            if "val_F1Score:" not in line:
                continue
            m = EPOCH.search(line)
            if not m:
                continue
            g = m.groups()
            st = stage or 1
            per_stage_epoch[st] = per_stage_epoch.get(st, 0) + 1
            rows.append(dict(
                stage=st, epoch=per_stage_epoch[st],
                loss=float(g[0]), precision=float(g[1]), recall=float(g[2]), f1=float(g[3]),
                val_loss=float(g[4]), val_precision=float(g[5]), val_recall=float(g[6]),
                val_f1=float(g[7]), lr=float(g[8]) if g[8] else ""))
    return rows, eval_metrics


def main():
    log_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows, eval_metrics = parse(log_path)
    cols = ["stage", "epoch", "loss", "precision", "recall", "f1", "val_loss",
            "val_precision", "val_recall", "val_f1", "lr"]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # Best (max val_f1) preferring the last stage present; final = held-out TestDS eval.
    last_stage = max((r["stage"] for r in rows), default=1)
    stage_rows = [r for r in rows if r["stage"] == last_stage] or rows
    best = max(stage_rows, key=lambda r: r["val_f1"]) if stage_rows else None
    final = {
        "epochs_total": len(rows),
        "stages": sorted({r["stage"] for r in rows}),
        "best_val": ({k: best[k] for k in ("stage", "epoch", "val_loss", "val_precision",
                                           "val_recall", "val_f1")} if best else None),
        "test": eval_metrics,
        "params": 31_030_593,   # standard 256x256 vanilla U-Net (model_UNet.R)
    }
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"parsed {len(rows)} epochs across stages {final['stages']}; "
          f"best val_F1 {best['val_f1']:.4f} (stage {best['stage']} epoch {best['epoch']})"
          if best else "no epochs parsed")
    print(f"TestDS eval: {eval_metrics}")


if __name__ == "__main__":
    main()

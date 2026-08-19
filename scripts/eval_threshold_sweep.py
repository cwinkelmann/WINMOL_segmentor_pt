"""Precision–recall across every threshold, and the AP that summarises it.

    python scripts/eval_threshold_sweep.py --model runs/x/model.pt --data-dir <DS>/test \
        --arch hrnet --out sweep.json

F1 at 0.5 is one point on a curve, and it is not usually the best one. On an unseen site
the models here keep precision and lose recall — which is exactly the situation a lower
threshold fixes, if the ranking is sound. AP says whether it is: it integrates precision
over recall and so measures how well the model *orders* pixels, independent of where the
cut is placed.

Probabilities are accumulated into a histogram rather than stored, so the curve comes from
every pixel of every tile (900 tiles is 236M pixels) at fixed memory. With `--bins 2000`
the threshold grid is finer than any decision you would act on.

Reported:

  * **AP** — area under the precision–recall curve, threshold-free.
  * **best F1** and the threshold that achieves it.
  * **F1 at 0.5**, so the cost of the default is visible.
  * the full curve, for plotting.
"""
import argparse
import json
import os
import sys

import numpy as np


def _ap_from_curve(precision, recall):
    """Step-wise integral of precision over recall: sum (R_i - R_{i-1}) * P_i.

    It must be walked in order of INCREASING recall. The arrays in `sweep` are indexed by
    ascending threshold, so recall runs the other way; integrating them in index order
    gives a negative "AP", which is how this bug announced itself.
    """
    order = np.argsort(recall)
    r_sorted, p_sorted = np.asarray(recall)[order], np.asarray(precision)[order]
    dr = np.diff(np.concatenate([[0.0], r_sorted]))
    return float(np.sum(dr * p_sorted))


def sweep(model_path, data_dir, arch="hrnet", bins=2000, limit=None, device="cpu",
          quiet=False):
    import glob

    import torch
    from PIL import Image

    from training.model_factory import build_model

    model = build_model(arch, encoder_weights=None)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state.state_dict() if hasattr(state, "state_dict") else state)
    model = model.eval().to(device)

    imgs = sorted(glob.glob(os.path.join(data_dir, "train", "*.jpeg")),
                  key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    if limit:
        imgs = imgs[:limit]

    # counts of predicted probability, split by ground truth. Everything the curve needs
    # is derivable from these two histograms, at fixed memory.
    pos = np.zeros(bins, dtype=np.int64)
    neg = np.zeros(bins, dtype=np.int64)
    for p in imgs:
        n = "".join(c for c in os.path.basename(p) if c.isdigit())
        x = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(x).permute(2, 0, 1)[None].to(device)))
        prob = prob[0, 0].cpu().numpy().ravel()
        gt = (np.asarray(Image.open(os.path.join(data_dir, "mask", f"mask{n}.gif"))
                         .convert("L")).ravel() > 127)
        idx = np.clip((prob * bins).astype(np.int64), 0, bins - 1)
        pos += np.bincount(idx[gt], minlength=bins)
        neg += np.bincount(idx[~gt], minlength=bins)

    # threshold t keeps every bin at or above it; sweep from high to low
    tp = np.cumsum(pos[::-1])[::-1].astype(float)
    fp = np.cumsum(neg[::-1])[::-1].astype(float)
    total_pos = float(pos.sum())
    thresholds = np.arange(bins) / bins
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 1.0)
        recall = tp / max(total_pos, 1e-9)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-9), 0.0)

    ap = _ap_from_curve(precision, recall)

    best = int(np.argmax(f1))
    half = int(0.5 * bins)
    out = {
        "model": model_path, "data_dir": data_dir, "arch": arch, "tiles": len(imgs),
        "ap": round(ap, 4),
        "best": {"threshold": round(float(thresholds[best]), 4),
                 "f1": round(float(f1[best]), 4),
                 "precision": round(float(precision[best]), 4),
                 "recall": round(float(recall[best]), 4)},
        "at_0.5": {"f1": round(float(f1[half]), 4),
                   "precision": round(float(precision[half]), 4),
                   "recall": round(float(recall[half]), 4)},
        "curve": [{"t": round(float(thresholds[i]), 3),
                   "p": round(float(precision[i]), 4),
                   "r": round(float(recall[i]), 4),
                   "f1": round(float(f1[i]), 4)}
                  for i in range(0, bins, max(1, bins // 100))],
    }
    if not quiet:
        name = os.path.basename(os.path.dirname(model_path))
        print(f"{name}  ({out['tiles']} tiles)")
        print(f"  AP {out['ap']:.4f}")
        print(f"  at 0.5      F1 {out['at_0.5']['f1']:.4f}  "
              f"P {out['at_0.5']['precision']:.4f}  R {out['at_0.5']['recall']:.4f}")
        print(f"  best t={out['best']['threshold']:.3f}  F1 {out['best']['f1']:.4f}  "
              f"P {out['best']['precision']:.4f}  R {out['best']['recall']:.4f}"
              f"   (+{100*(out['best']['f1']-out['at_0.5']['f1']):.2f} pts)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--arch", default="hrnet")
    p.add_argument("--bins", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = sweep(a.model, a.data_dir, a.arch, a.bins, a.limit, a.device)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

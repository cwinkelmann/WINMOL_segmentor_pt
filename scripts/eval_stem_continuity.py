"""Measure how *vectorizable* a stem prediction is, not just how accurate it is.

    python scripts/eval_stem_continuity.py --model runs/x/model.pt --data-dir <DS>/test \
        --out continuity.json [--arch hrnet] [--limit 200]

F1 answers "are the right pixels lit". It says nothing about whether those pixels form
one trunk or fourteen crumbs, and that difference is the whole job of the vectorizer
downstream: it has to decide which fragments belong to the same stem, then join them.

So this reports, alongside F1:

  * **components per tile** — connected components in the thresholded prediction. A stem
    broken by occlusion arrives as several; fewer is better.
  * **skeleton endpoints per tile** — a single unbroken stem contributes 2. Three
    fragments of one stem contribute 6. This counts breaks more directly than component
    count does, because it is unaffected by two stems touching.
  * **largest-component share** — how much of the predicted stem area sits in the single
    biggest piece. Rises as predictions become continuous.
  * **mean component area**.

These are comparable across models trained on *different targets* (modal vs amodal),
which raw F1 is not: a modal-trained and an amodal-trained model are each scored against
their own labels and the two numbers do not mean the same thing. Continuity is measured
on the prediction alone, so both arms are judged by the same ruler.

The same statistics are reported for the reference masks in `--data-dir`, which is the
honest ceiling: a model cannot be more continuous than the labels it learned from.
"""
import argparse
import json
import os
import sys

import numpy as np


def _skeleton_endpoints(mask):
    """Count skeleton pixels with exactly one neighbour — i.e. the ends of a stroke."""
    from scipy.ndimage import convolve
    from skimage.morphology import skeletonize

    if not mask.any():
        return 0
    sk = skeletonize(mask).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neighbours = convolve(sk, kernel, mode="constant")
    return int(((sk == 1) & (neighbours == 1)).sum())


def continuity(mask, min_px=64):
    """Per-tile continuity statistics for one binary mask.

    `component_len_px` — the mean major-axis length of the connected components — is the
    metric to read. Endpoint counting turned out to be dominated by skeleton spurs off a
    ragged traced outline (the reference masks score ~33 endpoints per tile against a
    prediction's ~6, which is an artefact of outline roughness, not of breaks). Component
    length measures the thing directly: a stem bridged across its occlusion gaps yields
    one long component instead of several short ones.

    Specks below `min_px` are ignored — they are noise, and counting them swamps the
    component statistics of whichever model happens to be less confident.
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops

    lab, n = label(mask)
    if n == 0:
        return {"components": 0, "endpoints": 0, "largest_share": 0.0,
                "mean_area_px": 0.0, "component_len_px": 0.0, "longest_len_px": 0.0}
    props = [r for r in regionprops(lab) if r.area >= min_px]
    if not props:
        return {"components": 0, "endpoints": 0, "largest_share": 0.0,
                "mean_area_px": 0.0, "component_len_px": 0.0, "longest_len_px": 0.0}
    sizes = np.array([r.area for r in props], dtype=float)
    lens = np.array([r.axis_major_length for r in props], dtype=float)
    return {"components": len(props),
            "endpoints": _skeleton_endpoints(mask),
            "largest_share": float(sizes.max() / sizes.sum()),
            "mean_area_px": float(sizes.mean()),
            "component_len_px": float(lens.mean()),
            "longest_len_px": float(lens.max())}


def _load_model(path, arch, device):
    import torch

    from training.model_factory import build_model

    model = build_model(arch, encoder_weights=None)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.state_dict() if hasattr(state, "state_dict") else state)
    return model.eval().to(device)


def evaluate(model_path, data_dir, arch="hrnet", threshold=0.5, limit=None, device="cpu",
             quiet=False):
    import glob

    import torch
    from PIL import Image

    model = _load_model(model_path, arch, device)
    imgs = sorted(glob.glob(os.path.join(data_dir, "train", "*.jpeg")),
                  key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    if limit:
        imgs = imgs[:limit]

    keys = ("components", "endpoints", "largest_share", "mean_area_px",
            "component_len_px", "longest_len_px")
    agg = {k: [] for k in keys}
    ref = {k: [] for k in agg}
    tp = fp = fn = 0
    for p in imgs:
        n = "".join(c for c in os.path.basename(p) if c.isdigit())
        x = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        with torch.no_grad():
            logits = model(torch.from_numpy(x).permute(2, 0, 1)[None].to(device))
            pred = torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold
        gt = np.asarray(Image.open(os.path.join(data_dir, "mask", f"mask{n}.gif"))
                        .convert("L")) > 127

        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
        for d, m in ((agg, pred), (ref, gt)):
            for k, v in continuity(m).items():
                d[k].append(v)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"model": model_path, "data_dir": data_dir, "arch": arch, "tiles": len(imgs),
           "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
           "precision": round(prec, 4), "recall": round(rec, 4),
           "prediction": {k: round(float(np.mean(v)), 3) for k, v in agg.items()},
           "reference": {k: round(float(np.mean(v)), 3) for k, v in ref.items()}}
    if not quiet:
        print(f"{os.path.basename(os.path.dirname(model_path))}  ({out['tiles']} tiles)")
        print(f"  F1 {out['f1']:.4f}  P {out['precision']:.4f}  R {out['recall']:.4f}")
        for label, d in (("prediction", out["prediction"]), ("labels   ", out["reference"])):
            print(f"  {label}: {d['components']:.2f} components, mean length "
                  f"{d['component_len_px']:.0f} px, longest {d['longest_len_px']:.0f} px, "
                  f"largest {100*d['largest_share']:.0f}% of area")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="trained .pt state_dict")
    p.add_argument("--data-dir", required=True, help="tile dir with train/ and mask/")
    p.add_argument("--arch", default="hrnet")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = evaluate(a.model, a.data_dir, a.arch, a.threshold, a.limit, a.device)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

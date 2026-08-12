"""Rank test tiles by model/annotation disagreement, and say what kind of disagreement it is.

"Which tiles does the model get wrong" is only half the question. The useful half is
whether the wrongness is the model's or the label's, and that is decidable from geometry:

* **boundary** — FP/FN pixels within `--boundary-px` of a ground-truth edge. Every traced
  polygon has a few pixels of slop; a model cannot be more precise than the annotation.
  Large boundary fractions argue for *label smoothing* (soft targets at edges), because
  the network is being punished for disagreeing with a line nobody drew exactly.
* **hole** — FN pixels inside a background component fully enclosed by stem. These are
  gaps in the annotation, not gaps in the tree. Large hole counts argue for
  *morphological closing* of the labels.
* **body** — everything else: whole stems missed, or stems invented on bare ground. This
  is genuine model error and no label surgery fixes it.

Only the first two are worth changing labels for. Reporting them separately is the point.

    python scripts/disagreement.py --model runs/x/model.onnx --data-dir DS/test \\
        --out docs/assets/disagreement.png --top 8
"""
import argparse
import json
import os

import numpy as np


def _read_mask(path):
    """Masks are palette GIFs: np.asarray gives indices (0/1), not values (0/255)."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("L")) > 127


def decompose(pred, gt, boundary_px=2):
    """Split FP/FN into boundary slop, annotation holes, and real body error."""
    import scipy.ndimage as ndi

    fp, fn = pred & ~gt, ~pred & gt
    # distance from every pixel to the nearest GT edge, inside and out
    edge = gt ^ ndi.binary_erosion(gt, iterations=1, border_value=0)
    near = ndi.distance_transform_edt(~edge) <= boundary_px

    # holes: background components enclosed by stem, i.e. filled-minus-original
    holes = ndi.binary_fill_holes(gt) & ~gt

    d = {
        "tp": int((pred & gt).sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
        "fp_boundary": int((fp & near).sum()), "fn_boundary": int((fn & near).sum()),
        "fn_hole": int((fn & holes).sum()), "fp_hole": int((fp & holes).sum()),
        "gt_holes_px": int(holes.sum()),
    }
    d["fp_body"] = d["fp"] - d["fp_boundary"]
    d["fn_body"] = d["fn"] - d["fn_boundary"] - max(0, d["fn_hole"] - 0)
    d["fn_body"] = max(0, d["fn_body"])
    denom = 2 * d["tp"] + d["fp"] + d["fn"]
    d["f1"] = (2 * d["tp"] / denom) if denom else 1.0
    return d, fp, fn, holes


def run(model, data_dir, out_png=None, top=8, threshold=0.5, boundary_px=2, limit=None,
        json_out=None):
    import onnxruntime as ort
    from PIL import Image

    img_dir = os.path.join(data_dir, "train")
    msk_dir = os.path.join(data_dir, "mask")
    ids = sorted(int(f[5:-5]) for f in os.listdir(img_dir) if f.endswith(".jpeg"))
    if limit:
        ids = ids[:limit]

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    rows = []
    for n in ids:
        img = np.asarray(Image.open(os.path.join(img_dir, f"train{n}.jpeg")).convert("RGB"))
        gt = _read_mask(os.path.join(msk_dir, f"mask{n}.gif"))
        x = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        prob = sess.run(None, {inp: x})[0][0, 0]
        d, *_ = decompose(prob > threshold, gt, boundary_px)
        d["tile"] = n
        rows.append(d)

    rows.sort(key=lambda r: r["f1"])
    tot = {k: sum(r[k] for r in rows) for k in
           ("tp", "fp", "fn", "fp_boundary", "fn_boundary", "fn_hole", "gt_holes_px")}
    summary = {
        "tiles": len(rows),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "fp_boundary_share": tot["fp_boundary"] / max(tot["fp"], 1),
        "fn_boundary_share": tot["fn_boundary"] / max(tot["fn"], 1),
        "fn_hole_share": tot["fn_hole"] / max(tot["fn"], 1),
        "gt_hole_px_share": tot["gt_holes_px"] / max(tot["tp"] + tot["fn"], 1),
        "totals": tot,
    }

    print(f"{len(rows)} tiles, mean F1 {summary['mean_f1']:.4f}")
    print(f"  FP within {boundary_px}px of a GT edge : {100*summary['fp_boundary_share']:5.1f}%")
    print(f"  FN within {boundary_px}px of a GT edge : {100*summary['fn_boundary_share']:5.1f}%")
    print(f"  FN inside annotation holes           : {100*summary['fn_hole_share']:5.1f}%")
    print(f"  annotation hole px / stem px         : {100*summary['gt_hole_px_share']:5.2f}%")
    print(f"\nworst {min(top, len(rows))} tiles by F1:")
    for r in rows[:top]:
        print(f"  tile {r['tile']:5d}  F1 {r['f1']:.3f}  fp {r['fp']:6d} "
              f"(edge {100*r['fp_boundary']/max(r['fp'],1):4.0f}%)  fn {r['fn']:6d} "
              f"(edge {100*r['fn_boundary']/max(r['fn'],1):4.0f}%)")

    if json_out:
        with open(json_out, "w") as f:
            json.dump({"summary": summary, "tiles": rows}, f, indent=1)

    if out_png:
        _render(model, data_dir, rows[:top], out_png, threshold, boundary_px)
    return summary, rows


def _render(model, data_dir, worst, out_png, threshold, boundary_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cols = len(worst)
    fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5.4), squeeze=False)

    for j, r in enumerate(worst):
        n = r["tile"]
        img = np.asarray(Image.open(
            os.path.join(data_dir, "train", f"train{n}.jpeg")).convert("RGB"))
        gt = _read_mask(os.path.join(data_dir, "mask", f"mask{n}.gif"))
        x = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        pred = sess.run(None, {inp: x})[0][0, 0] > threshold
        d, fp, fn, holes = decompose(pred, gt, boundary_px)

        axes[0][j].imshow(img)
        axes[0][j].contour(gt, [0.5], colors="#00E5FF", linewidths=.9)
        axes[0][j].contour(pred, [0.5], colors="#FFD400", linewidths=.9)
        axes[0][j].set_title(f"tile {n} — F1 {d['f1']:.2f}", fontsize=8)

        # red = predicted, not annotated; blue = annotated, not predicted
        overlay = np.zeros((*gt.shape, 3), np.float32)
        overlay[..., 0] = fp
        overlay[..., 2] = fn
        overlay[..., 1] = holes * 0.8
        axes[1][j].imshow(img.astype(np.float32) / 255 * 0.35 + overlay * 0.85)
        axes[1][j].set_title(f"FP {d['fp']} / FN {d['fn']}", fontsize=7)
        for ax in (axes[0][j], axes[1][j]):
            ax.axis("off")

    fig.suptitle("worst-agreement tiles — cyan: annotation, yellow: prediction | "
                 "red: false positive, blue: false negative, green: annotation hole",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--boundary-px", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    run(a.model, a.data_dir, a.out, a.top, a.threshold, a.boundary_px, a.limit, a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

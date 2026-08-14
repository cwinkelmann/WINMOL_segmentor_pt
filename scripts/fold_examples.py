"""Render tiles with ground truth and prediction as outlines, for a fold's test set.

Two rules from `docs/process.md` are baked in rather than left to the caller:

**Random by default, with the seed printed.** A hand-picked panel misrepresents; if you
want the worst cases pass `--worst`, and the figure is then labelled as selected so it
cannot be mistaken for a sample.

**Outlines, not fills.** The question these figures answer is usually "is the label
right", and a filled mask hides the evidence underneath it. Ground truth and prediction
get separate colours, so a miss reads as a bare GT outline and a false positive as a bare
prediction outline.

    python scripts/fold_examples.py --data-dir DS/folds/Bachsee_north/test \\
        --models fixed=a.onnx jitter=b.onnx --n 6 --out fig.png
"""
import argparse
import os
import re

import numpy as np


def _outline(mask, width=2):
    """Boundary of a binary mask, as a boolean array."""
    from scipy import ndimage
    m = mask.astype(bool)
    er = ndimage.binary_erosion(m, iterations=width, border_value=0)
    return m & ~er


def _predict(sess, arr):
    inp = sess.get_inputs()[0]
    nhwc = len(inp.shape) == 4 and inp.shape[-1] in (3, "3")
    x = arr[None].astype(np.float32)
    if not nhwc:
        x = np.transpose(x, (0, 3, 1, 2))
    y = np.squeeze(sess.run(None, {inp.name: x})[0])
    return y


def pick(data_dir, n, seed, worst=None, size=512):
    """Random tile ids, or the n worst by F1 against `worst` (a session) if given."""
    img = os.path.join(data_dir, "train")
    pat = re.compile(r"^train(\d+)\.jpe?g$")
    ids = sorted(int(m.group(1)) for m in (pat.match(f) for f in os.listdir(img)) if m)
    rng = np.random.default_rng(seed)
    if worst is None:
        return sorted(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist()), False
    scored = []
    for i in ids:
        im, mk = _load(data_dir, i, size)
        p = _predict(worst, im) > 0.5
        g = mk > 0.5
        tp = (p & g).sum()
        f1 = 2 * tp / max(p.sum() + g.sum(), 1)
        scored.append((f1, i))
    return [i for _, i in sorted(scored)[:n]], True


def _load(data_dir, n, size=512):
    from PIL import Image
    im = Image.open(os.path.join(data_dir, "train", f"train{n}.jpeg")).convert("RGB")
    mk = Image.open(os.path.join(data_dir, "mask", f"mask{n}.gif")).convert("L")
    w, _ = im.size
    off = (w - size) // 2                      # centre 512 window = the 1.00x view
    box = (off, off, off + size, off + size)
    return (np.asarray(im.crop(box), np.float32) / 255.0,
            np.asarray(mk.crop(box), np.float32) / 255.0)


def figure(data_dir, models, ids, out, selected=False, title=None, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import onnxruntime as ort

    sessions = {k: ort.InferenceSession(v, providers=["CPUExecutionProvider"])
                for k, v in models.items()}
    ncol = 1 + len(sessions)
    fig, axes = plt.subplots(len(ids), ncol, figsize=(3.1 * ncol, 3.1 * len(ids)),
                             squeeze=False)
    for r, n in enumerate(ids):
        im, mk = _load(data_dir, n)
        gt = _outline(mk > 0.5)
        axes[r][0].imshow(im)
        base = im.copy()
        base[gt] = (0.1, 0.9, 0.2)             # ground truth, green
        axes[r][0].imshow(base)
        axes[r][0].set_ylabel(f"tile {n}", fontsize=8)
        if r == 0:
            axes[r][0].set_title("image + ground truth", fontsize=9)
        for c, (name, sess) in enumerate(sessions.items(), start=1):
            pr = _predict(sess, im) > 0.5
            ov = im.copy()
            ov[_outline(pr)] = (1.0, 0.2, 0.1)  # prediction, red
            ov[gt] = (0.1, 0.9, 0.2)
            axes[r][c].imshow(ov)
            inter = (pr & (mk > 0.5)).sum()
            f1 = 2 * inter / max(pr.sum() + (mk > 0.5).sum(), 1)
            axes[r][c].set_xlabel(f"F1 {f1:.3f}", fontsize=8)
            if r == 0:
                axes[r][c].set_title(name, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    handles = [mpatches.Patch(color=(0.1, 0.9, 0.2), label="ground truth"),
               mpatches.Patch(color=(1.0, 0.2, 0.1), label="prediction")]
    tag = f"SELECTED: {len(ids)} worst by F1" if selected else \
          f"random sample of {len(ids)}, seed {seed}"
    fig.suptitle(f"{title or os.path.basename(data_dir)} — {tag}", fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--models", nargs="+", default=[], help="name=path.onnx")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worst", default=None,
                   help="onnx model to rank by; makes this a SELECTED figure")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    models = dict(m.split("=", 1) for m in a.models)
    sess = None
    if a.worst:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.worst, providers=["CPUExecutionProvider"])
    ids, selected = pick(a.data_dir, a.n, a.seed, sess)
    print(f"tiles: {ids}  ({'SELECTED worst' if selected else f'random, seed {a.seed}'})")
    print(figure(a.data_dir, models, ids, a.out, selected, a.title, a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

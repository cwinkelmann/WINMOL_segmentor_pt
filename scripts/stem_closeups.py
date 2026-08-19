"""Crop tightly around individual labelled stems: what is the model actually asked to find?

Tile-level panels answer "does the prediction match the label". They do not answer "is
there anything here to see", which is the question when a site scores near zero. This
crops to a single stem at a time and shows it twice — bare, then outlined — so the label
cannot do the seeing for you.

    python scripts/stem_closeups.py --sites Bachsee=DS/Bachsee Kaufland=DS/Kaufland \\
        --n 5 --out fig.png
"""
import argparse
import os
import re

import numpy as np


def _ids(d):
    pat = re.compile(r"^train(\d+)\.jpe?g$")
    return sorted(int(m.group(1)) for m in
                  (pat.match(f) for f in os.listdir(os.path.join(d, "train"))) if m)


def _biggest_component(m):
    """Row/col bounds of the largest labelled blob, so the crop lands on one stem."""
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
    return sl, float(sizes.max())


def closeups(site_dir, n, seed, half=90, min_px=400):
    """Return (image, mask) crops centred on single stems, chosen at random."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    ids = _ids(site_dir)
    rng.shuffle(ids)
    out = []
    for i in ids:
        im = Image.open(os.path.join(site_dir, "train", f"train{i}.jpeg")).convert("RGB")
        mk = Image.open(os.path.join(site_dir, "mask", f"mask{i}.gif")).convert("L")
        a = np.asarray(im, np.float32) / 255.0
        m = np.asarray(mk) > 127
        got = _biggest_component(m)
        if got is None or got[1] < min_px:
            continue
        sl, _ = got
        r = (sl[0].start + sl[0].stop) // 2
        c = (sl[1].start + sl[1].stop) // 2
        r = int(np.clip(r, half, a.shape[0] - half))
        c = int(np.clip(c, half, a.shape[1] - half))
        out.append((i, a[r - half:r + half, c - half:c + half],
                    m[r - half:r + half, c - half:c + half]))
        if len(out) >= n:
            break
    return out


def figure(sites, n, seed, out, half=90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    rows = len(sites) * 2
    fig, axes = plt.subplots(rows, n, figsize=(2.15 * n, 2.3 * rows), squeeze=False)
    for si, (name, d) in enumerate(sites.items()):
        crops = closeups(d, n, seed, half)
        for c, (tid, img, m) in enumerate(crops):
            lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
            sep = ((lum[m].mean() - lum[~m].mean()) / (lum.std() + 1e-9)
                   if m.any() and (~m).any() else float("nan"))
            axes[2 * si][c].imshow(img)
            axes[2 * si][c].set_title(f"tile {tid}   sep {sep:+.2f}", fontsize=8)
            ov = img.copy()
            edge = m & ~ndimage.binary_erosion(m, iterations=2, border_value=0)
            ov[edge] = (0.1, 1.0, 0.2)
            axes[2 * si + 1][c].imshow(ov)
        axes[2 * si][0].set_ylabel(f"{name}\nbare", fontsize=9)
        axes[2 * si + 1][0].set_ylabel("labelled", fontsize=9)
        for c in range(len(crops), n):
            axes[2 * si][c].axis("off"); axes[2 * si + 1][c].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"One labelled stem per panel — top row bare, bottom row outlined "
                 f"(random, seed {seed})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=140)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", nargs="+", required=True, help="name=dir")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--half", type=int, default=90, help="crop half-width in tile px")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print(figure(dict(s.split("=", 1) for s in a.sites), a.n, a.seed, a.out, a.half))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

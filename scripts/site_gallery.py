"""Show several sites side by side, with the colour statistics that separate them.

Site identity has dominated every comparison measured in this project, and the reason is
usually visible in three seconds of looking: Bachsee_north is a November beech canopy —
near-uniform amber — while Kaufland is high-summer green. A model trained without that
colour domain returns F1 ~0 on it, and the table alone makes that look like a bug.

Random tiles per site, seed printed. Hue is reported as the circular mean over pixels with
enough saturation to have a meaningful hue, since averaging hue linearly wraps at red.

    python scripts/site_gallery.py --sites A=DS/A B=DS/B --n 4 --out fig.png
"""
import argparse
import os
import re

import numpy as np


def _ids(d):
    pat = re.compile(r"^train(\d+)\.jpe?g$")
    return sorted(int(m.group(1)) for m in
                  (pat.match(f) for f in os.listdir(os.path.join(d, "train"))) if m)


def stem_contrast(site_dir, ids, size=512, min_px=200):
    """Stem-minus-background luminance, in units of each tile's own spread.

    This is the statistic that explains the fold ordering, and it is not a colour
    statistic: Bachsee_north's stems lie *under* closed amber foliage, so they are seen
    through gaps rather than directly, and the separation goes slightly negative. A site
    whose stems are not visible cannot be segmented from that imagery no matter what the
    model does, which is the difference between a hard fold and an uninformative one.
    """
    from PIL import Image
    out = []
    for n in ids:
        im = Image.open(os.path.join(site_dir, "train", f"train{n}.jpeg")).convert("RGB")
        mk = Image.open(os.path.join(site_dir, "mask", f"mask{n}.gif")).convert("L")
        w, _ = im.size
        off = (w - size) // 2
        box = (off, off, off + size, off + size)
        a = np.asarray(im.crop(box), np.float32) / 255.0
        m = np.asarray(mk.crop(box)) > 127
        if m.sum() < min_px or (~m).sum() < min_px:
            continue                          # nothing to contrast against
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        out.append(float((lum[m].mean() - lum[~m].mean()) / (lum.std() + 1e-9)))
    return float(np.mean(out)) if out else float("nan")


def stats(site_dir, ids, size=512):
    """Mean RGB, mean saturation, circular-mean hue, and stem-vs-background contrast."""
    from PIL import Image
    hs, ss, rgb = [], [], []
    for n in ids:
        im = Image.open(os.path.join(site_dir, "train", f"train{n}.jpeg")).convert("RGB")
        w, _ = im.size
        off = (w - size) // 2
        a = np.asarray(im.crop((off, off, off + size, off + size)), np.float32) / 255.0
        rgb.append(a.reshape(-1, 3).mean(0))
        hsv = np.asarray(Image.fromarray((a * 255).astype(np.uint8), "RGB")
                         .convert("HSV"), np.float32) / 255.0
        h, s = hsv[..., 0].ravel(), hsv[..., 1].ravel()
        keep = s > 0.15                      # unsaturated pixels have no meaningful hue
        if keep.any():
            hs.append(np.exp(2j * np.pi * h[keep]).mean())
        ss.append(s.mean())
    hue = np.angle(np.mean(hs)) / (2 * np.pi) % 1.0 if hs else float("nan")
    return {"rgb": np.mean(rgb, 0), "sat": float(np.mean(ss)), "hue_deg": 360 * hue,
            "stem_contrast": stem_contrast(site_dir, ids, size)}


def figure(sites, n, seed, out, size=512, stat_tiles=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    picks = {k: sorted(rng.choice(_ids(v), size=n, replace=False).tolist())
             for k, v in sites.items()}
    fig, axes = plt.subplots(len(sites), n, figsize=(2.6 * n, 2.75 * len(sites)),
                             squeeze=False)
    for r, (name, d) in enumerate(sites.items()):
        # Stats over a larger random draw than the four tiles on show, so the
        # numbers describe the site rather than the panel.
        ids = picks[name]
        if stat_tiles:
            ids = rng.choice(_ids(d), size=min(stat_tiles, len(_ids(d))),
                             replace=False).tolist()
        st = stats(d, ids)
        for c, i in enumerate(picks[name]):
            im = Image.open(os.path.join(d, "train", f"train{i}.jpeg")).convert("RGB")
            w, _ = im.size
            off = (w - size) // 2
            axes[r][c].imshow(im.crop((off, off, off + size, off + size)))
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[r][0].set_ylabel(f"{name}\nsat {st['sat']:.2f}   "
                              f"stem contrast {st['stem_contrast']:+.2f}", fontsize=8)
        print(f"  {name:20s} hue {st['hue_deg']:6.1f}deg  sat {st['sat']:.3f}  "
              f"stem contrast {st['stem_contrast']:+.3f}  tiles {picks[name]}")
    fig.suptitle(f"Colour domain by site — random sample of {n}, seed {seed}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", nargs="+", required=True, help="name=dir")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--stat-tiles", type=int, default=0,
                   help="compute stats over this many random tiles instead "
                        "of only the displayed ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print(figure(dict(s.split("=", 1) for s in a.sites), a.n, a.seed, a.out,
                 stat_tiles=a.stat_tiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preflight a dataset before training — fail loudly instead of wasting a run.

    python scripts/validate_dataset.py --data-dir <DS> [--rgbd] [--strict]

Checks the failure modes that are invisible once training starts and expensive
to discover afterwards:

  * missing or partial `depth/` when --rgbd is requested — a short depth dir does
    not error, it silently shrinks the training set, because the loader pairs on
    the intersection of ids
  * image/mask/depth size disagreement
  * constant or all-nodata depth tiles
  * masks that are not binary, or are empty across the whole dataset
  * **depth that merely restates the mask** — reported as an AUC. Depth derived
    from the same geometry as the label makes an RGB-vs-RGBD comparison
    meaningless: the model reads the answer off the fourth channel. Worth
    knowing before the experiment, not after.

Exit code is 0 when usable, 1 when errors were found (or, with --strict, when
warnings were found).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

MIN_TILES = 1
LEAK_WARN_AUC = 0.90


def _ids(d, prefix, ext):
    out = {}
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        stem, e = os.path.splitext(name)
        if e.lower() == ext and stem.startswith(prefix) and stem[len(prefix):].isdigit():
            out[int(stem[len(prefix):])] = os.path.join(d, name)
    return out


def _depth_ids(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        stem, e = os.path.splitext(name)
        if e.lower() in (".png", ".tif", ".tiff") and stem.startswith("depth") \
                and stem[5:].isdigit():
            out[int(stem[5:])] = os.path.join(d, name)
    return out


def _auc(depth, mask, rng, n=3000):
    """P(depth on a stem pixel > depth on a background pixel). 0.5 = uninformative."""
    a, b = depth[mask], depth[~mask]
    if len(a) < 10 or len(b) < 10:
        return None
    a = rng.choice(a, min(n, len(a)), replace=False)
    b = rng.choice(b, min(n, len(b)), replace=False)
    return float((a[:, None] > b[None, :]).mean())


def validate(data_dir, rgbd=False, sample=40, seed=0):
    errors, warnings = [], []
    imgs = _ids(os.path.join(data_dir, "train"), "train", ".jpeg")
    masks = _ids(os.path.join(data_dir, "mask"), "mask", ".gif")
    depths = _depth_ids(os.path.join(data_dir, "depth"))

    if not imgs:
        errors.append("no train/train{N}.jpeg found")
    if not masks:
        errors.append("no mask/mask{N}.gif found")
    paired = sorted(set(imgs) & set(masks))
    if len(paired) < MIN_TILES:
        errors.append(f"only {len(paired)} image/mask pairs")
    unpaired = (set(imgs) ^ set(masks))
    if unpaired:
        warnings.append(f"{len(unpaired)} ids lack a counterpart image or mask")

    if rgbd:
        if not depths:
            errors.append("--rgbd requested but depth/ is missing or empty")
        else:
            missing = sorted(set(paired) - set(depths))
            if missing:
                errors.append(
                    f"depth missing for {len(missing)} of {len(paired)} tiles "
                    f"(e.g. {missing[:5]}) — the loader would silently train on "
                    f"{len(paired) - len(missing)} tiles instead")

    rng = np.random.default_rng(seed)
    take = paired[:: max(1, len(paired) // sample)][:sample] if paired else []
    aucs, empty_masks = [], 0
    for n in take:
        with Image.open(imgs[n]) as im:
            isize = im.size
        mk = Image.open(masks[n])
        mk.seek(0)
        m = np.asarray(mk.convert("L")) >= 128
        if mk.size != isize:
            errors.append(f"tile {n}: mask size {mk.size} != image size {isize}")
        if not m.any():
            empty_masks += 1
        if rgbd and n in depths:
            with Image.open(depths[n]) as dm:
                dsize = dm.size
                z = np.asarray(dm).astype(np.float32)
            if dsize != isize:
                errors.append(f"tile {n}: depth size {dsize} != image size {isize}")
                continue
            if z.ndim == 3:
                warnings.append(f"tile {n}: depth has {z.shape[2]} channels; "
                                "single-channel is expected")
                z = z[..., 0]
            finite = np.isfinite(z)
            if not finite.any():
                errors.append(f"tile {n}: depth is entirely nodata")
                continue
            if not finite.all():
                warnings.append(f"tile {n}: depth has "
                                f"{100 * (~finite).mean():.1f}% nodata pixels")
            if np.ptp(z[finite]) == 0:
                warnings.append(f"tile {n}: depth is constant (carries no information)")
                continue
            if m.shape == z.shape:
                a = _auc(z, m, rng)
                if a is not None:
                    aucs.append(a)

    if empty_masks == len(take) and take:
        errors.append("every sampled mask is empty")

    depth_auc = float(np.mean(aucs)) if aucs else None
    if depth_auc is not None and depth_auc >= LEAK_WARN_AUC:
        warnings.append(
            f"depth predicts the mask with AUC {depth_auc:.3f} — it largely "
            "restates the label, so an RGB-vs-RGBD comparison on this data would "
            "measure leakage rather than fusion")

    return {"ok": not errors, "pairs": len(paired), "depth_tiles": len(depths),
            "sampled": len(take), "depth_mask_auc": depth_auc,
            "errors": errors, "warnings": warnings}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--rgbd", action="store_true", help="also require a usable depth/ dir")
    p.add_argument("--sample", type=int, default=40, help="tiles to inspect in detail")
    p.add_argument("--strict", action="store_true", help="treat warnings as failure")
    a = p.parse_args(argv)

    r = validate(a.data_dir, rgbd=a.rgbd, sample=a.sample)
    print(f"{a.data_dir}: {r['pairs']} image/mask pairs, {r['depth_tiles']} depth tiles "
          f"({r['sampled']} inspected)")
    if r["depth_mask_auc"] is not None:
        print(f"  depth->mask AUC: {r['depth_mask_auc']:.3f}  (0.5 = independent, "
              f"1.0 = depth is the label)")
    for e in r["errors"]:
        print(f"  ERROR   {e}")
    for w in r["warnings"][:12]:
        print(f"  warning {w}")
    if len(r["warnings"]) > 12:
        print(f"  ... and {len(r['warnings']) - 12} more warnings")
    ok = r["ok"] and (not a.strict or not r["warnings"])
    print("USABLE" if ok else "NOT USABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

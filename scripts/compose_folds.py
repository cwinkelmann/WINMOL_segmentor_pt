"""Compose leave-one-site-out folds from per-site tile sets, by symlink.

Building a dataset per fold would re-cut the same ground four times and quadruple both
disk and the transfer to the training box. Instead each site is sampled **once**, and a
fold is assembled by linking those tiles into train/val/test under fresh, contiguous
indices — `StemDataset` pairs `train{N}.jpeg` with `mask{N}.gif`, and separate sampling
runs each start their numbering at 1, so renumbering on compose is what keeps two sites
from colliding.

Fold layout, given sites that can be block-split (`tv`, contributing train+val) and sites
too small to split (`all`, contributing wholesale):

    fold X:  test  = all/X                      (the held-out site, entire)
             train = tv/{other splittable}/train + all/{other whole sites}
             val   = tv/{other splittable}/val

The held-out site appears in exactly one split, so a fold cannot leak by construction —
there is no shared ground to leak across. Within the training sites, leak-freedom is the
block buffer's job and is verified separately from `tiles.jsonl`.

    python scripts/compose_folds.py --site-all DS/site_all --site-tv DS/site_tv \\
        --out DS/folds --folds Campus Campus_Oberheide Bachsee_north Kaufland
"""
import argparse
import json
import os
import re

PAT = re.compile(r"^train(\d+)\.(jpe?g|png)$")


def _ids(d):
    """Paired ids under a tile dir, numerically ordered.

    Rejects an image whose mask is missing rather than skipping it: a fold quietly built
    from half its tiles is the kind of failure that reads as a real result.
    """
    img = os.path.join(d, "train")
    if not os.path.isdir(img):
        return []
    out = []
    for f in sorted(os.listdir(img)):
        m = PAT.match(f)
        if not m:
            continue
        n, ext = int(m.group(1)), m.group(2)
        mk = os.path.join(d, "mask", f"mask{n}.gif")
        if not os.path.exists(mk):
            raise SystemExit(f"{img}/{f} has no mask{n}.gif — refusing a partial fold")
        out.append((n, ext))
    return sorted(out)


def _link(src_dir, dst_dir, start, provenance, source_label, copy=False):
    """Link one site's tiles into a split under contiguous indices from `start`."""
    import shutil

    os.makedirs(os.path.join(dst_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "mask"), exist_ok=True)
    n_out = start
    for n, ext in _ids(src_dir):
        for sub, name, out in (("train", f"train{n}.{ext}", f"train{n_out}.{ext}"),
                               ("mask", f"mask{n}.gif", f"mask{n_out}.gif")):
            s = os.path.abspath(os.path.join(src_dir, sub, name))
            d = os.path.join(dst_dir, sub, out)
            if os.path.lexists(d):
                os.remove(d)
            shutil.copy2(s, d) if copy else os.symlink(s, d)
        provenance.append({"index": n_out, "source": source_label, "source_index": n})
        n_out += 1
    return n_out


def compose(fold, sources, out_dir, copy=False):
    """`sources` maps split -> [(label, dir), ...]. Returns per-split tile counts."""
    counts, prov = {}, []
    for split in ("train", "val", "test"):
        n = 1
        for label, d in sources.get(split, []):
            n = _link(d, os.path.join(out_dir, split), n, prov, f"{label}:{split}", copy)
        counts[split] = n - 1
        if counts[split] == 0:
            raise SystemExit(f"fold {fold!r}: split {split!r} got no tiles from "
                             f"{[l for l, _ in sources.get(split, [])]} — check the caps")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fold.json"), "w") as f:
        json.dump({"fold": fold, "held_out_site": fold, "counts": counts,
                   "sources": {k: [[l, os.path.abspath(d)] for l, d in v]
                               for k, v in sources.items()}}, f, indent=1)
    with open(os.path.join(out_dir, "provenance.jsonl"), "w") as f:
        for r in prov:
            f.write(json.dumps(r) + "\n")
    return counts


def plan(folds, site_all, site_tv, splittable):
    """Which site directory feeds which split, for each fold."""
    out = {}
    for held in folds:
        train = [(s, os.path.join(site_tv, s, "train")) for s in splittable if s != held]
        train += [(s, os.path.join(site_all, s)) for s in folds
                  if s != held and s not in splittable]
        val = [(s, os.path.join(site_tv, s, "val")) for s in splittable if s != held]
        out[held] = {"train": train, "val": val,
                     "test": [(held, os.path.join(site_all, held))]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site-all", required=True, help="dir of <site>/ whole-site tiles")
    p.add_argument("--site-tv", required=True, help="dir of <site>/{train,val} block tiles")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", nargs="+", required=True, help="one fold per site")
    p.add_argument("--splittable", nargs="+", required=True,
                   help="sites large enough to block-split (contribute train+val)")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = p.parse_args(argv)

    unknown = [s for s in a.splittable if s not in a.folds]
    if unknown:
        raise SystemExit(f"--splittable names sites not in --folds: {unknown}")

    for held, sources in plan(a.folds, a.site_all, a.site_tv, a.splittable).items():
        counts = compose(held, sources, os.path.join(a.out, held), a.copy)
        print(f"  fold {held:20s} train {counts['train']:5d}  val {counts['val']:4d}"
              f"  test {counts['test']:5d}   (test = {held} only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

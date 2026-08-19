"""Draw where each split's tiles came from, one panel per orthomosaic.

A split is a claim about ground, and the cheapest way to check a claim about ground is to
look at it. This reads `tiles.jsonl` from each split directory — the provenance every
dataset carries — and plots every tile centre against its site's AOI.

    python scripts/plot_splits.py --data-dir ~/winmol_data/BeechAll \\
        --config configs/beech_blocks.json --out docs/assets/beech-splits.png

What to check in the output:

* **Colours must not interleave.** Blocks are dealt whole, so each colour should occupy
  distinct patches with a visible gap along every shared edge. Salt-and-pepper mixing
  means the split was drawn per tile, and validation is measuring training data.
* **Every site should show all three colours** under `--strategy blocks`. A site missing
  a colour contributed nothing to that split, which is how a validation set ends up
  representing one acquisition.
* **The printed closest approach must clear the threshold.** Two rotated footprints can
  share area below `extent * sqrt(2)`; above it they cannot touch.
"""
import argparse
import itertools
import json
import math
import os

SPLITS = ("train", "val", "test")
COLOURS = {"train": "#0F6E5C", "val": "#C2571E", "test": "#3E5C9A"}


def load(data_dir):
    """site -> split -> [(x, y), ...], plus the footprint size in metres."""
    by_site, extent = {}, None
    for split in SPLITS:
        path = os.path.join(data_dir, split, "tiles.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                extent = extent or r.get("extent_m")
                by_site.setdefault(r["site"], {}).setdefault(split, []).append(
                    (r["centre_x"], r["centre_y"]))
    return by_site, extent


def closest_between(a, b):
    """Smallest centre-to-centre distance between two sets of tiles."""
    if not a or not b:
        return float("inf")
    return min(math.dist(p, q) for p, q in itertools.product(a, b))


def plot(data_dir, config_path=None, out_path=None, dpi=140):
    # NB: no matplotlib.use() here — a library function must not hijack the caller's
    # backend, or a notebook importing it silently stops rendering figures inline.
    # The CLI selects Agg for itself in main().
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_site, extent = load(data_dir)
    if not by_site:
        raise SystemExit(f"no tiles.jsonl under {data_dir}; nothing to draw")
    threshold = extent * math.sqrt(2)

    aois = {}
    if config_path:
        import fiona
        from shapely.geometry import shape
        from shapely.ops import unary_union
        for site in json.load(open(config_path))["sites"]:
            with fiona.open(site["aoi"]) as c:
                aois[site["name"]] = unary_union(
                    [shape(f["geometry"]).buffer(0) for f in c])

    names = sorted(by_site)
    cols = min(len(names), 4)
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.5 * rows), squeeze=False)

    summary = []
    for ax, name in zip(itertools.chain(*axes), names):
        splits = by_site[name]
        # the AOI keyed by the site's short name, or by whatever the manifest called it
        aoi = aois.get(name) or next(
            (g for k, g in aois.items() if k in name or name.endswith(k)), None)
        if aoi is not None:
            for g in getattr(aoi, "geoms", [aoi]):
                ax.plot(*g.exterior.xy, color="#444", lw=1.2, zorder=1)

        for split in SPLITS:
            pts = splits.get(split, [])
            if pts:
                ax.scatter(*zip(*pts), s=9, c=COLOURS[split], alpha=.6, lw=0, zorder=2)

        worst = min(closest_between(splits.get(a, []), splits.get(b, []))
                    for a, b in itertools.combinations(SPLITS, 2))
        ok = worst > threshold
        missing = [s for s in SPLITS if not splits.get(s)]
        summary.append((name, {s: len(splits.get(s, [])) for s in SPLITS}, worst, missing))

        counts = " ".join(f"{s[0]}{len(splits.get(s, []))}" for s in SPLITS)
        ax.set_title(f"{name}\n{counts} · closest {worst:.1f} m"
                     + ("" if ok else "  ⚠ OVERLAP"),
                     fontsize=9, color="#191C16" if ok else "#A6402B")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(useOffset=False, style="plain")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)

    for ax in list(itertools.chain(*axes))[len(names):]:
        ax.axis("off")

    handles = [Line2D([], [], marker="o", ls="", color=COLOURS[s], label=s) for s in SPLITS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"tile provenance by split — footprint {extent:.2f} m, "
                 f"overlap impossible above {threshold:.2f} m", fontsize=11)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="dataset with train/val/test subdirs")
    p.add_argument("--config", default=None, help="sites config, to draw AOI outlines")
    p.add_argument("--out", default=None, help="output png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")                    # headless: the CLI only ever writes a file
    _, summary = plot(a.data_dir, a.config, a.out)
    bad = 0
    print(f"{'site':28s} {'train':>6s} {'val':>5s} {'test':>5s} {'closest':>9s}")
    for name, counts, worst, missing in summary:
        flag = ""
        if missing:
            flag = f"  <- no {'/'.join(missing)} tiles"; bad += 1
        print(f"{name:28s} {counts['train']:6d} {counts['val']:5d} {counts['test']:5d} "
              f"{worst:8.1f}m{flag}")
    if a.out:
        print(f"\nwrote {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

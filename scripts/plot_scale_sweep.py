"""Aggregate `scale_sweep.py` JSON across arms and seeds — paired table + one figure.

The comparison is **paired at each scale**: seed s of arm A and seed s of arm B saw the
same tiles, the same initialisation stream and the same everything except the crop range,
so the seed's contribution cancels in each difference. Judging a difference against the
*within-arm* spread instead would discard exactly that — it read a 4/4-consistent effect
as noise once already in this repo (see `scripts/paired_ablation.py`).

Absolute F1 is **not** comparable across scales: a coarser tile covers more ground and
thinner stems, so F1 falls with GSD for reasons unrelated to either arm. The readable
quantity is the arm-vs-arm difference *within* a scale, and each arm's drop from its own
peak — which is the robustness claim, independent of absolute level.

    python scripts/plot_scale_sweep.py --sweep-dir runs/scale-aug --arms fixed jitter \\
        --out-fig docs/figures/scale-sweep.png --json-out docs/results/scale-sweep.json
"""
import argparse
import glob
import json
import math
import os
import re


def load(sweep_dir, arms):
    """sweep-<arm>-s<seed>.json -> {arm: {seed: {crop_px: row}}}, plus the crop axis."""
    out, crops = {a: {} for a in arms}, []
    for p in sorted(glob.glob(os.path.join(sweep_dir, "sweep-*.json"))):
        with open(p) as f:
            d = json.load(f)
        m = re.match(r"^(.+)-s(\d+)$", d.get("label", ""))
        if not m or m.group(1) not in out:
            continue
        rows = {r["crop_px"]: r for r in d["rows"]}
        out[m.group(1)][int(m.group(2))] = rows
        if not crops:
            crops = [r["crop_px"] for r in d["rows"]]
        elif crops != [r["crop_px"] for r in d["rows"]]:
            raise SystemExit(f"{p} has a different crop axis — arms are not comparable")
    return out, crops


def paired(data, arms, crops):
    """Per-scale paired differences (arms[1] - arms[0]) over the seeds both arms share."""
    a, b = arms
    seeds = sorted(set(data[a]) & set(data[b]))
    if not seeds:
        raise SystemExit(f"no seed appears in both {a} and {b}")
    rows = []
    for c in crops:
        diffs = [data[b][s][c]["f1"] - data[a][s][c]["f1"] for s in seeds]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
        pos = sum(d > 0 for d in diffs)
        # Paired t on the differences themselves. |t| >= 3 with unanimous signs is the
        # bar used elsewhere in this repo; with n=3 that is deliberately conservative.
        t = mean / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mean else 0.0)
        rows.append({
            "crop_px": c,
            "gsd_cm": data[a][seeds[0]][c].get("gsd_cm"),
            "ratio": data[a][seeds[0]][c].get("ratio"),
            "seeds": seeds,
            f"{a}_f1": [data[a][s][c]["f1"] for s in seeds],
            f"{b}_f1": [data[b][s][c]["f1"] for s in seeds],
            f"{a}_mean": sum(data[a][s][c]["f1"] for s in seeds) / len(seeds),
            f"{b}_mean": sum(data[b][s][c]["f1"] for s in seeds) / len(seeds),
            "diffs": diffs, "mean_diff": mean, "sd_diff": sd,
            "positive": pos, "n": n, "t": t,
            "consistent": pos in (0, n),
            "strong": abs(t) >= 3.0 and pos in (0, n),
        })
    return rows


def drop_from_peak(data, arm, crops):
    """Each seed's worst-minus-best across the sweep — the robustness claim itself."""
    out = {}
    for s, rows in sorted(data[arm].items()):
        f1 = [rows[c]["f1"] for c in crops]
        out[s] = {"peak": max(f1), "worst": min(f1), "spread": max(f1) - min(f1),
                  "peak_crop": crops[f1.index(max(f1))]}
    return out


def paired_spread(peaks, arms):
    """The robustness claim itself, paired: does arm B flatten the curve, seed by seed?

    Per-scale differences answer "which arm is better *here*"; this answers "which arm
    varies less across the sweep", which is what a user changing `tile_size` experiences.
    """
    a, b = arms
    seeds = sorted(set(peaks[a]) & set(peaks[b]))
    diffs = [peaks[b][s]["spread"] - peaks[a][s]["spread"] for s in seeds]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mean else 0.0)
    neg = sum(d < 0 for d in diffs)          # negative = arm B is flatter
    return {"seeds": seeds, "diffs": diffs, "mean_diff": mean, "sd_diff": sd,
            "n": n, "t": t, "flatter": neg, "consistent": neg in (0, n),
            "strong": abs(t) >= 3.0 and neg in (0, n)}


def figure(data, arms, crops, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {arms[0]: "#1f77b4", arms[1]: "#d62728"}
    x = [data[arms[0]][sorted(data[arms[0]])[0]][c].get("gsd_cm") or c for c in crops]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a in arms:
        seeds = sorted(data[a])
        series = [[data[a][s][c]["f1"] for c in crops] for s in seeds]
        for y in series:                       # per-seed spread, drawn not summarised
            ax.plot(x, y, color=colors[a], alpha=0.25, lw=1)
        mean = [sum(v[i] for v in series) / len(series) for i in range(len(crops))]
        ax.plot(x, mean, color=colors[a], lw=2.4, marker="o",
                label=f"{a}  (n={len(seeds)})")
    ax.axvline(2.9297, color="0.4", ls="--", lw=1)
    ax.annotate("Analyzer default\ntile_size 15 m", (2.9297, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8, color="0.35")
    ax.set_xlabel("effective ground resolution (cm/px)")
    ax.set_ylabel("pixel F1")
    ax.set_title("Scale robustness: fixed vs ±30% footprint jitter\n"
                 "one 666 px test set, centre-cropped to five scales", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--arms", nargs=2, default=["fixed", "jitter"],
                   help="baseline first; differences are second minus first")
    p.add_argument("--out-fig", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    data, crops = load(a.sweep_dir, a.arms)
    missing = [x for x in a.arms if not data[x]]
    if missing:
        raise SystemExit(f"no sweep JSON found for: {', '.join(missing)}")
    rows = paired(data, a.arms, crops)
    lo, hi = a.arms

    print(f"{'GSD':>7} {'ratio':>6} {lo:>9} {hi:>9} {'diff':>8} {'sign':>6} {'t':>7}")
    for r in rows:
        print(f"{r['gsd_cm'] or 0:7.3f} {r['ratio'] or 0:6.2f} "
              f"{r[lo + '_mean']:9.4f} {r[hi + '_mean']:9.4f} {r['mean_diff']:+8.4f} "
              f"{r['positive']}/{r['n']:>4} {r['t']:7.2f}"
              + ("  *" if r["strong"] else ""))

    print("\ndrop from each arm's own peak (robustness, per seed):")
    peaks = {}
    for arm in a.arms:
        peaks[arm] = drop_from_peak(data, arm, crops)
        sp = [v["spread"] for v in peaks[arm].values()]
        print(f"  {arm:>9}  " + "  ".join(f"s{s}:{v['spread']:.4f}"
                                          for s, v in peaks[arm].items())
              + f"   mean {sum(sp) / len(sp):.4f}")

    ps = paired_spread(peaks, a.arms)
    print(f"  paired: {hi} minus {lo} spread = {ps['mean_diff']:+.4f} "
          f"({ps['flatter']}/{ps['n']} seeds flatter, t {ps['t']:.2f})"
          + ("  *" if ps["strong"] else ""))

    if a.out_fig:
        print(f"\nfigure -> {figure(data, a.arms, crops, a.out_fig)}")
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"arms": a.arms, "crops": crops, "per_scale": rows,
                       "peaks": peaks, "paired_spread": ps}, f, indent=1)
        print(f"json   -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

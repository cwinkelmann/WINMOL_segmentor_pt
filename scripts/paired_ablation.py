"""Paired ablation over seeds: does a single change move the metric beyond noise?

Two runs of *identical* code on this corpus measured 2.0 F1 apart — `bce` and
`bce_hard_f1` have bit-identical gradients and still diverged, because cuDNN picks
algorithms by autotuning. Any single-run comparison smaller than that is unreadable.

The fix is pairing, not more runs: both arms train on the **same seed**, so the seed's
contribution cancels in the per-seed difference and only the change under test remains.
Report the per-seed differences, their mean, and how many seeds agree in sign — never a
mean alone, which one bad seed can carry.

    python scripts/paired_ablation.py --runs-root <runs>/loss-pairs \\
        --arm-a bce_soft_f1 --arm-b bce --seeds 1 2 3 4 5
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import parse  # noqa: E402


def gather(runs_root, arm, seeds, metric="f1"):
    out = {}
    for s in seeds:
        p = os.path.join(runs_root, f"{arm}-s{s}", "test_results.md")
        if os.path.isfile(p):
            m = parse(p)
            if m:
                out[s] = m
    return out


def compare(runs_root, arm_a, arm_b, seeds, metric="f1"):
    A, B = gather(runs_root, arm_a, seeds), gather(runs_root, arm_b, seeds)
    paired = sorted(set(A) & set(B))
    rows = [(s, A[s][metric], B[s][metric], A[s][metric] - B[s][metric]) for s in paired]
    diffs = [r[3] for r in rows]
    res = {"arm_a": arm_a, "arm_b": arm_b, "metric": metric,
           "seeds_paired": paired, "rows": rows,
           "missing_a": sorted(set(seeds) - set(A)),
           "missing_b": sorted(set(seeds) - set(B))}
    if diffs:
        res["mean_diff"] = statistics.fmean(diffs)
        res["sd_diff"] = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
        res["positive"] = sum(d > 0 for d in diffs)
        res["n"] = len(diffs)
        # spread within each arm across seeds = the noise this design is controlling for
        res["spread_a"] = max(A[s][metric] for s in paired) - min(A[s][metric] for s in paired)
        res["spread_b"] = max(B[s][metric] for s in paired) - min(B[s][metric] for s in paired)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--arm-a", required=True, help="run dirs are <arm>-s<seed>")
    p.add_argument("--arm-b", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--metric", default="f1")
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    r = compare(a.runs_root, a.arm_a, a.arm_b, a.seeds, a.metric)
    if not r.get("n"):
        print(f"no paired seeds. missing {a.arm_a}: {r['missing_a']}, "
              f"{a.arm_b}: {r['missing_b']}")
        return 1

    print(f"{a.arm_a}  vs  {a.arm_b}   ({a.metric}, {r['n']} paired seeds)\n")
    print(f"{'seed':>5} {a.arm_a[:14]:>14} {a.arm_b[:14]:>14} {'diff':>9}")
    for s, va, vb, d in r["rows"]:
        print(f"{s:5d} {va:14.4f} {vb:14.4f} {d:+9.4f}")
    print(f"\nmean diff {r['mean_diff']:+.4f}   sd {r['sd_diff']:.4f}   "
          f"same sign in {r['positive']}/{r['n']} seeds")
    print(f"within-arm spread across seeds: {a.arm_a} {r['spread_a']:.4f}, "
          f"{a.arm_b} {r['spread_b']:.4f}")
    # a difference smaller than the spread the seeds alone produce is not a result
    worst = max(r["spread_a"], r["spread_b"])
    verdict = ("separable — |mean diff| exceeds the within-arm seed spread"
               if abs(r["mean_diff"]) > worst else
               "NOT separable — the change moves less than seeds alone do")
    print(f"verdict: {verdict}")
    if r["missing_a"] or r["missing_b"]:
        print(f"note: unpaired seeds dropped — {a.arm_a} missing {r['missing_a']}, "
              f"{a.arm_b} missing {r['missing_b']}")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

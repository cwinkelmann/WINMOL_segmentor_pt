"""Collect every run's `test_results.md` into one JSON, verbatim.

The experiment skill's rule: reports render from JSON that was copied from each run's own
`test_results.md`, and compute no metrics themselves. That way a report cannot drift from
what training actually reported, and adding a run is a data edit rather than a code edit.

    python -m scripts.collect_results --runs-root /raid/.../runs --out results.json
"""
import argparse
import json
import os
import re

_METRIC = re.compile(r"^\|\s*(f1|precision|recall|loss)\s*\|\s*([0-9.]+)\s*\|", re.M)
_HEADER = re.compile(r"\*\*TestDS:\*\*\s*`([^`]+)`\s*\((\d+) tiles\)\s*\|\s*\*\*arch:\*\*\s*(\S+)")


def parse(path):
    """Return the metrics dict a run wrote, or None if it never finished."""
    with open(path) as f:
        text = f.read()
    metrics = {k: float(v) for k, v in _METRIC.findall(text)}
    if not metrics:
        return None
    h = _HEADER.search(text)
    if h:
        metrics.update(test_data_dir=h.group(1), test_tiles=int(h.group(2)),
                       arch=h.group(3))
    return metrics


def collect(runs_root):
    out = {}
    for group in sorted(os.listdir(runs_root)):
        gdir = os.path.join(runs_root, group)
        if not os.path.isdir(gdir):
            continue
        runs = {}
        for run in sorted(os.listdir(gdir)):
            p = os.path.join(gdir, run, "test_results.md")
            if os.path.isfile(p):
                m = parse(p)
                if m:
                    runs[run] = m
        if runs:
            out[group] = runs
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    data = collect(a.runs_root)
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    n = sum(len(v) for v in data.values())
    print(f"collected {n} runs across {len(data)} groups -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

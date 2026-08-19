"""Extract tiles for several sites concurrently, then merge into one dataset.

`geo.splits` walks its sites one at a time on a single core, and the cost is dominated
by decoding the orthomosaic — a 19.512 m window on 1.2 cm/px imagery is 1,626 px square,
spanning ~16 WebP blocks, and 100x oversampling means many more windows are decoded than
kept. Measured on T14: 98.9% of one core, ~1 tile/s, while the NAS supplied 117 MB/s and
was never the constraint.

The sites are independent, so this runs one `geo.splits` per site in its own process and
merges the results. The merge is the part that needs care: each part numbers its tiles from
1, so indices must be reassigned or two sites silently overwrite each other — and
`tiles.jsonl` carries the old `n`, which must be rewritten to match or provenance points at
the wrong tile.

    python prepare.py --jobs N --config sites.json --out DS --jobs 5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SPLITS = ("train", "val", "test")


def _part_configs(cfg_path, work_dir):
    """One config per site, each keeping that site's own split assignment."""
    cfg = json.load(open(cfg_path))
    os.makedirs(work_dir, exist_ok=True)
    out = []
    for site in cfg["sites"]:
        one = {k: v for k, v in cfg.items() if k != "sites"}
        one["sites"] = [site]
        p = os.path.join(work_dir, f"cfg_{site['name']}.json")
        with open(p, "w") as f:
            json.dump(one, f, indent=1)
        out.append((site["name"], p))
    return out


def _run_parts(parts, work_dir, strategy, jobs, python=None,
               seed=1, cut_axis="auto", skip_fix=False):
    """Launch the per-site extractions, at most `jobs` at a time.

    seed / cut_axis / skip_fix are forwarded to every worker. They were previously dropped,
    which made --jobs a correctness knob rather than a speed one: the same config extracted
    with --jobs 1 and --jobs 4 produced different tiles, silently.
    """
    python = python or sys.executable
    running, results = [], []

    def _reap(block):
        while running and (block or len(running) >= jobs):
            name, proc, log = running.pop(0)
            rc = proc.wait()
            results.append((name, rc, log))
            if rc != 0:
                tail = open(log).read()[-800:] if os.path.exists(log) else ""
                raise SystemExit(f"extraction failed for {name} (exit {rc}):\n{tail}")

    for name, cfg in parts:
        _reap(block=False)
        outdir = os.path.join(work_dir, name)
        log = os.path.join(work_dir, f"{name}.log")
        cmd = [python, "-m", "winmol_unet.geo.splits",
               "--config", cfg, "--out", outdir, "--strategy", strategy,
               "--seed", str(seed), "--cut-axis", cut_axis]
        if skip_fix:
            cmd.append("--skip-fix")
        with open(log, "w") as lf:
            running.append((name, subprocess.Popen(cmd, stdout=lf, stderr=lf), log))
        print(f"  started {name}")
    _reap(block=True)
    return results


def merge(work_dir, part_names, out_dir, quiet=False):
    """Concatenate the per-site parts into one dataset, renumbering as we go."""
    counts = {}
    for split in SPLITS:
        n_out = 0
        rows = []
        for name in part_names:
            src = os.path.join(work_dir, name, split)
            img_dir = os.path.join(src, "train")
            if not os.path.isdir(img_dir):
                continue
            by_n = {}
            for f in os.listdir(img_dir):
                if not f.startswith("train"):
                    continue
                stem, ext = os.path.splitext(f)
                by_n[int(stem[5:])] = ext
            if not by_n:
                continue
            jl = os.path.join(src, "tiles.jsonl")
            meta = {}
            if os.path.exists(jl):
                for line in open(jl):
                    r = json.loads(line)
                    meta[r["n"]] = r
            for n in sorted(by_n):
                ext = by_n[n]
                mask = os.path.join(src, "mask", f"mask{n}.gif")
                if not os.path.exists(mask):
                    raise SystemExit(f"{img_dir}/train{n}{ext} has no mask — refusing a "
                                     f"partial merge")
                n_out += 1
                for sub, sname, dname in (("train", f"train{n}{ext}", f"train{n_out}{ext}"),
                                          ("mask", f"mask{n}.gif", f"mask{n_out}.gif")):
                    d = os.path.join(out_dir, split, sub)
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(os.path.join(src, sub, sname), os.path.join(d, dname))
                r = dict(meta.get(n, {}))
                # rewrite n so provenance still points at this tile after renumbering
                r["n"] = n_out
                r["part"] = name
                rows.append(r)
        if rows:
            os.makedirs(os.path.join(out_dir, split), exist_ok=True)
            with open(os.path.join(out_dir, split, "tiles.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        counts[split] = n_out
        if not quiet:
            print(f"  {split:5s} {n_out}")
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", default="sites", choices=("sites", "blocks", "halve"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--python", default=None, help="interpreter for the workers")
    p.add_argument("--keep-parts", action="store_true")
    # Forwarded to the workers. Anything that changes what gets sampled belongs here, or
    # --jobs stops being a pure speed knob.
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cut-axis", default="auto", choices=("auto", "ns", "ew"))
    p.add_argument("--skip-fix", action="store_true")
    a = p.parse_args(argv)

    work = os.path.join(a.out, "_parts")
    os.makedirs(work, exist_ok=True)
    parts = _part_configs(a.config, work)
    print(f"extracting {len(parts)} sites, {a.jobs} at a time")
    _run_parts(parts, work, a.strategy, a.jobs, a.python,
               seed=a.seed, cut_axis=a.cut_axis, skip_fix=a.skip_fix)
    print("merging:")
    counts = merge(work, [n for n, _ in parts], a.out)
    if not a.keep_parts:
        shutil.rmtree(work, ignore_errors=True)
    print(f"done: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

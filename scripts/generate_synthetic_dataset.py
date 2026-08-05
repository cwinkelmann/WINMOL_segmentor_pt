"""Generate a loader-ready synthetic dataset: sample scenes, render, convert.

    python scripts/generate_synthetic_dataset.py --out <DS> --n 500 [--seed 1]
        [--samples 48] [--gsd 0.02] [--keep-depth]

Writes the convention training/dataset.py expects — train/train{N}.jpeg +
mask/mask{N}.gif, plus depth/depth{N}.png with --keep-depth — so a synthetic set
drops straight into `--gen-data-dir` with no loader changes.

Each scene renders in its own subprocess: a Blender crash costs one tile, and the
run reports what it dropped rather than silently emitting a short dataset.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthgen.sampler import GenConfig, sample_scene   # noqa: E402

MIN_STEM_FRACTION = 0.001      # below this the tile is effectively empty
MAX_STEM_FRACTION = 0.6        # above this the stems have swallowed the tile


def _rendered(out_dir, stem_name):
    """File Output nodes append the frame number, so match by prefix."""
    for name in sorted(os.listdir(out_dir)):
        if name.startswith(stem_name) and name.endswith(".png"):
            return os.path.join(out_dir, name)
    return None


def _convert(render_dir, dst, n, keep_depth):
    """Render outputs -> dataset convention. Returns the stem-pixel fraction."""
    rgb_p, mask_p = _rendered(render_dir, "rgb"), _rendered(render_dir, "mask")
    if not rgb_p or not mask_p:
        raise FileNotFoundError(f"renderer produced no rgb/mask in {render_dir}")

    mask = np.asarray(Image.open(mask_p).convert("L")) >= 128
    frac = float(mask.mean())
    if not MIN_STEM_FRACTION <= frac <= MAX_STEM_FRACTION:
        return frac                                     # degenerate; caller drops it

    Image.open(rgb_p).convert("RGB").save(
        os.path.join(dst, "train", f"train{n}.jpeg"), quality=95)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("P").save(
        os.path.join(dst, "mask", f"mask{n}.gif"))
    if keep_depth:
        depth_p = _rendered(render_dir, "depth")
        if depth_p:
            Image.open(depth_p).save(os.path.join(dst, "depth", f"depth{n}.png"))
    return frac


def generate(dst, n_scenes, seed, samples, cfg, keep_depth):
    for sub in ("train", "mask") + (("depth",) if keep_depth else ()):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    written, dropped = 0, []
    for i in range(n_scenes):
        spec = sample_scene(cfg, seed=seed + i)
        with tempfile.TemporaryDirectory() as td:
            spec_path = os.path.join(td, "spec.json")
            with open(spec_path, "w") as f:
                f.write(spec.to_json())
            proc = subprocess.run(
                [sys.executable, "-m", "synthgen.render_bpy", spec_path, td],
                capture_output=True, text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                env={**os.environ, "SYNTHGEN_SAMPLES": str(samples)})
            if proc.returncode != 0:
                dropped.append((seed + i, f"render failed: {proc.stderr.strip()[-200:]}"))
                continue
            try:
                frac = _convert(td, dst, written + 1, keep_depth)
            except FileNotFoundError as exc:
                dropped.append((seed + i, str(exc)))
                continue
            if not MIN_STEM_FRACTION <= frac <= MAX_STEM_FRACTION:
                dropped.append((seed + i, f"degenerate stem fraction {frac:.4f}"))
                continue
        written += 1
        if written % 25 == 0:
            print(f"  {written}/{n_scenes} tiles", flush=True)

    manifest = {"scenes_requested": n_scenes, "tiles_written": written,
                "dropped": [{"seed": s, "reason": r} for s, r in dropped],
                "base_seed": seed, "gsd_m_per_px": cfg.gsd_m_per_px,
                "tile_px": cfg.tile_px, "cycles_samples": samples}
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {written} tiles to {dst} ({len(dropped)} dropped)")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="dataset dir to create")
    p.add_argument("--n", type=int, default=100, help="scenes to render")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--samples", type=int, default=48, help="Cycles samples per tile")
    p.add_argument("--gsd", type=float, default=0.02, help="ground sampling distance (m/px)")
    p.add_argument("--tile-px", type=int, default=512)
    p.add_argument("--keep-depth", action="store_true",
                   help="also write depth/depth{N}.png for --rgbd training")
    a = p.parse_args(argv)
    cfg = GenConfig(tile_px=a.tile_px, gsd_m_per_px=a.gsd)
    generate(a.out, a.n, a.seed, a.samples, cfg, a.keep_depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())

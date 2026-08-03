"""Add vegetation to synthetic tiles by inpainting the background only.

    python scripts/inpaint_vegetation.py --src <SynthDS> --dst <SynthDS-veg> \
        [--margin-px 3] [--strength 0.85] [--steps 30] [--seed 1] [--limit N] \
        [--model stabilityai/stable-diffusion-2-inpainting]

Masks and depth are copied through untouched: only the RGB changes, and only
outside a dilated collar around the stems, so `mask{N}.gif` still describes
`train{N}.jpeg` exactly. Every tile is verified pixel-by-pixel before it is
written — a tile whose labelled pixels moved is rejected, not silently kept.

Needs a GPU and `pip install diffusers transformers accelerate` (the render env
on the training box). The first run downloads the model (~5 GB).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthgen.inpaint import (                      # noqa: E402
    composite, repaint_region, stems_are_intact, vegetation_coverage,
)

DEFAULT_PROMPT = (
    "aerial close-up photograph of a temperate forest floor, green undergrowth, "
    "ferns, moss patches, grass tufts, fallen leaves, natural daylight, "
    "highly detailed, realistic"
)
DEFAULT_NEGATIVE = (
    "tree trunk, log, branch, timber, wood, people, buildings, text, watermark, "
    "cartoon, painting, blurry"
)


def _ids(src):
    out = []
    for name in os.listdir(os.path.join(src, "train")):
        stem, ext = os.path.splitext(name)
        if ext == ".jpeg" and stem.startswith("train") and stem[5:].isdigit():
            out.append(int(stem[5:]))
    return sorted(out)


def _load_pipeline(model, device="cuda"):
    import torch
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        model, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def inpaint_dataset(src, dst, args):
    import torch

    pipe = _load_pipeline(args.model)
    for sub in ("train", "mask", "depth"):
        if os.path.isdir(os.path.join(src, sub)):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)

    ids = _ids(src)[: args.limit] if args.limit else _ids(src)
    kept, rejected, coverages = 0, [], []
    for i, n in enumerate(ids):
        img = Image.open(os.path.join(src, "train", f"train{n}.jpeg")).convert("RGB")
        mask_img = Image.open(os.path.join(src, "mask", f"mask{n}.gif")).convert("L")
        stem = np.asarray(mask_img) >= 128
        region = repaint_region(stem, margin_px=args.margin_px)
        if not region.any():                       # nothing left to paint into
            rejected.append((n, "no repaintable area"))
            continue

        # the pipeline's mask convention: white = generate here
        region_img = Image.fromarray((region * 255).astype(np.uint8), mode="L")
        gen = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=img,
            mask_image=region_img,
            strength=args.strength,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + n),
        ).images[0].resize(img.size)

        orig_a = np.asarray(img)
        out = composite(orig_a, np.asarray(gen), region)
        if not stems_are_intact(orig_a, out, stem):        # cannot happen, but prove it
            rejected.append((n, "labelled pixels changed"))
            continue
        cov = vegetation_coverage(orig_a, out, region)
        if cov < args.min_coverage:
            rejected.append((n, f"generator changed only {cov:.1%} of the background"))
            continue

        Image.fromarray(out).save(os.path.join(dst, "train", f"train{n}.jpeg"), quality=95)
        shutil.copy2(os.path.join(src, "mask", f"mask{n}.gif"),
                     os.path.join(dst, "mask", f"mask{n}.gif"))
        depth_src = os.path.join(src, "depth", f"depth{n}.png")
        if os.path.exists(depth_src):
            shutil.copy2(depth_src, os.path.join(dst, "depth", f"depth{n}.png"))
        kept += 1
        coverages.append(cov)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)} tiles", flush=True)

    manifest = {
        "source": os.path.abspath(src), "model": args.model, "prompt": args.prompt,
        "negative_prompt": args.negative_prompt, "strength": args.strength,
        "steps": args.steps, "margin_px": args.margin_px, "seed": args.seed,
        "tiles_kept": kept,
        "rejected": [{"id": n, "reason": r} for n, r in rejected],
        "mean_background_changed": float(np.mean(coverages)) if coverages else 0.0,
    }
    with open(os.path.join(dst, "manifest_inpaint.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"kept {kept} tiles, rejected {len(rejected)}; "
          f"mean background changed {manifest['mean_background_changed']:.1%}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True, help="synthetic dataset to read")
    p.add_argument("--dst", required=True, help="dataset to write")
    p.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE,
                   help="steer away from wood-like content: generated branches would "
                        "be unlabelled stems, i.e. false-negative teaching signal")
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--margin-px", type=int, default=3,
                   help="protected collar around stems (blending bleeds inward)")
    p.add_argument("--min-coverage", type=float, default=0.05,
                   help="reject tiles the generator barely touched")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    inpaint_dataset(a.src, a.dst, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Publish WINMOL model artifacts to a GitHub Release.

Uploads the **original** full-precision UNet (fp32 ONNX) plus the **best optimised** models
(CPU int8 and GPU fp16 — both the width-0.5 UNet, lossless), with SHA256 checksums and a
self-describing manifest. Idempotent: creates the release if missing, otherwise updates it and
re-uploads assets with --clobber.

Requires an authenticated `gh` CLI. Model sources live outside the repo (the model archive +
the cpu-speedup build outputs); paths are configurable. Run --dry-run first to see the plan.

  python scripts/deploy_models_to_release.py --dry-run
  python scripts/deploy_models_to_release.py            # actually publish
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

# Each entry: release asset name -> (source relative to a base dir, base, human metadata).
# base is "archive" (the model archive) or "optimised" (cpu-speedup build outputs).
RELEASE_MODELS = [
    {"asset": "unet_fp32.onnx", "base": "archive",
     "src": "pytorch/twostage_lrfix/unet.onnx",
     "title": "Original UNet (fp32)", "backend": "any",
     "notes": "reference WINMOL UNet, two-stage lrfix, TestDS F1 0.760"},
    {"asset": "unet_w05_int8_cpu.onnx", "base": "optimised", "src": "w05_static.onnx",
     "title": "CPU-optimised (width-0.5 + static int8)", "backend": "CPU",
     "notes": "10x faster CPU (AVX-VNNI), TestDS F1 0.760 (lossless)"},
    {"asset": "unet_w05_fp16_gpu.onnx", "base": "optimised", "src": "w05_fp16.onnx",
     "title": "GPU-optimised (width-0.5 + fp16)", "backend": "GPU",
     "notes": "Tensor-Core fp16, TestDS F1 0.760 (lossless), served via CUDA EP"},
    {"asset": "unet_w025_int8_cpu.onnx", "base": "optimised", "src": "w025_static.onnx",
     "title": "CPU-optimised aggressive (width-0.25 + static int8)", "backend": "CPU",
     "notes": "29x faster CPU, 2 MB, TestDS F1 0.755 (-0.006)"},
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(tag, entries):
    """A self-describing markdown manifest for the release (one row per asset)."""
    lines = [
        f"# WINMOL tree-stem segmentation models — `{tag}`", "",
        "All models share the frozen ONNX contract: input `[batch,3,512,512]` float32 in [0,1] "
        "NCHW, output `[batch,1,512,512]` (sigmoid baked in, opset 17, dynamic batch). They load "
        "unchanged in `winmol_unet.runtime.OnnxSegmenter`.", "",
        "| asset | backend | size (MB) | notes | sha256 |",
        "|-------|---------|----------:|-------|--------|",
    ]
    for e in entries:
        lines.append(f"| `{e['asset']}` | {e['backend']} | {e.get('size_mb', 0):.1f} | "
                     f"{e['notes']} | `{e['sha256']}` |")
    lines += ["",
              "**Which to use:** `unet_w05_int8_cpu.onnx` for CPU deployment (10x, lossless), "
              "`unet_w05_fp16_gpu.onnx` for GPU (lossless), `unet_fp32.onnx` as the full-precision "
              "reference. `unet_w025_int8_cpu.onnx` is the aggressive 2 MB option (slight F1 cost).",
              "", "Provenance: `docs/2026-07-21-cpu-inference-speedup-results.md`."]
    return "\n".join(lines) + "\n"


def stage(models, archive_dir, optimised_dir, staging_dir):
    """Copy each source into staging under its asset name; return entries + hash/size."""
    os.makedirs(staging_dir, exist_ok=True)
    bases = {"archive": archive_dir, "optimised": optimised_dir}
    out = []
    for m in models:
        src = os.path.join(bases[m["base"]], m["src"])
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing model source: {src}")
        dst = os.path.join(staging_dir, m["asset"])
        shutil.copyfile(src, dst)
        out.append({**m, "path": dst, "sha256": sha256_file(dst),
                    "size_mb": os.path.getsize(dst) / 1e6})
    return out


def _run(cmd, dry):
    print(("DRY  " if dry else "RUN  ") + " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def publish(tag, title, repo, entries, staging_dir, dry_run):
    # release exists?
    exists = subprocess.run(["gh", "release", "view", tag, "--repo", repo],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    notes_path = os.path.join(staging_dir, "manifest.md")
    if not exists:
        _run(["gh", "release", "create", tag, "--repo", repo, "--title", title,
              "--notes-file", notes_path], dry_run)
    else:
        _run(["gh", "release", "edit", tag, "--repo", repo, "--title", title,
              "--notes-file", notes_path], dry_run)
    assets = [e["path"] for e in entries] + [notes_path,
                                             os.path.join(staging_dir, "SHA256SUMS")]
    _run(["gh", "release", "upload", tag, "--repo", repo, "--clobber", *assets], dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="cwinkelmann/WINMOL_segmentor_pt")
    ap.add_argument("--tag", default="models-v1")
    ap.add_argument("--title", default="WINMOL models v1 — original + CPU/GPU-optimised")
    ap.add_argument("--archive-dir", default="/data/mnt/storage/hnee/WINMOL/models")
    ap.add_argument("--optimised-dir", default="results/cpu_speedup/models")
    ap.add_argument("--staging-dir", default="results/cpu_speedup/release_staging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = stage(RELEASE_MODELS, args.archive_dir, args.optimised_dir, args.staging_dir)
    # checksums + manifest
    with open(os.path.join(args.staging_dir, "SHA256SUMS"), "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['asset']}\n")
    with open(os.path.join(args.staging_dir, "manifest.md"), "w") as f:
        f.write(build_manifest(args.tag, entries))

    total = sum(e["size_mb"] for e in entries)
    print(f"staged {len(entries)} assets ({total:.0f} MB) in {args.staging_dir}:")
    for e in entries:
        print(f"  {e['asset']:<26} {e['size_mb']:6.1f} MB  {e['backend']:<4} {e['sha256'][:12]}")
    publish(args.tag, args.title, args.repo, entries, args.staging_dir, args.dry_run)
    print("done." if not args.dry_run else "dry-run complete (nothing uploaded).")


if __name__ == "__main__":
    main()

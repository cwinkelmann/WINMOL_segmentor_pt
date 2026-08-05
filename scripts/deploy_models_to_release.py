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
# base dirs: "archive" (model archive root), "optimised" (cpu-speedup build outputs),
# "keras" (Keras->ONNX conversions), "zenodo" (upstream HDF5 under archive/zenodo_analyzer).

# Our retrained models are all beech (GenDS10 -> SpecDS-beech), so they carry the SpecDS_Beech
# flavour with an <arch>_<framework> qualifier (and _w05 for the width-scaled optimised UNet).
_B = "model_{arch}_SpecDS_Beech_512{q}"

# The retrained PyTorch UNet: original (full-width) + the best optimised (width-0.5) variants.
PYTORCH_MODELS = [
    {"asset": _B.format(arch="UNet", q="_pytorch") + ".onnx", "base": "archive",
     "src": "pytorch/twostage_lrfix/unet.onnx", "group": "PyTorch UNet (our beech retrain)",
     "title": "UNet fp32 (full width)", "backend": "any",
     "notes": "our PyTorch WINMOL UNet, two-stage lrfix, TestDS F1 0.760"},
    {"asset": _B.format(arch="UNet", q="_pytorch_w05_int8") + ".onnx", "base": "optimised",
     "src": "w05_static.onnx", "group": "PyTorch UNet (our beech retrain)",
     "title": "UNet width-0.5 int8 (CPU-optimised)", "backend": "CPU",
     "notes": "10x faster CPU (AVX-VNNI), TestDS F1 0.760 (lossless)"},
    {"asset": _B.format(arch="UNet", q="_pytorch_w05_fp16") + ".onnx", "base": "optimised",
     "src": "w05_fp16.onnx", "group": "PyTorch UNet (our beech retrain)",
     "title": "UNet width-0.5 fp16 (GPU-optimised)", "backend": "GPU",
     "notes": "Tensor-Core fp16, TestDS F1 0.760 (lossless)"},
]

# Our R/Keras reimplementation, 512px beech UNet (same family as Zenodo SpecDS_Beech),
# converted to ONNX + PTQ variants. The 256px R models can't meet the 512 ONNX contract.
RKERAS_MODELS = [
    {"asset": _B.format(arch="UNet", q="_rkeras") + ".onnx", "base": "keras",
     "src": "unet_rkeras_beech_512.onnx", "group": "R/Keras UNet (our 512 beech retrain)",
     "title": "R/Keras UNet fp32", "backend": "any",
     "notes": "our R reimplementation, two-stage LR-fixed, TestDS F1 0.738 (512px)"},
    {"asset": _B.format(arch="UNet", q="_rkeras_fp16") + ".onnx", "base": "keras",
     "src": "unet_rkeras_beech_512_fp16.onnx", "group": "R/Keras UNet (our 512 beech retrain)",
     "title": "R/Keras UNet fp16 (GPU)", "backend": "GPU", "notes": "post-training fp16, lossless"},
    {"asset": _B.format(arch="UNet", q="_rkeras_int8") + ".onnx", "base": "keras",
     "src": "unet_rkeras_beech_512_int8.onnx", "group": "R/Keras UNet (our 512 beech retrain)",
     "title": "R/Keras UNet int8 (CPU)", "backend": "CPU", "notes": "post-training static int8, TestDS F1 0.740"},
]

# Our other 512px PyTorch architectures (beech, two-stage LR-fixed): fp32 + fp16 (GPU). No
# width-scaled variant (width_mult is UNet-only); no int8 -- the smp decoders' symbolic shapes
# break the ORT static quantizer, and TRT-fp16 is the GPU path anyway.
ARCH_MODELS = []
for _arch, _archname, _f1 in [("deeplabv3plus", "DeepLabV3plus", "0.742"), ("hrnet", "HRNet", "0.764")]:
    _g = f"PyTorch {_archname} (our beech retrain)"
    ARCH_MODELS += [
        {"asset": _B.format(arch=_archname, q="") + ".onnx", "base": "optimised",
         "src": f"{_arch}_fp32.onnx", "group": _g, "title": f"{_archname} fp32", "backend": "any",
         "notes": f"two-stage LR-fixed, TestDS F1 {_f1}"},
        {"asset": _B.format(arch=_archname, q="_fp16") + ".onnx", "base": "optimised",
         "src": f"{_arch}_fp16_gpu.onnx", "group": _g, "title": f"{_archname} fp16 (GPU)",
         "backend": "GPU", "notes": "post-training fp16, lossless"},
    ]

# Upstream Keras flavours (Zenodo 15907576, CC-BY-4.0): can't retrain, but convert + PTQ-quantize.
# (schema-name, timestamped HDF5, human description). Naming schema preserved: model_UNet_<F>_512.
KERAS_FLAVOURS = [
    ("model_UNet_GenDS_512", "model_UNet_GenDS_512_2023-02-27_211141.hdf5", "generic (GenDS) pretrain"),
    ("model_UNet_SpecDS_Beech_512", "model_UNet_SpecDS_Beech_512_2023-02-28_042751.hdf5", "beech stems"),
    ("model_UNet_SpecDS_Spruce_512", "model_UNet_SpecDS_Spruce_512_2023-02-27_061925.hdf5", "spruce stems"),
    ("model_UNet_SpecDS_Spruce_Deadwood_512",
     "model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758.hdf5", "spruce + deadwood"),
]


def _keras_entries():
    # The upstream HDF5 originals stay on Zenodo (DOI 10.5281/zenodo.15907576) and are NOT
    # re-hosted here -- the release carries only the converted ONNX + its optimised variants.
    # Asset name = the exact upstream HDF5 filename with an .onnx suffix (keeps the timestamp so
    # each ONNX traces to its Zenodo source); src = the on-disk converted file (timestamp-stripped).
    out = []
    for name, hdf5, desc in KERAS_FLAVOURS:
        stem = hdf5[:-5] if hdf5.endswith(".hdf5") else hdf5   # full name incl. timestamp
        g = "Zenodo flavours — converted ONNX (upstream 15907576)"
        out += [
            {"asset": f"{stem}.onnx", "base": "keras", "src": f"{name}.onnx", "group": g,
             "title": f"{desc} — ONNX fp32 (converted)", "backend": "any",
             "notes": "converted from the upstream Keras HDF5, contract-conformant (numerically identical)"},
            {"asset": f"{stem}_fp16.onnx", "base": "keras", "src": f"{name}_fp16.onnx", "group": g,
             "title": f"{desc} — ONNX fp16 (GPU)", "backend": "GPU",
             "notes": "post-training fp16, lossless"},
            {"asset": f"{stem}_int8.onnx", "base": "keras", "src": f"{name}_int8.onnx", "group": g,
             "title": f"{desc} — ONNX int8 (CPU)", "backend": "CPU",
             "notes": "post-training static int8, domain-calibrated"},
        ]
    return out


RELEASE_MODELS = PYTORCH_MODELS + RKERAS_MODELS + ARCH_MODELS + _keras_entries()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(tag, entries):
    """A self-describing markdown manifest for the release, grouped by model family."""
    lines = [
        f"# WINMOL tree-stem segmentation models — `{tag}`", "",
        "**ONNX contract** (all `.onnx` here): input `[batch,3,512,512]` float32 in [0,1] NCHW, "
        "output `[batch,1,512,512]` (sigmoid baked in, opset 17, dynamic batch). Loads unchanged "
        "in `winmol_unet.runtime.OnnxSegmenter`. Suffixes: `_int8` = CPU-optimised (static int8), "
        "`_fp16` = GPU-optimised (Tensor-Core), both **post-training** (no retraining).",
    ]
    groups = []
    for e in entries:
        g = e.get("group", "Models")
        if g not in groups:
            groups.append(g)
    for g in groups:
        lines += ["", f"## {g}", "",
                  "| asset | backend | size (MB) | notes | sha256 |",
                  "|-------|---------|----------:|-------|--------|"]
        for e in entries:
            if e.get("group", "Models") == g:
                lines.append(f"| `{e['asset']}` | {e['backend']} | {e.get('size_mb', 0):.1f} | "
                             f"{e['notes']} | `{e['sha256'][:16]}…` |")
    lines += ["",
              "**Naming:** `model_<Arch>_<Flavour>_512[...]`. Converted Zenodo models keep their "
              "exact upstream filename + `.onnx` (timestamp traces to the source). Our retrained "
              "models carry a `_pytorch` / `_rkeras` qualifier (and `_w05` for the width-0.5 UNet). "
              "Precision suffix: none = fp32, `_fp16` = GPU, `_int8` = CPU.",
              "**Which to use (our best):** `model_UNet_SpecDS_Beech_512_pytorch_w05_int8.onnx` for "
              "CPU (10×, lossless), `...w05_fp16.onnx` for GPU. **Zenodo flavours:** pick your domain "
              "(Beech / Spruce / Spruce+Deadwood / generic GenDS). The original `.hdf5` stay at "
              "Zenodo DOI 10.5281/zenodo.15907576 (CC-BY-4.0), not re-hosted here.",
              "", "How optimisation works: see the repo README + "
              "`docs/2026-07-21-cpu-inference-speedup-results.md`. Full SHA256 in `SHA256SUMS`."]
    return "\n".join(lines) + "\n"


def stage(models, archive_dir, optimised_dir, keras_dir, staging_dir):
    """Copy each source into staging under its asset name; return entries + hash/size."""
    os.makedirs(staging_dir, exist_ok=True)
    bases = {"archive": archive_dir, "optimised": optimised_dir, "keras": keras_dir,
             "zenodo": os.path.join(archive_dir, "zenodo_analyzer")}
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
    ap.add_argument("--keras-dir", default="results/cpu_speedup/keras_onnx")
    ap.add_argument("--staging-dir", default="results/cpu_speedup/release_staging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = stage(RELEASE_MODELS, args.archive_dir, args.optimised_dir, args.keras_dir,
                    args.staging_dir)
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

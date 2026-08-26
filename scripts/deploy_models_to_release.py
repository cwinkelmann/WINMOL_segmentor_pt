"""Publish WINMOL model artifacts to a GitHub Release.

Uploads the **original** full-precision UNet (fp32 ONNX) plus the **best optimised** models
(CPU int8 and GPU fp16 — both the width-0.5 UNet, lossless), with SHA256 checksums and a
self-describing manifest. Idempotent: creates the release if missing, otherwise updates it and
re-uploads assets with --clobber.

Requires an authenticated `gh` CLI. Model sources live outside the repo (the model archive +
the cpu-speedup build outputs); paths are configurable. Run --dry-run first to see the plan.

  python scripts/deploy_models_to_release.py --dry-run
  python scripts/deploy_models_to_release.py            # actually publish

Two release sets are defined:

  --set v1  (default)  models-v1: the original Zenodo flavours + our first beech retrains,
                       plus their CPU/GPU-optimised variants.
  --set v2             models-v2: retrained on the four-site beech corpus (with scale jitter)
                       and on the Tegel R12/R13 survey. Sources come from carrot -- run
                       scripts/fetch_release_v2.sh first to stage them into results/release_v2.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ---------------------------------------------------------------------------
# models-v2 — retrained on the four-site beech corpus and on the Tegel R12/R13 survey.
# Sources are run directories on carrot (/raid/cwinkelmann/winmol/runs/...), staged locally
# by scripts/fetch_release_v2.sh. One model per arm: the best seed of three, chosen on the
# arm's own test set. Every F1 below is copied verbatim from the run's test_results.md
# (or from an eval_onnx JSON) -- this script computes no metrics.
#
# Two F1 columns, because they are two different exams:
#   "training test F1" -- test_results.md, the protocol the run was scored with (eval-tiling
#                         crop windows; 1600 windows over the 400 beech test tiles).
#   "ONNX re-score"    -- scripts/eval_onnx.py, plain 512 resize of each test tile, identical
#                         code for fp32/fp16/int8. Use it to read the *quantisation delta*,
#                         never to compare an arm against a differently-trained arm.
V2_BEECH = "four-site beech corpus (Campus, Campus_Oberheide, Bachsee_north, Kaufland), ±30% scale jitter"
V2_TEGEL = "Tegel R12-P2 + R13-P1 (July 2025), 2.93 cm/px"

V2_MODELS = [
    {"asset": "model_HRNet_Beech4Site_512_jitter.onnx", "src": "model_HRNet_Beech4Site_512_jitter.onnx",
     "group": "HRNet — four-site beech corpus (`scale-aug-20260813/jitter-s3`)", "backend": "any (CoreML/CPU/GPU)",
     "trained_on": V2_BEECH, "scored_on": "BeechScale666 test (Campus + Campus_Oberheide, 400 tiles / 1600 eval-tiling windows)",
     "reported": "0.7872", "notes": "best of 3 seeds. Beats the UNet 3/3 seeds under the training protocol, but loses to it under the plain-resize re-score — see the note below"},
    {"asset": "model_HRNet_Beech4Site_512_jitter_fp16.onnx", "src": "model_HRNet_Beech4Site_512_jitter_fp16.onnx",
     "group": "HRNet — four-site beech corpus (`scale-aug-20260813/jitter-s3`)", "backend": "GPU",
     "trained_on": V2_BEECH, "scored_on": "BeechScale666 test (Campus + Campus_Oberheide, 400 tiles / 1600 eval-tiling windows)",
     "reported": "0.7872", "notes": "post-training fp16 (Tensor-Core). No int8: the smp decoder's symbolic "
                                    "shapes fail ORT static quantisation (`Incomplete symbolic shape inference`)"},

    {"asset": "model_UNet_Beech4Site_512_jitter.onnx", "src": "model_UNet_Beech4Site_512_jitter.onnx",
     "group": "UNet — four-site beech corpus (`scale-aug-unet-20260813/jitter-s1`)", "backend": "any (CoreML/CPU/GPU)",
     "trained_on": V2_BEECH, "scored_on": "BeechScale666 test (Campus + Campus_Oberheide, 400 tiles / 1600 eval-tiling windows)",
     "reported": "0.7831", "notes": "best of 3 seeds"},
    {"asset": "model_UNet_Beech4Site_512_jitter_fp16.onnx", "src": "model_UNet_Beech4Site_512_jitter_fp16.onnx",
     "group": "UNet — four-site beech corpus (`scale-aug-unet-20260813/jitter-s1`)", "backend": "GPU",
     "trained_on": V2_BEECH, "scored_on": "BeechScale666 test (Campus + Campus_Oberheide, 400 tiles / 1600 eval-tiling windows)",
     "reported": "0.7831", "notes": "post-training fp16"},
    {"asset": "model_UNet_Beech4Site_512_jitter_int8.onnx", "src": "model_UNet_Beech4Site_512_jitter_int8.onnx",
     "group": "UNet — four-site beech corpus (`scale-aug-unet-20260813/jitter-s1`)", "backend": "CPU",
     "trained_on": V2_BEECH, "scored_on": "BeechScale666 test (Campus + Campus_Oberheide, 400 tiles / 1600 eval-tiling windows)",
     "reported": "0.7831", "notes": "post-training static int8, calibrated on 128 corpus tiles"},

    {"asset": "model_UNet_TegelR12R13_512_scratch.onnx", "src": "model_UNet_TegelR12R13_512_scratch.onnx",
     "group": "UNet — Tegel R12/R13, from scratch (`tegel-293/s3`)", "backend": "any (CoreML/CPU/GPU)",
     "trained_on": V2_TEGEL, "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7770", "notes": "best of 3 seeds; trained only on Tegel, no beech initialisation"},
    {"asset": "model_UNet_TegelR12R13_512_scratch_fp16.onnx", "src": "model_UNet_TegelR12R13_512_scratch_fp16.onnx",
     "group": "UNet — Tegel R12/R13, from scratch (`tegel-293/s3`)", "backend": "GPU",
     "trained_on": V2_TEGEL, "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7770", "notes": "post-training fp16"},
    {"asset": "model_UNet_TegelR12R13_512_scratch_int8.onnx", "src": "model_UNet_TegelR12R13_512_scratch_int8.onnx",
     "group": "UNet — Tegel R12/R13, from scratch (`tegel-293/s3`)", "backend": "CPU",
     "trained_on": V2_TEGEL, "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7770", "notes": "post-training static int8, calibrated on 128 Tegel tiles"},

    {"asset": "model_UNet_TegelR12R13_512_finetune.onnx", "src": "model_UNet_TegelR12R13_512_finetune.onnx",
     "group": "UNet — Tegel R12/R13, fine-tuned from beech (`tegel-finetune/s3`)", "backend": "any (CoreML/CPU/GPU)",
     "trained_on": V2_TEGEL + ", initialised from the beech UNet", 
     "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7890", "notes": "best of 3 seeds. Across all three seeds fine-tuning is worth "
                                    "+0.4 F1 over scratch on R12-P3 — inside seed spread"},
    {"asset": "model_UNet_TegelR12R13_512_finetune_fp16.onnx", "src": "model_UNet_TegelR12R13_512_finetune_fp16.onnx",
     "group": "UNet — Tegel R12/R13, fine-tuned from beech (`tegel-finetune/s3`)", "backend": "GPU",
     "trained_on": V2_TEGEL + ", initialised from the beech UNet",
     "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7890", "notes": "post-training fp16"},
    {"asset": "model_UNet_TegelR12R13_512_finetune_int8.onnx", "src": "model_UNet_TegelR12R13_512_finetune_int8.onnx",
     "group": "UNet — Tegel R12/R13, fine-tuned from beech (`tegel-finetune/s3`)", "backend": "CPU",
     "trained_on": V2_TEGEL + ", initialised from the beech UNet",
     "scored_on": "Tegel666 test = the frozen plots R12-P3 + R13-P2 (1000 tiles / 4000 eval-tiling windows)",
     "reported": "0.7890", "notes": "post-training static int8, calibrated on 128 Tegel tiles"},
]
for _m in V2_MODELS:
    _m["base"] = "v2"
    _m["validate"] = True
    _m["eval_json"] = _m["src"].replace(".onnx", ".eval.json")


def build_manifest_v2(tag, entries):
    """Manifest for models-v2. Every number is read from a run's test_results.md (hard-coded
    verbatim above) or from an eval_onnx JSON staged beside the model; nothing is computed here."""
    lines = [
        f"# WINMOL tree-stem segmentation models — `{tag}`", "",
        "Trained on data that `models-v1` did not have: the **four-site beech corpus** "
        "(Campus, Campus_Oberheide, Bachsee_north, Kaufland — the GIS annotation set) with ±30% "
        "scale jitter, and the **Tegel R12/R13** survey (July 2025, five digitised sample plots).",
        "",
        "**ONNX contract** (unchanged from v1): input `[batch,3,512,512]` float32 in [0,1] NCHW, "
        "output `[batch,1,512,512]` (sigmoid baked in, opset 17, dynamic batch). Loads unchanged in "
        "`winmol_unet.runtime.OnnxSegmenter` and in the WINMOL Analyzer. No suffix = fp32 (use on "
        "macOS/CoreML), `_fp16` = GPU (Tensor-Core), `_int8` = CPU (static, calibrated) — both "
        "**post-training**, no retraining.",
        "",
        "### Two F1 columns, because they are two different exams",
        "",
        "- **train-time F1** — copied verbatim from the run's `test_results.md`, on that arm's own "
        "held-out test set with the eval-tiling crop protocol training used (4 windows per tile).",
        "- **ONNX F1** — `scripts/eval_onnx.py`, plain 512 resize of each test tile, the same code for "
        "fp32/fp16/int8. Read it *within* a group to see the quantisation cost; the absolute value is "
        "lower than train-time F1 because it is a different protocol, not because the model is worse.",
        "",
        "**Never compare F1 across the two families.** The beech and Tegel models are scored on "
        "different ground; a beech model's 0.79 and a Tegel model's 0.78 are not the same exam. "
        "Per-plot, world-space numbers (the only cross-arm comparison that means anything) are in "
        "[`docs/tegel-r12-r13-results.md`](../blob/main/docs/tegel-r12-r13-results.md).",
    ]
    groups = []
    for e in entries:
        if e["group"] not in groups:
            groups.append(e["group"])
    for g in groups:
        members = [e for e in entries if e["group"] == g]
        lines += ["", f"## {g}", "",
                  f"*Trained on:* {members[0]['trained_on']}  ",
                  f"*Scored on:* {members[0]['scored_on']}", "",
                  "| asset | backend | size (MB) | train-time F1 | ONNX F1 | notes | sha256 |",
                  "|-------|---------|----------:|--------------:|--------:|-------|--------|"]
        for e in members:
            ev = e.get("eval", {})
            f1 = f"{ev['f1']:.4f}" if ev.get("f1") is not None else "—"
            lines.append(f"| `{e['asset']}` | {e['backend']} | {e.get('size_mb', 0):.1f} | "
                         f"{e['reported']} | {f1} | {e['notes']} | `{e['sha256'][:16]}…` |")
    lines += ["",
              "**Which to use.** Beech / mixed European windthrow at the default 15 m tile size: the two "
              "architectures disagree depending on the exam. HRNet wins 3/3 seeds under the training "
              "(eval-tiling) protocol; the UNet wins under the plain 512-resize re-score (0.7681 vs 0.7611), which is what "
              "the Analyzer actually serves. That second comparison is one model per architecture — the "
              "released seed — not a seed-paired result, so treat it as indicative. On that basis, prefer "
              "`model_UNet_Beech4Site_512_jitter.onnx` for Analyzer use and "
              "`model_HRNet_Beech4Site_512_jitter.onnx` if you tile with crop windows. "
              "The gap is ~0.5 F1 either way. Tegel-like stands, or any survey "
              "close to R12/R13: the `TegelR12R13` UNets — but note the beech models already reach "
              "**F1 0.76 zero-shot on R12-P3** with no Tegel data at all, so the Tegel models are worth "
              "+3.5 to +6.2 F1, not a step change.",
              "",
              "**Scale is a knob, not a property of your orthomosaic.** The Analyzer cuts "
              "`ceil(tile_size / pixel_size)` px and resizes to 512, so effective ground resolution is "
              "`tile_size / 512` — 2.93 cm/px at the default 15 m, which is what every model here was "
              "trained at. These models carry ±30% scale-jitter augmentation, which flattens but does "
              "not remove that dependence.",
              "",
              "Full SHA256 in `SHA256SUMS`. Provenance, splits and per-seed numbers: "
              "`docs/scale-augmentation-results.md` and `docs/tegel-r12-r13-results.md`."]
    return "\n".join(lines) + "\n"


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


def stage(models, archive_dir, optimised_dir, keras_dir, staging_dir, v2_dir=None):
    """Copy each source into staging under its asset name; return entries + hash/size.

    For v2 entries an `eval_json` written by scripts/eval_onnx.py is read in verbatim next to
    the model, so the manifest quotes measured numbers instead of recomputing any."""
    os.makedirs(staging_dir, exist_ok=True)
    bases = {"archive": archive_dir, "optimised": optimised_dir, "keras": keras_dir,
             "zenodo": os.path.join(archive_dir, "zenodo_analyzer"), "v2": v2_dir}
    out = []
    for m in models:
        base = bases[m["base"]]
        src = os.path.join(base, m["src"])
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing model source: {src}")
        dst = os.path.join(staging_dir, m["asset"])
        shutil.copyfile(src, dst)
        if m.get("validate"):
            # the whole point of the release is that every asset loads unchanged in the
            # Analyzer -- check the frozen contract here rather than after someone downloads it
            import onnx
            from winmol_unet.contract import validate_onnx_model
            validate_onnx_model(onnx.load(dst))
        entry = {**m, "path": dst, "sha256": sha256_file(dst),
                 "size_mb": os.path.getsize(dst) / 1e6}
        if m.get("eval_json"):
            ej = os.path.join(base, m["eval_json"])
            if os.path.isfile(ej):
                with open(ej) as f:
                    entry["eval"] = json.load(f)
            else:
                print(f"  note: no eval JSON for {m['asset']} ({ej})")
                entry["eval"] = {}
        out.append(entry)
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


_DEFAULT_STAGING = "results/cpu_speedup/release_staging"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="cwinkelmann/WINMOL_segmentor_pt")
    ap.add_argument("--set", dest="model_set", choices=["v1", "v2"], default="v1",
                    help="v1 = original + optimised (models-v1); v2 = four-site beech + Tegel R12/R13")
    ap.add_argument("--v2-dir", default="results/release_v2",
                    help="staged carrot outputs for --set v2 (models + *.eval.json)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--archive-dir", default="/data/mnt/storage/hnee/WINMOL/models")
    ap.add_argument("--optimised-dir", default="results/cpu_speedup/models")
    ap.add_argument("--keras-dir", default="results/cpu_speedup/keras_onnx")
    ap.add_argument("--staging-dir", default=_DEFAULT_STAGING)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.model_set == "v2":
        models, manifest_fn = V2_MODELS, build_manifest_v2
        tag = args.tag or "models-v2"
        title = args.title or "WINMOL models v2 — four-site beech corpus + Tegel R12/R13"
        staging_dir = args.staging_dir if args.staging_dir != _DEFAULT_STAGING \
            else "results/release_v2_staging"
    else:
        models, manifest_fn = RELEASE_MODELS, build_manifest
        tag = args.tag or "models-v1"
        title = args.title or "WINMOL models v1 — original + CPU/GPU-optimised"
        staging_dir = args.staging_dir

    entries = stage(models, args.archive_dir, args.optimised_dir, args.keras_dir,
                    staging_dir, v2_dir=args.v2_dir)
    # checksums + manifest
    with open(os.path.join(staging_dir, "SHA256SUMS"), "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['asset']}\n")
    with open(os.path.join(staging_dir, "manifest.md"), "w") as f:
        f.write(manifest_fn(tag, entries))

    total = sum(e["size_mb"] for e in entries)
    print(f"staged {len(entries)} assets ({total:.0f} MB) in {staging_dir}:")
    for e in entries:
        print(f"  {e['asset']:<44} {e['size_mb']:6.1f} MB  {e['backend']:<20} {e['sha256'][:12]}")
    publish(tag, title, args.repo, entries, staging_dir, args.dry_run)
    print("done." if not args.dry_run else "dry-run complete (nothing uploaded).")


if __name__ == "__main__":
    main()

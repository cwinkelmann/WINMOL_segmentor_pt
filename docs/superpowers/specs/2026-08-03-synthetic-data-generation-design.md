# Synthetic Training-Data Generation — Design (v1)

**Date:** 2026-08-03
**Status:** Approved (brainstorm with Christian)
**Branch:** `feat/synthetic-data-generation` (off `main`)
**Conceptual source:** Fakhri et al., *Using a synthetic 3D-environment and model-guided
scene refinement to train a deep learning model for mapping fine woody debris from
RGB-imagery*, Forestry 99(4), cpag050.

## What the paper does, and what we take

Their pipeline: a 3ds Max + Forest Pack forest floor (procedural soil/litter/stones,
photogrammetry-scanned debris), procedurally varied debris objects, a photorealistic RGB
render plus a black/white render for masks, multi-view cameras, and — the load-bearing
part — five generations of **model-guided scene refinement**: train, inspect failures on
*real* imagery, add exactly those conditions to the next scene generation. Real-world F1
rose 0.55 → 0.80 (IoU 0.38 → 0.67) across generations while synthetic-set F1 stayed ~0.95.

**Taken:** ecologically-structured scenes over chaotic domain randomization; automatic
pixel-perfect masks from the renderer; deliberate variance in lighting, clutter, occlusion
and object morphology; the refinement principle (v2 here).

**Not taken:** the commercial GUI stack (3ds Max/Forest Pack), photogrammetry scanning as
a prerequisite, multi-view oblique cameras (WINMOL consumes nadir orthomosaics), and
"synthetic only" training — see Goal.

## Goal and success criterion

Synthetic tiles serve as the **stage-1 (GenDS) pretraining set** in the existing two-stage
pipeline; the stage-2 SpecDS fine-tune stays real. Success is a single A/B on existing
rails, same seed, both scored on the real held-out TestDS:

```bash
python -m training.run_train --gen-data-dir <SYNTH> --spec-data-dir <SpecDS> \
  --test-data-dir <TestDS> ...      # vs. --gen-data-dir <GenDS> ...
```

Synthetic pretraining wins if TestDS F1 ≥ the GenDS-pretrained baseline. No new training
code is required — `run_two_stage` already does this.

## Architecture

Dev-only package `synthgen/`, never imported by `winmol_unet` (same boundary as
`training/`; excluded from the wheel). Exactly one module touches Blender.

| Unit | Responsibility | Depends on |
|---|---|---|
| `synthgen/scene_spec.py` | Frozen, JSON-serializable dataclasses describing one scene: ground, stems (pose + shape), clutter, sun, camera, GSD | stdlib |
| `synthgen/sampler.py` | Seeded sampling of a `SceneSpec` from a `GenConfig` of distributions | numpy |
| `synthgen/render_bpy.py` | Builds geometry from a spec and renders tile + mask (+ depth). The only `import bpy` in the repo | bpy |
| `scripts/generate_synthetic_dataset.py` | Orchestrator: sample N specs → render each in a subprocess → write a loader-ready dataset dir | stdlib |

`SceneSpec` is the engine-agnostic contract: it makes the sampler testable without Blender,
and it is the object the v2 refinement loop mutates. It is data, never behavior.

**Blender is a pip module, not an application.** `bpy` installs into the T14's Python 3.11
env (`pip install bpy`, 4.5.x/5.0.x) — no GUI, no separate binary. The Mac dev env
(Python 3.9) cannot install it, so rendering happens on the T14 while the Mac runs the
pure-Python core and its tests. The orchestrator shells out per scene
(`python -m synthgen.render_bpy spec.json out/`) so a renderer crash loses one tile, not
the run.

## Rendering

- **One render pass, not two.** Stems carry `pass_index=1`; the mask is Blender's
  object-index pass taken through the compositor. Co-registration holds by construction —
  there is no second render to drift out of alignment, and it costs less than the paper's
  dual-pass scheme.
- **Depth for free.** The Z-pass is written as `depth/depth{N}.png` (uint16, per-tile
  min-max) in the convention the RGBD feature already consumes. Synthetic data is the one
  place RGBD training gets *ground-truth* depth rather than mask-derived depth.
- **Scale is metric.** The config states a target GSD (m/px) and stem diameters in metres;
  camera altitude and focal length are solved to hit that GSD at 512×512. Without this,
  synthetic stems land at the wrong apparent size and nothing transfers.
- Cycles with GPU (`CUDA`) on the T14, a low sample count (denoised) — these are training
  tiles, not hero renders.

## Scene content (the variance axes)

- **Stems:** procedural — curve backbone with random bend, tapered radius, broken/split
  ends, root plate, branch stubs; bark shader with per-stem hue/roughness/darkness jitter.
  Lengths and diameters sampled in metres from config ranges.
- **Arrangement:** random position/orientation; explicit crossing and piling (stems resting
  on one another); partial burial in the ground plane.
- **Ground:** plane textured with crops from real orthomosaics (random rotation/flip), with
  procedural fallback (noise-based soil/litter/moss) when no crops are supplied.
- **Clutter/occlusion:** scattered foliage, branches and stones, some deliberately overlying
  stems — the paper attributes its false-negative reduction to exactly this.
- **Light:** sun elevation/azimuth ranges + off-scene canopy proxies casting dappled shadow.

## Error handling

- A scene that fails to render is logged and skipped; the orchestrator reports how many
  tiles were dropped rather than silently emitting a short dataset.
- Missing/empty ground-crop dir → procedural ground, with a warning (never a hard failure).
- A rendered mask whose stem-pixel fraction falls outside `[0.001, 0.6]` is rejected as
  degenerate (all-background or all-stem) and counted in the drop report.

## Testing

Everything except `render_bpy.py` is hermetic and unit-tested on the Mac:

1. **Sampler determinism** — same seed ⇒ identical `SceneSpec`; different seed ⇒ different.
2. **Golden spec** — a fixed seed's serialized spec matches a committed golden JSON, so a
   silent change to any distribution is caught.
3. **Config invariants** — sampled values honour configured ranges (counts, lengths,
   diameters, sun elevation); stems stay inside the tile footprint.
4. **Round-trip** — `SceneSpec` → JSON → `SceneSpec` is lossless (the orchestrator/renderer
   boundary depends on it).
5. **Renderer smoke test** — guarded by `pytest.importorskip("bpy")`: renders one small
   scene and asserts tile/mask/depth exist, mask is binary, all three are co-registered at
   the same size, and the stem-pixel fraction is plausible. Skipped where bpy is absent
   (the Mac, CI) exactly as the repo already guards optional-dependency tests.

## Out of scope (v1)

- The model-guided refinement loop — v2, and the reason `SceneSpec` is plain data.
- Oblique/multi-view cameras, seasonal or multispectral variation, instance masks.
- Photogrammetry-scanned assets (a later fidelity upgrade, not a prerequisite).

# Depth Simulator Design — Synthetic Terrain Depth from Stem Masks

**Date:** 2026-07-29
**Status:** Approved (brainstorm with Christian)
**Relates to:** `docs/superpowers/plans/2026-07-29-rgbd-input.md` (RGBD input feature), `docs/FEATURES.md` "4 D input data" + "Training data simulator" backlog items.

## Purpose

Generate synthetic depth maps (`depth/depth{N}.png`, 16-bit) from the segmentation
training masks (`mask/mask{N}.gif`) that datasets already contain, so the planned
`--rgbd` (4-channel) pipeline has training data before any real depth source exists.

This is explicitly a **plumbing + pretraining** data source, not evaluation-grade
data: it validates the RGBD path end-to-end and may improve segmentation via a
plausible depth prior, but results on it do not predict real-sensor performance.

## Non-goals

- Photogrammetric realism (no ray casting, no RGB-conditioned synthesis).
- On-the-fly synthesis inside `StemDataset` (offline script only — reproducible,
  inspectable, zero coupling into the loader).
- Real depth ingestion (GeoTIFF DSM/CHM conversion stays a `build_dataset.py`
  follow-up, out of scope here and in the RGBD plan).

## Deliverable

`scripts/simulate_depth.py` — offline CLI:

```
python scripts/simulate_depth.py --dataset <DS> [--seed 1] [--stem-drop-p 0.2] \
    [--distractors 3] [--overwrite]
```

- Reads every `<DS>/mask/mask{N}.gif` (integer-N pairing, same rule as
  `training/dataset.py::_index_by_n`).
- Writes `<DS>/depth/depth{N}.png` as **uint16**, at **native mask resolution**
  (the RGBD dataset code resizes and per-image min-max normalizes downstream, so
  units are arbitrary relative height).
- Refuses to write into an existing `<DS>/depth/` unless `--overwrite` is given.
- Deterministic: one master `--seed` derives a per-image seed from N, so a given
  (seed, N) pair always produces the same depth map regardless of processing order.

## Generation model (per mask)

All randomness from `numpy.random.default_rng(per_image_seed)`. Dependencies:
numpy, PIL, `scipy.ndimage` (`zoom`, `gaussian_filter`, `label`,
`distance_transform_edt`) — scipy is already a transitive dependency of
scikit-image; **no new install requirements**.

1. **Terrain:** fractal heightfield — 3–4 octaves of seeded random grids
   (coarse → fine, e.g. 4px, 16px, 64px feature scales) upsampled with
   `scipy.ndimage.zoom` and smoothed with `gaussian_filter`, amplitudes halving
   per octave. Terrain amplitude is the unit scale: stem height is the same
   order of magnitude, so depth alone cannot trivially separate ground from stem.
2. **Stem bulges:** connected components via `scipy.ndimage.label`. Per
   component: `d = distance_transform_edt(component)`, `r = d.max()` (max
   inradius ≈ stem half-width), rounded cylindrical profile
   `h = h_scale * r * sqrt(clip(2*d/r - (d/r)**2, 0, 1))` added **onto the local
   terrain** — thick stems bulge higher than thin ones, edges have realistic
   gradients (no step function).
3. **Anti-leakage imperfections** (prevents the network learning `depth ⇒ label`
   as a shortcut):
   - **Bulge dropout:** each stem component is skipped with probability
     `--stem-drop-p` (default 0.2) — simulates buried / vegetation-covered logs.
   - **Distractors:** `--distractors` (default 3, sampled 2–5) elliptical bulges
     placed off-mask with sizes in the stem-bulge range — simulates rocks, debris,
     root plates.
   - **Sensor noise:** gaussian pixel noise (σ ≈ 5% of terrain amplitude) plus a
     slight gaussian blur (σ ≈ 1px) over the final heightfield.
4. **Quantization:** min-max scale the final heightfield to the uint16 range and
   save as PNG (mode `I;16`).

## Error handling

- Missing `<DS>/mask/` → exit with a clear error naming the expected layout.
- Non-integer or non-`mask{N}.gif` files are ignored (same tolerance as the loader).
- Existing `depth/` without `--overwrite` → error listing the flag; never silently
  mixes old and new files.

## Testing (hermetic, pytest)

Synthetic masks in `tmp_path`; no network, no real data. Cases:

1. One `depth{N}.png` per `mask{N}.gif`, uint16, native mask size.
2. Determinism: same seed → byte-identical output; different seed → different.
3. Signal: mean depth on stem pixels exceeds mean depth in a dilated ring of
   nearby ground pixels (averaged over components, drop-p=0).
4. Anti-leakage: with `--stem-drop-p 1.0`, stem-pixel mean is not elevated above
   ground (no mask-correlated bulges remain); distractor bulges still present.
5. Overwrite guard: second run without `--overwrite` fails; with it, succeeds.

## Sequencing

Implemented **before** the RGBD plan (its own small plan). It only writes files
matching the already-agreed `depth{N}.png` convention, so it has zero code
dependency on the RGBD changes — and lets Christian eyeball generated depth maps
before any model change lands.

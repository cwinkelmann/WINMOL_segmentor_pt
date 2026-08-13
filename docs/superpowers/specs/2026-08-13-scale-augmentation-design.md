# Scale augmentation — experiment design

**Question.** Does jittering the training footprint ±30% make the model robust to the
ground resolution it is served at, and what does that robustness cost at the native scale?

**Why it matters.** The Analyzer's effective resolution is `tile_size / 512`, a user-set
knob. A fixed-scale model measured here loses **5.2 F1 across ±30% zoom**, asymmetrically
(−5.2 zooming out, −1.9 zooming in). The corpus itself spans 1.58–6.39 cm/px, a 4× range.
Matching training scale to serving scale was worth **+14.6 F1** — the largest single effect
in this project — so the remaining question is whether one model can cover a *range* of
scales instead of one point on it.

---

## The yardstick, fixed before training

**A five-point scale sweep, identical for both arms, cut from one source.**

The trap this avoids: building five test sets by re-sampling the orthomosaic at five
extents would change the inward buffer (`extent · √2/2`) at each scale, so each test set
would sit on *different ground*. Comparing across scales would then confound scale with
location.

Instead, one test set is cut at **666 px / 19.512 m** (native 2.9297 cm/px), and the five
scales are produced by **centre-cropping that same tile** and resizing to 512:

| crop | covers | effective GSD | ratio to native |
|---:|---:|---:|---:|
| 394 px | 11.54 m | 2.254 cm/px | 0.77× |
| 453 px | 13.27 m | 2.591 cm/px | 0.885× |
| **512 px** | **15.00 m** | **2.929 cm/px** | **1.00×** |
| 589 px | 17.25 m | 3.370 cm/px | 1.15× |
| 666 px | 19.51 m | 3.810 cm/px | 1.30× |

Every scale therefore shares tile centres, ground, and stem population. Only resolution
changes. 2.929 cm/px is exactly the Analyzer's default (`tile_size 15 / 512`), so 1.00× is
the deployment point and the flanks are what a user gets by moving the tile-size spinbox.

**Primary metric:** F1 at each of the five scales, precision and recall reported
separately. **Secondary:** the drop from each arm's own peak, which is the robustness
claim independent of absolute level.

## Arms — one difference only

Both arms train on **the same 666 px source tiles**, same ground, same count, same
rotation/flip/photometric augmentation, same optimiser, epochs, batch size and seed. The
loader's `RandomSizedCrop(min_max_height=(crop_min, crop_max)) → 512` is the only knob
that differs, and it sets effective GSD = `native_gsd × crop_px / 512`.

| arm | crop range | effective GSD seen in training |
|---|---|---|
| **A — fixed** | 512–512 px | 2.929 cm/px, constant |
| **B — jitter** | 394–666 px | 2.254 – 3.810 cm/px (±30%) |

Arm A still gets a random crop *position*, so translation augmentation is held constant
too; only the scale distribution differs. This is what makes the comparison clean — an
earlier version of this design would have trained arm A on a separate 512 px dataset,
which would have confounded scale jitter with a different tile population.

**n = 3 seeds per arm, paired**, `--deterministic`. Two runs with bit-identical gradients
measured 2.0 F1 apart on this corpus without deterministic kernels, and 0.5–1.0 with, so
unpaired single runs cannot resolve the effect sizes expected here.

## Splits

`make_splits.py --strategy blocks` over the four annotated beech sites. The 19.512 m
footprint needs a **13.80 m** inward buffer per block (up from 10.61 m at 15 m), which
shrinks usable area, so block sizes are re-derived per site and any site that cannot yield
≥12 usable blocks goes wholesale to train and is recorded in `splits.json`.

Leak-freedom is verified numerically from `tiles.jsonl`: the closest train↔val↔test centre
distance must exceed `extent · √2 = 27.60 m`, below which two rotated footprints can share
a pixel. Reported once as a number, per `docs/process.md`.

## Confounds, declared up front

- **Arm A never sees the full 666 px tile**, only 512 px windows of it. Arm B occasionally
  sees all of it. This is inherent to the manipulation — a wider crop range means a wider
  field of view — and cannot be separated from scale jitter without also changing the
  footprint.
- **F1 is not comparable across scales.** A coarser tile covers more ground and thinner
  stems; at 3.81 cm/px a 23 cm stem is 6.0 px against 10.2 px at 2.25 cm/px. Absolute F1
  will fall with coarseness for reasons unrelated to either arm. The arm-vs-arm difference
  at each scale is the readable quantity; the cross-scale trend is context.
- **Fewer tiles than `BeechAll15`.** The larger buffer costs usable AOI, so this dataset
  will be smaller than the 3,971-tile 15 m set. Both arms share it, so it does not bias the
  comparison, but it caps absolute performance.
- **Sites are shared across splits** (block strategy, not site holdout). The claim is about
  scale robustness within known forests, not transfer to a new site.

## What would falsify the hypothesis

- Arm B fails to flatten the curve — its drop from peak is no smaller than arm A's.
- Arm B costs more at 1.00× than it recovers at the flanks, making it a bad trade for a
  deployment that pins `tile_size = 15`.

Either outcome is reportable and settles whether to ship scale jitter by default.

## Deliverables

1. `configs/beech_scale666.json` and the built dataset with `tiles.jsonl` provenance.
2. Six runs (2 arms × 3 seeds), results collected verbatim to JSON.
3. `scripts/scale_sweep.py` — builds the five cropped test sets and scores any model set
   across them.
4. A paired report per scale via `scripts/paired_ablation.py`, plus one figure: F1 vs
   effective GSD, both arms, with per-seed spread.

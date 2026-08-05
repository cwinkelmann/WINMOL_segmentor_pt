# Open pull requests — what each one does

Status 2026-08-05. Six PRs are open against `main`. They fall into three groups:
**ship now** (reviewed, tested, independent), **research branches** (draft, kept for
the record), and **one stack** where merge order matters.

```
main
 ├── #15  docs: R-vs-PyTorch parity + baseline latency        docs only        READY*
 ├── #12  CPU-inference speedup (width_mult + int8)           +1585/-27, 25f   READY
 ├── #16  Depth simulator + RGBD (4-channel) input            +2074/-47, 29f   READY
 │     └── #17  Synthetic data generation (contains #16)      +4904/-47, 66f   DRAFT
 ├── #13  BAMFORESTS benchmark + multiscale loader            +1296/-50, 17f   DRAFT
 └── #14  Learned stem vectorization (distillation study)     +2552/-0,  37f   DRAFT
```
\* #15 needs one dead link fixed first.

---

## #12 — CPU-inference speedup (ready)

**What it does.** Two independent levers for faster UNet inference that keep the
frozen ONNX contract: a `width_mult` that scales every channel width (1.0 is
byte-identical to the historical ladder), and int8 quantization (static QDQ,
calibrated on real tiles). Ships a Keras→ONNX converter for the upstream Zenodo
flavours and a release-publishing script. Reported ~10× CPU speedup at unchanged F1.

**Why it is ready.** Rebased off the benchmark branch onto `main`, carrying only the
two commits it genuinely depends on (ONNX execution-provider selection and the
`onnxruntime>=1.19` floor). Full suite green: **103 passed**.

**Two real bugs were found and fixed during review**, both of which would have
shipped:
- `width_mult` values outside {1.0, 0.5, 0.25} crashed at forward time — decoder
  blocks took their input channels from the next ladder width rather than from the
  actual skip+upsample concat. The CLI accepted arbitrary floats.
- Dynamic int8 export was **unloadable**: `quantize_dynamic(QInt8)` emits
  `ConvInteger` with signed weights, which onnxruntime's CPU EP does not implement,
  so a session-load failed on the branch's own minimum onnxruntime. Switched to
  `QUInt8`. The static path behind the 10× claim was unaffected.

---

## #13 — BAMFORESTS benchmark + multiscale loader (draft, held)

**What it does.** Native-resolution multiscale training (rotate-then-crop),
deterministic grid-tiled evaluation, a site-held-out BAMFORESTS benchmark harness,
and ONNX execution-provider selection (CUDA → CoreML → CPU).

**State.** The dataloader and runtime code are well tested (7 multiscale tests, 5
provider-precedence tests). The *benchmark claims* are prose-only: the cited
`results/*.md` artifacts are not in the PR, and everything is single-seed.

**To un-draft:** smoke tests for the sitesplit and ablation scripts, and either
include or link the result artifacts.

---

## #14 — Learned stem vectorization (draft, research)

**What it does.** A self-contained mini-project distilling the analyzer's heuristic
mask→stem-graph vectorization (skeletonize → graph → angle-vote → diameter) into a
multi-head UNet predicting centreline / orientation / diameter fields, decoded back
to polylines.

**Why it stays interesting.** #17's instance-segmentation result (below) is a
negative one that points straight back here: box-based detection fails on long
diagonal stems, and this branch's field-based formulation avoids boxes entirely.

**State.** 16 hermetic unit tests; quantitative claims are prose-only because the
data and checkpoints are not committed (honestly disclosed). Two `rasterio`-dependent
tests are now `importorskip`-guarded so `pytest` is green out of the box.

---

## #15 — Parity + latency documentation (ready with one fix)

**What it does.** Records the empirical R-vs-PyTorch parity result (the port is
accuracy-neutral at matched 256², modestly ahead at the deployed 512²) and a baseline
pre-optimization latency table.

**One fix needed:** it points at `results/cpu_speedup/RESULTS.md`, which exists
nowhere; the real writeup is `docs/2026-07-21-cpu-inference-speedup-results.md` on
#12. All other "Source:" citations resolve into a gitignored `results/` directory, so
the numbers are not reproducible from the repo alone — worth a one-line note.

---

## #16 — Depth simulator + RGBD input (ready)

**What it does.** Two things that fit together:

1. `scripts/simulate_depth.py` synthesizes plausible terrain depth from existing stem
   masks — fractal terrain plus cylindrical stem bulges, deliberately degraded with
   bulge dropout, off-mask distractors and sensor noise so depth is a
   helpful-but-unreliable cue. `--mismatch` pairs each tile with the *wrong* mask as a
   leakage control.
2. `--rgbd`: channels-parameterized ONNX contract (`IN_CHANNELS` stays 3),
   `normalize_depth`, `in_channels` through the model factory and export, a
   channel-inferring `OnnxSegmenter`, optional `depth_dir` in `StemDataset`, and depth
   as a geometric-only augmentation target.

**Result.** On held-out real TestDS: RGB **0.738** → RGBD **0.892**, with the
mismatched-depth control at **0.755** ≈ RGB. That control is what makes the result
credible — the gain is genuine fusion, not the model reading the label off channel 4.

![qualitative](assets/rgbd-qualitative.png)

**Note:** #17 contains this branch (merged, not cherry-picked). Merge #16 first on its
own merits, then #17 rebases cleanly.

---

## #17 — Synthetic training data generation (draft)

**What it does.** Generates labelled training tiles from nothing but a random seed: a
`SceneSpec` → aligned labels (binary mask, per-stem instances, amodal instances with
visibility, 16-bit height field) → an SDXL **ControlNet**, trained on 3,681 real
SpecDS mask→tile pairs, paints the photorealistic RGB.

**Headline.** A segmenter trained on **only generated imagery** reaches **F1 0.693**
on real hand-labelled TestDS — **94% of the 0.738 real training data achieves**, with
zero manual annotation.

![mask to tile](assets/controlnet-mask-to-tile.png)

**What the evidence showed** (details in `docs/synthetic-data-results.md`):
- **Label fidelity beats visual realism.** The most realistic generator was the worst
  teacher (F1 0.195, recall 0.112) because it painted brash over labelled stems.
- **Prompting cannot move appearance; data can.** A ControlNet trained on one domain
  overrides the text prompt almost entirely.
- **Instance segmentation needs a box-free method.** Mask R-CNN reached only 0.079,
  and the cause was measured: a stem fills a median **12.9%** of its axis-aligned box,
  so the mask head paints a thin diagonal inside a box that is ~87% background.
- **Real depth is now supported robustly** — nodata handling, fixed physical ranges,
  and a `validate_dataset.py` preflight that reports the depth→mask AUC so leaky
  depth is caught before training rather than after.

**To un-draft:** scale past 1,200 tiles to test whether the last 0.045 closes, and
decide the #16 merge order.

---

## Suggested merge order

1. **#15** (after the dead-link fix) — docs only, no code risk.
2. **#12** — independent, tested, two real bugs already fixed.
3. **#16** — self-contained; the RGBD contract change is backwards compatible.
4. **#17** — rebase after #16 lands; keep as draft until scaled up.
5. **#13 / #14** — leave as drafts; both need their result artifacts committed.

`training/dataset.py` is the one file several branches touch (`resize=` on #13,
`depth_dir=` on #16/#17). #13 and #16 both fork from the same `main` commit and are
not ancestors of each other, so whichever merges second needs a manual reconcile
there — the parameters compose, but git will not do it automatically.

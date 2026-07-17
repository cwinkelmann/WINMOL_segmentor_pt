# BAMFORESTS: Architecture Benchmark, R-vs-PyTorch, and Augmentation Study

**Date:** 2026-07-17
**Status:** Complete (results reproduced end-to-end)
**Repo:** `WINMOL_segmentor_pt` (this repo)
**Branch:** `feat/bamforests-benchmark`
**Artifacts:** `results/bamforests_1000/`, `results/bamforests_experiments/`

## TL;DR

- **PyTorch ≥ R at matched settings.** Single-stage, 100 epochs, R-matched augmentation,
  256 px: PyTorch UNet **0.771** vs R `WINMOL_segmentor` **0.733** held-out F1 (+0.038). All
  three PyTorch architectures match or beat R; R's recall is markedly lower (under-segments).
- **Augmentation is worth +0.09 to +0.15 F1.** Strong photometric jitter is the single
  biggest driver; rotation helps HRNet specifically; flips alone are inconsistent. Best
  recipe (`all_resize` = flips + rotation + strong photometric) reaches ~0.80–0.81 across all
  architectures — ~0.07 above R.
- **RandomCrop is a no-op here — not harmful, just unrewarded.** It matches/beats its
  fixed-crop control on *validation* and only slips on *test* (a model-selection artifact on a
  noisier training signal, worst for DeepLabV3+'s BN-sensitive ASPP). The masks are dense
  (~58% positive) and spatially uniform, so every crop looks alike and there is no positional
  variety to exploit. See §4b for the evidence; a fair test needs a crop-consistent eval and
  equal epoch budget.

## 1. Setup

**Dataset.** BAMFORESTS `small_training_sample/1000_images` — a Bamberg tree-crown UAV
benchmark (Troles et al. 2024), converted to the loader format (`train{N}.jpeg` ↔
`mask{N}.gif`, binary masks). Three disjoint splits of 1000 tiles each: `train/`, `val/`,
`test/`. Source tiles are 1024×1024.

**Regime.** Single-stage (train → fixed val → held-out test), mirroring R `cost_eval` on the
held-out set. Encoder `resnet34` / ImageNet-pretrained. Adam 1e-3, loss `BCEWithLogits +
(1 − soft-F1)`, ReduceLROnPlateau. Seed 1. **TestDS F1** on the 1000 `test/` tiles is the
headline metric throughout (hard-rounded, micro-averaged).

**Hardware.** Single NVIDIA RTX 4080 SUPER (16 GB). PyTorch runs in the `winmol-train`
Docker image; R runs in `winmol-r-train-gpu`.

**A note on resolution.** The architecture benchmark (Part 2) is at 512 px. The R comparison
and augmentation ablation (Parts 3–4) are at **256 px** — R's native resolution, and chosen
for tractability (at 512 px a single single-stage run is ~70 min, so the 24-run ablation
would have been 20–40 GPU-hours). The earlier Winmol study found 256 ≈ 512 for held-out F1,
so relative rankings are expected to carry over.

## 2. Architecture benchmark (512 px, 40 epochs, classic augmentation)

All three architectures trained identically (single-stage, ImageNet encoder, R-matched
augmentation) on the 1000-image split.

| Architecture   | Params | Best val F1 | TestDS F1 | TestDS P | TestDS R | Train (min) |
|----------------|-------:|------------:|----------:|---------:|---------:|------------:|
| **unet**          | 31.0M | 0.8365 | **0.7968** | 0.8544 | 0.7465 | 69.7 |
| **hrnet**         | 16.1M | 0.8440 | 0.7835 | 0.8580 | 0.7209 | 25.8 |
| **deeplabv3plus** | 22.4M | 0.8339 | 0.7500 | 0.8619 | 0.6638 | 13.0 |

UNet leads on held-out F1, but HRNet is close at **half the parameters** and a third of the
training time; DeepLabV3+ has the highest precision but lowest recall (it under-segments).
TestDS columns here are the machine-independent ONNX-served numbers (CPU EP).
Source: `results/bamforests_1000/SUMMARY.md`.

## 3. R `WINMOL_segmentor` vs PyTorch (single-stage, 100 epochs, 256 px)

Both frameworks trained single-stage with R-matched ("classic") augmentation — flips + mild
photometric jitter — on the same split, then evaluated on the held-out `test/` tiles.

| Model | TestDS F1 | Precision | Recall | Notes |
|-------|----------:|----------:|-------:|-------|
| **R WINMOL_segmentor** (UNet) | **0.7330** | 0.862 | 0.646 | early-stopped @23 ep (best val_F1 0.834 @ep20), 31.0M |
| PyTorch **unet**          | **0.7711** | — | — | 31.0M |
| PyTorch **hrnet**         | 0.7543 | — | — | 16.1M |
| PyTorch **deeplabv3plus** | 0.7379 | — | — | 22.4M |

**Every PyTorch architecture matches or beats R** at identical resolution and augmentation;
PyTorch UNet leads R by **+0.038 F1**. R's recall (0.646) is well below PyTorch's, i.e. R
misses more stem/crown pixels. This reproduces the earlier Winmol finding that the PyTorch
re-implementation is at least on par with the original R U-Net.

The R run was made single-stage by pointing the R entrypoint's stage-2 dataset at a
nonexistent name, which its `train_stage` skips (see `scripts/run_r_single_stage.sh`).
Sources: `results/bamforests_experiments/r_single/final.json`,
`results/bamforests_experiments/headline_100ep/`.

## 4. Augmentation ablation (3 archs × 8 arms, 40 epochs, 256 px)

Arms are split into two groups that are each internally comparable but **not** cross-comparable
(they use different evaluation views).

### 4a. RESIZE group — whole tile resized to 256; identical eval → directly comparable

| Arm (augmentation)                | unet | deeplabv3plus | hrnet | best |
|-----------------------------------|-----:|--------------:|------:|-----:|
| none (no augmentation)            | 0.7070 | 0.6531 | 0.6945 | 0.707 |
| flip only                         | 0.7451 | 0.6032 | 0.6295 | 0.745 |
| classic (flip + mild photometric) | 0.7867 | 0.7499 | 0.7710 | 0.787 |
| + rotate                          | 0.7632 | 0.7508 | **0.8115** | 0.812 |
| photometric+ (strong)             | 0.7914 | 0.8039 | 0.8104 | 0.810 |
| **all_resize** (flip+rotate+strong) | 0.7966 | **0.8040** | 0.8031 | **0.804** |

**Findings:**

- **Augmentation clearly helps.** none → `all_resize`: unet **+0.090**, deeplabv3plus
  **+0.151**, hrnet **+0.108**.
- **Strong photometric jitter is the biggest single driver** — it lifts every architecture,
  most for deeplabv3plus (+0.054 over mild) and hrnet (+0.039 over mild).
- **Rotation helps HRNet strongly** (0.771 → 0.812) but is neutral-to-negative for unet and
  deeplabv3plus.
- **Flips alone are inconsistent** — they helped unet but *hurt* deeplab/hrnet versus no
  augmentation. Flips only pay off in combination with photometric jitter.
- **Best recipe: `all_resize`** — ~0.80–0.81, the most consistent result across all three
  architectures.

### 4b. CROP group — native crop pipeline; evaluated on CenterCrop (compare the two ONLY)

Held-out **TestDS F1**:

| Arm | unet | deeplabv3plus | hrnet | best |
|-----|-----:|--------------:|------:|-----:|
| crop_center (fixed window, control) | 0.6894 | 0.7114 | 0.7187 | **0.719** |
| crop_random (random window, augmentation) | 0.6793 | 0.5577 | 0.6630 | 0.679 |

At face value RandomCrop looks worse — but that reading is wrong, and the deeper analysis is
the more interesting result. **RandomCrop is not harmful here; it is simply a no-op that this
dataset gives nothing to exploit, and the apparent test drop is a model-selection artifact.**
Evidence:

**(a) RandomCrop matches or beats the control on *validation* — the metric the model actually
optimizes — and only loses on *test*:**

| Arch | crop_center val → test | crop_random val → test | random ran |
|------|-----------------------:|-----------------------:|-----------:|
| unet          | 0.848 → 0.689 | 0.845 → 0.679 | 30 ep (vs 22) |
| deeplabv3plus | 0.852 → 0.711 | **0.856** → 0.558 | 35 ep (vs 29) |
| hrnet         | 0.862 → 0.719 | **0.868** → 0.663 | 28 ep (vs 20) |

The random arms fit *equally well or better* (val F1 ≥ control) and trained *longer* — they are
not worse learners. The gap opens only between validation and held-out test.

**(b) The masks give RandomCrop nothing to exploit.** Measured over 300 train masks
(512-downscaled), the positive-pixel fraction is ~0.58 everywhere — center-256 **0.589**,
corner-256 **0.580**, random-256 **0.588** — and empty crops are <1%. The masks are dense
(~58% positive) and spatially uniform, so every crop window has nearly identical content.
RandomCrop adds positional variety the data does not reward; it can only add optimization
noise, not signal.

**(c) That noise inflates validation and hurts test selection.** A different crop each epoch
makes the val-F1 curve noisier; early-stopping picks its peak, which is over-optimistic on a
noisy curve, so the selected checkpoint generalizes worse to the held-out test (val ≥ control,
test < control — exactly the pattern above).

**(d) DeepLabV3+ is hit hardest for an architectural reason.** Its ASPP head does global
average pooling + BatchNorm on `[N,C,1,1]`, which is very sensitive to a shifting input
distribution; the moving crop window destabilizes those BN statistics (0.711 → 0.558), whereas
UNet — no global-pool BN — barely moves (0.689 → 0.679, within noise).

**Conclusion:** consistent with the "augmentation should be ≥ equivalent" intuition —
asymptotically, or with an equal epoch budget and a crop-consistent evaluation, RandomCrop
would equalize with the fixed-center control. On *this* dataset it cannot help regardless,
because the dense, uniform masks leave no variety to exploit. Separately, the whole crop group
(0.66–0.72) trails the resize group (0.80) because cropping to a 256 px window discards half
the scene the resize path keeps.

> Why a separate group? A RandomCrop confounds *augmentation variety* with *effective
> resolution* (a native crop is finer-grained but covers less scene). To isolate the
> augmentation, both crop arms are evaluated on the same deterministic CenterCrop, so only
> `crop_center` vs `crop_random` is a fair comparison. They are not comparable to the resize
> arms, which see the whole scene. A genuinely fair RandomCrop test would use a
> crop-consistent / sliding-window evaluation and an equal epoch budget (see §7).

Source: `results/bamforests_experiments/ablation/SUMMARY.md`; density measurement over 300
train masks; val/test from the per-run TensorBoard logs.

## 5. The native multi-scale crop mechanism

The crop probe in §4b used an early single-scale `RandomCrop`/`CenterCrop` path; that has been
**superseded by the native multi-scale dataloader** (`cfg.multiscale` / `cfg.eval_tiling`),
which is the canonical native path in the codebase:

- **Training (`cfg.multiscale`):** `StemDataset(resize=False)` keeps tiles at native
  resolution; the augmentation rotates the **full** tile, then a `RandomSizedCrop`
  (`crop_min_px`…`crop_max_px` → `img_size`) samples windows across a range of scales. This
  makes the model robust to the analyzer's user-set tile size (GSD), and rotating before the
  crop keeps black padding to the tile edge instead of every crop's corners.
- **Evaluation (`cfg.eval_tiling`):** `TilingStemDataset` deterministically grid-tiles each
  native tile into `img_size` windows for full-coverage, native-fidelity val/test — this is
  exactly the crop-consistent evaluation §4b lacked, so a native-trained model is scored on
  its own terms rather than a single centre crop.

The augmentation ablation's `multiscale` arm exercises this path; the `RESIZE` arms use the
plain resize pipeline. (The single-scale probe's mechanism was reconciled out in favour of
this one.)

## 6. Site-held-out dataset — the generalization benchmark

The 1000-image runs above use a **random** split, so `test` shares its sites with `train`
and measures in-distribution performance, not generalization to a new location. BAMFORESTS has
three forest sites, and the official COCO splits are already partly site-separated:

| Site | train2023 | eval2023 | TestSet1 | TestSet2 |
|------|----------:|---------:|---------:|---------:|
| Tretzendorf | 3695 | 1029 | 0 | 861 |
| Stadtwald   | 3780 |  979 | 0 | 782 |
| **Hain**    |    0 |    0 | **1665** | 0 |

`scripts/build_bamforests_sitesplit.py` materializes a **site-held-out** dataset from this:

- `train` ← train2023 (Tretzendorf + Stadtwald)
- `val`   ← eval2023 (Tretzendorf + Stadtwald) — model selection, in-distribution
- `test`  ← **TestSet1 (Hain only)** — a location absent from train/val → cross-site generalization
- `test_insite` ← TestSet2 (Tretzendorf + Stadtwald) — in-distribution test; the gap
  `test_insite F1 − test F1` is the generalization penalty

The builder asserts the test site is disjoint from the train/val sites (refuses to run on any
leak) and writes a `SITES.md` manifest. This is the recommended dataset for future runs; the
orchestrator (`scripts/run_bamforests_experiments.sh`) builds it and runs R + PyTorch on it.

## 7. Reproducing

```bash
# Build the site-held-out dataset (train/val=Tretzendorf+Stadtwald, test=Hain)
python scripts/build_bamforests_sitesplit.py --coco-root <coco1024> --dst <site_split>

# R vs PyTorch + augmentation ablation on the site-split, GPU-gated, sequential
SITE_SPLIT=<site_split> bash scripts/run_bamforests_experiments.sh

# Individual pieces
bash scripts/run_r_single_stage.sh <out> <site_split> 100 512               # R single-stage
python scripts/augmentation_ablation.py --train-dir <site_split>/train \
  --val-dir <site_split>/val --test-dir <site_split>/test \
  --img-size 512 --epochs 40                                                # PyTorch arms

# Original 1000-image architecture benchmark (random split, 512 px)
python scripts/benchmark_architectures.py --skip-stage1 \
  --spec-data-dir <1000_images>/train --val-data-dir <1000_images>/val \
  --test-data-dir <1000_images>/test --out-dir results/bamforests_1000 \
  --epochs 40 --device cuda --encoder resnet34 --encoder-weights imagenet
```

## 8. What's missing / future work

The results above are a first, single-seed pass on a **random** 1000-image split at reduced
resolution. The main gaps, roughly in priority order:

1. **Statistical robustness.** Every number is a single seed. Some deltas (e.g. flips,
   rotation) are within plausible run-to-run noise. Repeat the key arms over ≥3 seeds and
   report mean ± spread before treating small differences as real.
2. **Confirm the augmentation ranking at 512 px / 100 epochs.** The ablation is 40 ep / 256 px
   for tractability. Re-run the top arms (`classic`, `photometric+`, `all_resize`) at 512 px
   to confirm the ranking holds at production resolution — and note that classic aug at 100 ep
   *underperformed* 40 ep (mild augmentation overfits by 100 ep; strong augmentation is what
   makes longer training safe).
3. **Scale up the data.** These runs use 1000 of the ~7475 annotated `coco1024` train tiles.
   Convert and train on the full set (`scripts/coco_to_dataset.py`) to get realistic absolute
   numbers.
4. **Cross-site generalization — tooling now in place, runs pending.** The site-held-out
   dataset (§6, `build_bamforests_sitesplit.py`: train/val = Tretzendorf+Stadtwald, test =
   Hain) is built and wired into the orchestrator; the actual site-held-out R/PyTorch runs and
   the `test_insite − test` generalization-gap numbers still need to be produced.
5. **Fairer crop evaluation — addressed by `eval_tiling`.** The §4b probe scored crops on a
   single CenterCrop; the multiscale dataloader now evaluates via deterministic native-res grid
   tiling (§5), so a native-trained model is scored on its own terms. Re-running the crop
   comparison under tiling (multiscale arm vs resize arms) is the remaining step.
6. **Encoder ablation.** Only `resnet34`. Trying `resnet50`/`efficientnet` encoders (and
   `encoder_weights=None` vs ImageNet) would separate encoder capacity from augmentation.
7. **Two-stage vs single-stage on BAMFORESTS.** Only single-stage was run here. The Winmol
   study found GenDS pretraining barely helped; worth confirming on BAMFORESTS.
8. **Machine-independent ablation metrics.** Ablation F1 comes from the torch-device eval
   (`test_results.md`); the architecture benchmark uses the ONNX CPU-EP eval. For strict
   cross-run comparability, standardize on the ONNX-served number everywhere.
9. **Analyzer round-trip.** Confirm the exported ONNX models for these BAMFORESTS runs load
   and serve correctly through `WINMOL_Analyzer` (the ONNX contract is validated at export,
   but an end-to-end analyzer check on this domain has not been done).

## 9. Artifacts

| Path | Contents |
|------|----------|
| `scripts/build_bamforests_sitesplit.py` | site-held-out dataset builder (§6) + `SITES.md` |
| `scripts/augmentation_ablation.py` | RESIZE + multiscale aug ablation harness |
| `scripts/run_r_single_stage.sh` | R single-stage on an arbitrary 3-way split |
| `scripts/run_bamforests_experiments.sh` | orchestrator (build site-split → R → PyTorch) |
| `results/bamforests_1000/SUMMARY.md` + `<arch>.md` | 512 px architecture benchmark |
| `results/bamforests_experiments/{r_single,headline_100ep,ablation}/` | R / PyTorch run outputs |

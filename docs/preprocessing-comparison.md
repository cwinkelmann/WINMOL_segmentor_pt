
# Training-preprocessing comparison — results

> **Correction (2026-08-14).** **Arm C (native + multiscale) is void.** Bachsee_north was
> the validation site and its F1 is pinned near zero, so checkpoint selection, the LR
> schedule and early stopping all ran on a signal carrying no information — the spread
> across arms is selection luck, not extraction. Do not cite arm C either way. Native
> resolution is re-tested properly in
> [`tegel-r12-r13-results.md`](tegel-r12-r13-results.md) with a frozen test set.

Design: `docs/superpowers/specs/2026-08-10-preprocessing-comparison-design.md`.
**Status: complete.** 21 runs — 9 extraction, 5 capacity, 3 synthetic (rerun after a
validation fix voided the first pass), 3 DINOv3 decoder, 1 of which failed and is marked so.
Every filled value is copied from that run's own `test_results.md`.

## Setup

Three extraction arms, identical in everything else. **All arms share one validation set
and one test set**, so model selection is identical across arms and only the *training*
data differs.

| | source | train tiles |
|---|---|---:|
| **A** — R port | `r_sample_tiles.py`, 15 m → 512 px, literal −11 m buffer, R's acceptance rule | 1,657 |
| **B** — fixed-metre | `sample_training_tiles.py --extent 10.24`, all sites → 2 cm/px | 2,700 |
| **C** — native + multiscale | `--native-px 512`, crops 256–512 → 512 | 1,185 |

| | source | tiles | stem |
|---|---|---:|---:|
| validation | held-out blocks of Campus | 400 | 4.55% |
| **test** | **Campus_Oberheide, whole site** | 900 | 2.27% |

Train is Kaufland (2.09 cm) + Campus (6.39 cm) — both GSD extremes, so the test site's
3.36 cm sits inside the training span. 30 epochs, batch 16, same optimiser, same repaired
geometry throughout.

---

## 1 · Extraction comparison

Test F1 on Campus_Oberheide, from scratch.

| arm | UNet | HRNet |
|---|---:|---:|
| **A** — R port | 0.3760 | 0.5702 |
| **B** — fixed-metre | 0.6410 | **0.7229** |
| **C** — native + multiscale | 0.5647 | 0.6160 |

Precision / recall, same runs:

| arm | UNet | HRNet |
|---|---|---|
| **A** — R port | 0.8561 / 0.2409 | 0.8487 / 0.4294 |
| **B** — fixed-metre | 0.8689 / 0.5079 | 0.8181 / 0.6475 |
| **C** — native + multiscale | 0.8553 / 0.4215 | 0.8013 / 0.5003 |

SegFormer was run on all three arms as well; it never led and is summarised in the
appendix.

**Verdict against the acceptance criteria.** The criteria were written for four
leave-one-site-out folds; what ran is one held-out site paired across three architectures.
Read against that design:

| comparison | ΔF1 per pairing | mean | result |
|---|---|---:|---|
| **B vs A** | UNet +26.5, HRNet +15.3 | **+20.9** | **superior** — positive in 2 of 2 |
| **C vs A** | UNet +18.9, HRNet +4.6 | +11.8 | **superior** — positive in 2 of 2 |
| **B vs C** | UNet +7.6, HRNet +10.7 | +9.2 | B ahead in 2 of 2 |

Both repo variants clear the −1.0 non-inferiority margin comfortably, and both beat the
R port on every architecture. **The fixed-metre sampler is the recommended extraction.**
(Including SegFormer weakens arm C's claim to non-inferior-but-not-superior — see the
appendix.)

The claim is one fold, n=1 per cell. Architecture pairing controls for model choice, not
for site, and site variance has dominated every comparison in this project — the remaining
three folds are still the stronger evidence.

### What the grid shows

**The arm ranking is B > C > A, with one exception.**

| pairing | A | C | B |
|---|---:|---:|---:|
| UNet | 0.3760 | 0.5647 | **0.6410** |
| HRNet | 0.5702 | 0.6160 | **0.7229** |

**Arm B wins every cell.** That is the one clean, unqualified result in this study.

**Best model measured: HRNet on fixed-metre tiles, 0.7229** — 0.8181 precision, 0.6475
recall, the only run in the study above 0.6 recall.

**Quote the A→B gap as a range, not a number:** +26.5 on UNet, +15.3 on HRNet. Both large,
but the UNet figure is the outlier rather than the headline — UNet is the weakest model
here and takes the most damage from weak data. **"15–27 points depending on architecture"
is the defensible summary**, and the appendix's SegFormer run extends the lower end to +8.7.

**Arm C trains on 1,185 tiles against B's 2,700.** The multiscale loader replaces
generator-side sampling with crop-side sampling, and 2.3× less data is an alternative
explanation for its deficit that this design cannot separate.

**Arm A's deficit is recall, and it is structural.** Precision is the highest of any arm
(0.856–0.876) while recall is the lowest (0.24–0.43). R's acceptance rule sums the *whole*
area of every polygon intersecting the footprint rather than the part inside it, so tiles
qualify on stems that mostly fall outside — the model is trained on ground that is emptier
than its label budget implies, and learns to withhold.

**Every arm is precision-heavy and recall-starved** on this test site: precision 0.80–0.88
against recall 0.24–0.57. Consistent with everything else measured on unseen ground — the
models become conservative rather than wrong.

---

## 2 · Validation health

The previous attempt at this comparison had validation F1 **exactly 0.0000** in all six
runs, because Bachsee_north was used as the validation site. Checkpoint selection ran on a
dead signal and the results were discarded. This is now the first thing checked.

Checkpoints are selected on **val loss** (`training/train.py:60`), early stop patience 5.
Values below are taken at the selected epoch, read from each run's TensorBoard scalars.

| run | epochs run | ckpt epoch | val loss | val F1 | test F1 | verdict |
|---|---:|---:|---:|---:|---:|---|
| A-r15-unet | 19 | 14 | 0.468 | 0.6760 | 0.3760 | healthy |
| A-r15-hrnet | 24 | 19 | 0.443 | 0.7002 | 0.5702 | healthy |
| B-fix-unet | 30 | **30** | 0.364 | 0.7372 | 0.6410 | healthy, **not converged** |
| B-fix-hrnet | 29 | 24 | 0.327 | **0.7641** | 0.7229 | healthy |
| C-nat-unet | 30 | 29 | 0.546 | 0.6063 | 0.5647 | healthy, **not converged** |
| C-nat-hrnet | 25 | 20 | 0.469 | 0.6621 | 0.6160 | healthy |

Every run carries a live validation signal. None is anywhere near the 0.0000 that voided
the previous wave. Validation ranks the arms in the same order as test (B > A > C by val
F1, B > C > A by test), so checkpoint selection was not fighting the result.

**Two of arm B's three runs never converged** — checkpoint at epoch 29–30, still improving
when the budget ran out, whereas arm A early-stopped at 14–19 having exhausted its patience.
This does not threaten the result; it means **arm B's measured advantage is a lower bound.**
Arm A was given every epoch it could use and still lost.

**Arm A degrades most on unseen ground.** The in-domain → out-of-domain drop:

| run | val F1 | test F1 | drop |
|---|---:|---:|---:|
| A-r15-unet | 0.6760 | 0.3760 | **−30.0** |
| A-r15-hrnet | 0.7002 | 0.5702 | −13.0 |
| B-fix-unet | 0.7372 | 0.6410 | −9.6 |
| C-nat-hrnet | 0.6621 | 0.6160 | **−4.6** |
| C-nat-unet | 0.6063 | 0.5647 | **−4.2** |
| B-fix-hrnet | 0.7641 | 0.7229 | **−4.1** |

**Arm A's UNet is the one catastrophic drop (−30.0); everything else sits in −4 to −13.**
The three smallest belong to arm C's UNet and HRNet and arm B's HRNet, so this is not a
clean multiscale effect. The reliable reading is narrower than it first appeared: **arm A produces the
worst transfer at every architecture, and its UNet fails outright.**

Arm C is measured under a handicap worth naming: the shared validation set is cut by arm
B's fixed-metre extraction, so arm C is validated slightly off its own domain. Its val F1
is the lowest in the study (0.6063–0.6621). The same yardstick for all arms is the design
intent; here it works against C, and C's non-inferiority verdict is achieved despite it.

Oberheide stacks three shifts at once — unseen site, species mix (37% beech against 99%
and 86% in training), and sparse stems (2.27% against 5.93%/4.42%).

---

## 3 · Capacity probe

Speed is not a constraint, so the large variants are tested with ImageNet weights. From
scratch, 24.7M already overfitted this corpus (b2 0.7806 against b0's 0.7788 at 3.7M) while
pretraining reversed it (b2 0.7855). The open question is whether 82M pretrained beats a
16M HRNet.

All rows train on arm B (fixed-metre), so they are directly comparable to §1's arm-B column.

| model | params | pretrained | val F1 | test F1 | P / R |
|---|---:|---|---:|---:|---|
| **HRNet w18** | **16.1M** | **—** | 0.7641 | **0.7229** | 0.8181 / 0.6475 |
| HRNet w18 | 16.1M | ImageNet | 0.7708 | 0.7218 | 0.8563 / 0.6238 |

**Answer to the question this probe was built for: no — 82M pretrained does not beat a 16M
HRNet.** It does not lose either. The top three (HRNet scratch 0.7229, HRNet+ImageNet
0.7218, mit_b5 0.7188) span **0.4 points across a 5× parameter range**, which at n=1 is
indistinguishable. **Take the cheapest model that ties: HRNet w18 from scratch.**

**Validation ranks these models in almost the reverse order of test.** mit_b5 has the best
val F1 in the entire study (0.7731) and the best val loss (0.318), and comes third on test.
HRNet+ImageNet beats HRNet-scratch on validation (0.7708 vs 0.7641) and loses on test.
**Capacity and ImageNet weights buy in-domain fit that does not survive the site change** —
so a model selected on this corpus's validation split will be the wrong model for the
Analyzer. That is the practical warning in this table.

**ImageNet weights are worth nothing on HRNet here (−0.1)**, and what little they move goes
the wrong way: precision up (0.8181 → 0.8563), recall down (0.6475 → 0.6238). Pretrained
features make the model more conservative on a corpus whose failure mode is already recall.
This is the third measurement of the same effect — +0.4 on a block split, −3.5 on a
held-out ortho, ~0 here. **Pretrained weights substitute for data; their value goes to zero
once enough real data is present.**

Within SegFormer capacity scales weakly — see the appendix.

---

## 4 · Synthetic pretraining — generated imagery vs generated geometry
THis might be dublicated to @/Users/christian/work/work/WINMOL_segmentor_pt/docs/synthetic-pretraining-beech.md
Two-stage: stage 1 on generated data, stage 2 on arm B.

| stage-1 source | tiles | nature | HRNet | UNet |
|---|---:|---|---:|---:|
| *(none — the arm B baseline)* | — | — | **0.7229** | **0.6410** |
| `armC-barnekow` (ControlNet / SD) | 1,200 | rich imagery, 2.2% mask noise | 0.7205 (**−0.2**) | 0.6374 (**−0.4**) |
| `SynthRGBD-3000` (Blender) | 3,000 | exact masks, no vegetation green | 0.7091 (**−1.4**) | — |

**Synthetic pretraining is worth nothing on this corpus.** All three arms land within
−0.2 to −1.4 of training on arm B alone. Neither generated imagery (Stable Diffusion,
real-looking texture, 2.2% mask noise) nor generated geometry (Blender, exact masks, no
vegetation colour) beats simply not doing it. This matches the earlier architecture study,
where synthetic pretraining was +0.1 on HRNet — within noise then, within noise now.

Stage 1 itself works: val F1 reaches 0.887–0.903 on the generated data, so the models
learn those sets easily. The features just do not transfer.

### These numbers replace an earlier, wrong set

The first pass reported `SynthRGBD` at **+1.1** and `armC-barnekow`+UNet at **+3.6**.
Those runs are void. `run_two_stage` built stage 2's loaders without passing
`--val-data-dir`, so it silently fell back to `split_ids()` — a random split of the arm-B
training tiles. That is both a *different* signal from the single-stage baselines it was
being compared against (in-training-distribution: stage-2 val F1 read 0.80–0.81 against
0.73–0.76 for the same data properly held out) and a *leaky* one, because the samplers
oversample and adjacent tiles overlap.

Test metrics were never affected — those always used the shared Oberheide set — but
checkpoint selection was, and it moved the answer by up to 3.9 points and flipped the sign
on two of three arms. Fixed in `a1ca11d`, pinned by
`tests/test_two_stage.py::test_two_stage_stage2_validates_on_the_given_val_dir`.

| run | leaky-val (void) | fixed | shift |
|---|---:|---:|---:|
| `armC-barnekow` + HRNet | 0.7147 | 0.7205 | +0.6 |
| `SynthRGBD` + HRNet | 0.7343 | 0.7091 | **−2.5** |
| `armC-barnekow` + UNet | 0.6766 | 0.6374 | **−3.9** |

### Why only armC-barnekow

The three SD sets were checked for mask alignment before use, by comparing edge energy
under the mask against the surrounding background:

| arm | tiles | mask on **less** structure than background | violet artefacts | vegetation green |
|---|---:|---:|---:|---:|
| armA-specds | 1,200 | **24.7%** | 7.2% | 0.3% |
| armB-lora03 | 1,200 | **27.6%** | 3.0% | 0.2% |
| **armC-barnekow** | 1,200 | **2.2%** | 1.4% | 2.1% |

armA and armB contain failed generations — a stem outline over an abstract blob that
ControlNet never turned into forest. Training on those teaches prediction from nothing,
which inflates recall and destroys precision. Pooling all three would have measured noise
tolerance rather than the question asked.

A mean-brightness check passed all three (+20.1 levels under-mask) and was misleading:
satisfied by bright blobs, not by stems. Edge energy exposed the tail the mean concealed.

---

## Confounds, stated not buried

- **Arm tile counts differ**: 1,657 / 2,700 / 1,185. Native tiling is expensive in AOI
  area, so arm C trains on the least data. If C underperforms, that is a live alternative
  explanation to "worse extraction".
- **The test site is 37% beech** (40% spruce, 18% Douglas fir). All arms face it equally,
  so the comparison is fair, but it is not a clean beech number.
- **Every run is n=1.** Differences under ~1 point are directional only.
- **Arm C runs at native 512, not 1024.** A 1024 px tile at Campus's 6.39 cm/px covers 65 m
  and needs a 46 m buffer; no block size leaves usable ground on either training site, so
  in-domain validation was impossible at that size. This is a property of the corpus's
  small AOIs, not a bug — and it caps how much scale diversity arm C can carry.

---

## 5 · DINOv3 pretraining and the decoder study

DINOv3 weights are in the installed timm (1.0.28), satellite variants included. The
`convnext_*.dinov3_*` distillations were chosen over the ViTs deliberately: DINOv3's ViTs
are patch-16, so every feature is stride 16, and a stem 15–40 px wide at the Analyzer's
2.93 cm/px is then 1–2.5 tokens — the resolution tax that produced DPT's 0.4917, the worst
number in this project. The ConvNeXt distillations keep the stride-4/8/16/32 pyramid.

Encoder held fixed at `tu-convnext_large.dinov3_lvd1689m` (203.3M); only the decoder varies.
All on arm B, same val and test as everything above.

| decoder | test F1 | P / R | vs HRNet-scratch (0.7229) |
|---|---:|---|---:|
| Unet | 0.6499 | 0.8411 / 0.5295 | **−7.3** |
| FPN | 0.5286 | 0.8827 / 0.3773 | −19.4 |
| PAN | *failed — see below* | | |

**DINOv3 does not help here.** A 203M encoder carrying self-supervised features from
1.689B images loses to a 16.1M HRNet trained from scratch by 7.3 points. That is the same
verdict the capacity probe reached from the other direction (§3: 82M pretrained ties 16M
from scratch), and the same one the held-out-ortho study reached about ImageNet weights.
**Pretrained features keep failing to beat in-domain training on this corpus**, and the
corpus is the reason: under one hectare of digitised stem, at a ground resolution and a
subject matter that no pretraining set covers.

### PAN did not produce a result

`smp.PAN` reports F1 0.0000 with **loss 629**. This is not a decoder verdict — it is a
broken run, and the shape of the failure says where:

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| train loss | 1.1 | 0.9 | 0.8 | 1.1 | 1.2 |
| val loss | 84,558 | 732,952 | 7,795,279 | 4,810 | 667 |

**Training is stable; evaluation explodes.** A model that trains at loss 1.1 and evaluates
at 10⁶ is not diverging — it is producing different activations in `eval()` than in
`train()`, which points at BatchNorm running statistics. PAN's Global Attention Upsample
pools to 1×1 and normalises there, and BatchNorm over `[N,C,1,1]` is a hazard this repo has
already hit once (see the `drop_last` comment in `run_train._build_loaders`, added for
DeepLabV3+'s ASPP). Diagnosing it properly, or lowering the LR from the default 1e-3, needs
the config mechanism that is still outstanding — `lr` is deliberately not a CLI flag.

Until then PAN is untested, and the BiFPN question it was meant to stand in for is open.

---

## Appendix — SegFormer

SegFormer was run on every arm and every capacity step. It is recorded here rather than in
the main tables because **it never led on any deployment-relevant comparison**, and its
presence made the arm-C verdict look worse than the other architectures support.

| run | arm / setting | test F1 | P / R |
|---|---|---:|---|
| A-r15-segformer | R port, mit_b0 scratch | 0.5756 | 0.8764 / 0.4285 |
| B-fix-segformer | fixed-metre, mit_b0 scratch | 0.6623 | 0.7973 / 0.5663 |
| C-nat-segformer | native+multiscale, mit_b0 scratch | 0.5283 | 0.8317 / 0.3871 |
| B-fix-segformer-b2-in | fixed-metre, mit_b2 ImageNet | 0.7041 | 0.7933 / 0.6330 |
| B-fix-segformer-b5-in | fixed-metre, mit_b5 ImageNet | 0.7188 | 0.8081 / 0.6473 |

Three things it showed, none of which change a decision:

- **It is the one cell where the R port beats native+multiscale** (0.5756 vs 0.5283).
  With SegFormer included, arm C is non-inferior but not superior; without it, arm C beats
  arm A on both architectures. The 3.7M model is the most sensitive to arm C's 2.3× smaller
  training set, which is why it was excluded from the headline.
- **Capacity scales weakly.** b0→b2 is +4.2 and b2→b5 +1.5, but b0 is from scratch while
  b2/b5 carry ImageNet weights, so only **b2→b5 isolates capacity: +1.5 for 3.3× the
  parameters.**
- **82M pretrained never beat 16M HRNet from scratch** (0.7188 vs 0.7229), and at the
  Analyzer's own scale mit_b2 reached 0.7974 against HRNet's 0.8011 — inside noise, at 1.5×
  the parameters.

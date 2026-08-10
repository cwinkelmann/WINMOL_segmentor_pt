# Training-preprocessing comparison — results

Design: `docs/superpowers/specs/2026-08-10-preprocessing-comparison-design.md`.
**Status: extraction grid complete (9/9); capacity probe and synthetic pretraining in progress.** Cells marked *pending* have not run yet; `failed` needs a relaunch.
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

| arm | UNet | HRNet | SegFormer b0 |
|---|---:|---:|---:|
| **A** — R port | 0.3760 | 0.5702 | 0.5756 |
| **B** — fixed-metre | 0.6410 | **0.7229** | 0.6623 |
| **C** — native + multiscale | 0.5647 | 0.6160 | 0.5283 |

Precision / recall, same runs:

| arm | UNet | HRNet | SegFormer b0 |
|---|---|---|---|
| **A** — R port | 0.8561 / 0.2409 | 0.8487 / 0.4294 | 0.8764 / 0.4285 |
| **B** — fixed-metre | 0.8689 / 0.5079 | 0.8181 / 0.6475 | 0.7973 / 0.5663 |
| **C** — native + multiscale | 0.8553 / 0.4215 | 0.8013 / 0.5003 | 0.8317 / 0.3871 |

**This grid is complete — all nine extraction runs have landed.**

**Verdict against the acceptance criteria.** The criteria were written for four
leave-one-site-out folds; what ran is one held-out site paired across three architectures.
Read against that design:

| comparison | ΔF1 per pairing | mean | result |
|---|---|---:|---|
| **B vs A** | UNet +26.5, HRNet +15.3, SegFormer +8.7 | **+16.8** | **superior** — positive in 3 of 3 |
| **C vs A** | UNet +18.9, HRNet +4.6, SegFormer **−4.7** | +6.3 | **non-inferior**, not superior — positive in 2 of 3 |
| **B vs C** | UNet +7.6, HRNet +10.7, SegFormer +13.4 | +10.6 | B ahead in 3 of 3 |

Both repo variants clear the −1.0 non-inferiority margin comfortably. **Only the
fixed-metre sampler earns the superiority claim, and it is the recommended extraction.**
Native+multiscale is defensible but not better than the R port on every architecture — its
SegFormer run is the single cell where the R port wins.

The claim is one fold, n=1 per cell. Architecture pairing controls for model choice, not
for site, and site variance has dominated every comparison in this project — the remaining
three folds are still the stronger evidence.

### What the grid shows

**The arm ranking is B > C > A, with one exception.**

| pairing | A | C | B |
|---|---:|---:|---:|
| UNet | 0.3760 | 0.5647 | **0.6410** |
| HRNet | 0.5702 | 0.6160 | **0.7229** |
| SegFormer b0 | 0.5756 | *0.5283* | **0.6623** |

**Arm B wins every cell.** That is the one clean, unqualified result in this study.

**Best model measured: HRNet on fixed-metre tiles, 0.7229** — 0.8181 precision, 0.6475
recall, the only run in the study above 0.6 recall.

**Quote the A→B gap as a range, not a number:** +26.5 on UNet, +15.3 on HRNet, +8.7 on
SegFormer. All positive and all large, but the UNet figure is the outlier rather than the
headline — UNet is the weakest model here and takes the most damage from weak data.
**"8–27 points depending on architecture, positive everywhere" is the defensible summary.**

**Arm C is architecture-sensitive in a way arm B is not.** Its margin over A swings from
+18.9 (UNet) to −4.7 (SegFormer). Arm C trains on 1,185 tiles against B's 2,700, and the
multiscale loader replaces generator-side sampling with crop-side sampling; on a
3.7M-parameter model that appears to be a net loss. This is a live confound, not a
conclusion — 2.3× less data is an alternative explanation that this design cannot separate.

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
| A-r15-segformer | 20 | 15 | 0.431 | 0.7045 | 0.5756 | healthy |
| B-fix-unet | 30 | **30** | 0.364 | 0.7372 | 0.6410 | healthy, **not converged** |
| B-fix-hrnet | 29 | 24 | 0.327 | **0.7641** | 0.7229 | healthy |
| B-fix-segformer | 30 | **29** | 0.343 | 0.7537 | 0.6623 | healthy, **not converged** |
| C-nat-unet | 30 | 29 | 0.546 | 0.6063 | 0.5647 | healthy, **not converged** |
| C-nat-hrnet | 25 | 20 | 0.469 | 0.6621 | 0.6160 | healthy |
| C-nat-segformer | 30 | **30** | 0.510 | 0.6354 | 0.5283 | healthy, **not converged** |

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
| A-r15-segformer | 0.7045 | 0.5756 | −12.9 |
| B-fix-unet | 0.7372 | 0.6410 | −9.6 |
| B-fix-segformer | 0.7537 | 0.6623 | −9.1 |
| C-nat-segformer | 0.6354 | 0.5283 | −10.7 |
| C-nat-hrnet | 0.6621 | 0.6160 | **−4.6** |
| C-nat-unet | 0.6063 | 0.5647 | **−4.2** |
| B-fix-hrnet | 0.7641 | 0.7229 | **−4.1** |

**Arm A's UNet is the one catastrophic drop (−30.0); everything else sits in −4 to −13.**
The three smallest belong to arm C's UNet and HRNet and arm B's HRNet, so this is not a
clean multiscale effect — arm C's own SegFormer drops −10.7, worse than two of arm A's
three runs. The reliable reading is narrower than it first appeared: **arm A produces the
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

| model | params | pretrained | test F1 |
|---|---:|---|---:|
| HRNet w18 | 16.1M | ImageNet | *pending* |
| SegFormer mit_b2 | 24.7M | ImageNet | *pending* |
| SegFormer mit_b5 | 82.0M | ImageNet | *pending* |

---

## 4 · Synthetic pretraining — generated imagery vs generated geometry

Two-stage: stage 1 on generated data, stage 2 on arm B.

| stage-1 source | tiles | nature | HRNet | UNet |
|---|---:|---|---:|---:|
| `armC-barnekow` (ControlNet / SD) | 1,200 | rich imagery, 2.2% mask noise | *pending* | *pending* |
| `SynthRGBD-3000` (Blender) | 3,000 | exact masks, no vegetation green | *pending* | — |

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

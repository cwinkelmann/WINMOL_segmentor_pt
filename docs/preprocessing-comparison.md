# Training-preprocessing comparison — results

Design: `docs/superpowers/specs/2026-08-10-preprocessing-comparison-design.md`.
**Status: in progress.** Cells marked *pending* have not run yet; `failed` needs a relaunch.
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
| **A** — R port | 0.3760 | 0.5702 | *failed — relaunch* |
| **B** — fixed-metre | **0.6410** | *pending* | *pending* |
| **C** — native + multiscale | *pending* | *pending* | *pending* |

Precision / recall, same runs:

| arm | UNet | HRNet | SegFormer b0 |
|---|---|---|---|
| **A** — R port | 0.8561 / 0.2409 | 0.8487 / 0.4294 | *failed* |
| **B** — fixed-metre | 0.8689 / 0.5079 | *pending* | *pending* |
| **C** — native + multiscale | *pending* | *pending* | *pending* |

**Verdict against the acceptance criteria:** *pending — needs arms B and C complete.*

### What is visible so far

**Fixed-metre beats the R port on UNet by 26.5 points** (0.6410 vs 0.3760), and the whole
difference is recall (0.5079 vs 0.2409) at nearly identical precision (0.869 vs 0.856).
One architecture, one run — directional only until arm B's HRNet lands.

**Every arm is precision-heavy and recall-starved** on this test site: precision ~0.85
against recall 0.24–0.51. Consistent with everything else measured on unseen ground — the
models become conservative rather than wrong.

---

## 2 · Validation health

The previous attempt at this comparison had validation F1 **exactly 0.0000** in all six
runs, because Bachsee_north was used as the validation site. Checkpoint selection ran on a
dead signal and the results were discarded. This is now the first thing checked.

| run | val F1 | val loss | verdict |
|---|---:|---:|---|
| A-r15-unet | 0.6760 | 0.468 | healthy |
| A-r15-hrnet | 0.7002 | 0.443 | healthy |
| B-fix-unet | *pending* | *pending* | |
| *(remaining)* | *pending* | *pending* | |

The in-domain → out-of-domain drop is large: HRNet 0.700 → 0.570 (−13 points), UNet
0.676 → 0.376 (−30). Oberheide stacks three shifts at once — unseen site, species mix
(37% beech against 99% and 86% in training), and sparse stems (2.27% against 5.93%/4.42%).

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

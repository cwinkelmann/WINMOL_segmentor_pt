# WINMOL segmentor — what was measured, and what it means

All figures below come from a run's own `test_results.md`; `docs/assets/session-results.json`
holds them verbatim. Nothing here is recomputed from memory.

---

## 1 · The corpus is smaller than it looks

| | |
|---|---:|
| annotated sites (with imagery) | **8** |
| stem polygons | 3,842 |
| **independent trees** | **1,777** |
| trees fragmented across ≥2 polygons | 909 (51%) |
| geometrically invalid polygons | 50 |
| digitized stem area | **9,804 m²** |
| annotated AOI | 32.35 ha |

**Under one hectare of digitized stem exists in total.** The polygon count double-counts
fragments: Barnekow_5 alone is 1,012 polygons over 332 trees. Tile counts mislead too — 100×
oversampling can mint ~90,000 tiles from those 32 ha, but they are 100× redundant views of
the same 1,777 trees.

Per species: **beech has 4 annotated sites and exactly 1 unannotated orthomosaic left**;
spruce has 4 annotated and 12 unannotated; **pine has none at all**, so no pine model is
currently possible. The 306 ha of unlabelled imagery is overwhelmingly conifer, so
annotating more of it does little for beech.

The original R `SpecDS` came from a site (`Quesenbank`) absent from this corpus, so the
published model's numbers are not reproducible here and are not a fair target.

---

## 2 · Architecture matters most

Five architectures on a leak-free block split, each with and without a synthetic
pretraining stage, and (where available) an ImageNet encoder:

| architecture | params | from scratch | + synthetic | + ImageNet |
|---|---:|---:|---:|---:|
| DPT (ViT-base) | 122.1M | 0.4917 | 0.5361 | 0.4671 |
| UNet (in-repo) | ~31M | 0.7638 | 0.7748 | — |
| SegFormer mit_b0 | 3.7M | 0.7788 | 0.7833 | 0.7773 |
| SegFormer mit_b2 | 24.7M | 0.7806 | 0.7588 | 0.7855 |
| **HRNet w18** | 16.1M | **0.7856** | 0.7870 | **0.7898** |

**Capacity without pretrained weights is a liability.** 3.7M works, 16.1M is the sweet
spot, 24.7M overfits, 122M fails outright. ViTs are the extreme case: almost no
convolutional prior, so with ~3,000 tiles there is nothing to fall back on.

**Measured against the UNet baseline of 0.7638, the three levers are worth very different
amounts:** architecture (→ HRNet) **+2.2**, ImageNet encoder **+0.4**, synthetic
pretraining **+0.1**.

**Synthetic pretraining helps most where the model is weakest** — +4.4 on DPT, +0.1 on
HRNet. It substitutes for capacity rather than adding what a good model lacks. SegFormer
b2 is the one negative (−2.2), consistent with overfitting in the fine-tune stage.

SegFormer mit_b0 deserves separate note: **0.7788 from 3.7M parameters**, above the ~31M
UNet.

---

## 3 · What a held-out orthomosaic costs

Same three configurations, two split strategies. `halve` shares each site between splits
(disjoint ground, same forest); `sites` holds Kaufland out entirely.

| configuration | halved site | held-out ortho | drop |
|---|---:|---:|---:|
| UNet | 0.7696 | 0.6699 | −10.0 |
| **HRNet, from scratch** | **0.8020** | **0.7253** | **−7.7** |
| HRNet + ImageNet | 0.7960 | 0.6901 | −10.6 |

**8–11 F1 points**, and the failure is almost entirely **recall**: precision holds or rises
(HRNet+ImageNet 0.8155 → 0.8328) while recall falls 12–19 points. The models become
*conservative*, not confused — they miss ~40% of stems but what they mark is still right.
That is the benign failure mode; the alternative produces confident nonsense.

**HRNet transfers best and degrades least**, and its lead widens with difficulty: +3.2
points halved, +5.5 held-out.

**ImageNet pretraining hurts transfer** (−0.6 halved, −3.5 held-out) despite helping on the
smaller block split (+0.4). Pretrained weights substitute for data; their value turns
negative once enough real data is present. **From scratch is the right default here.**

---

## 4 · Scale dependence is real and asymmetric

HRNet trained at 2.00 cm/px, evaluated across zoom on a held-out site:

| zoom | effective cm/px | AP | F1 |
|---:|---:|---:|---:|
| 0.70 | 2.86 | 0.7513 | 0.6702 |
| 0.85 | 2.35 | 0.7830 | 0.7077 |
| **1.00** | **2.00** | **0.7950** | **0.7225** |
| 1.15 | 1.74 | 0.7949 | 0.7208 |
| 1.30 | 1.54 | 0.7794 | 0.7031 |

**5.2 F1 points across ±30%**, lopsided: **−5.2 zooming out, −1.9 zooming in.** Thin stems
are where the model was already weakest, and zooming out makes them thinner.

This matters for the Analyzer. `ExecutionPlan.py:95-97` cuts `ceil(tile_size / pixel_size)`
pixels and resizes to 512, so **the model's effective ground resolution is `tile_size / 512`,
independent of the orthomosaic's own resolution** — 2.93 cm/px at the default 15 m, exactly
what the R generator produced. A user moving the tile-size spinbox silently rescales the
model's input. Check `tile_size` before blaming a model for inconsistent results.

**Threshold tuning is worth little in-domain and something out-of-domain.** On familiar
ground HRNet's best threshold is 0.503 and tuning gains nothing. On an unseen site,
HRNet+ImageNet's best threshold is **0.036** — a 14× shift — worth +1.7 points. Its AP
(0.7826) beats UNet's (0.7411) comfortably while their F1 at 0.5 is nearly identical: the
ranking is fine, the calibration is not.

---

## 5 · Amodal masks make stems more continuous

The annotations trace one windthrown stem as several polygons broken by occlusion, with
`id` recording which fragments are the same tree — 909 of 1,777 trees are fragmented, a
median 0.67 m gap, ~10% of a stem's extent missing.

Bridging those gaps and training on the result:

| | mean component length | longest | F1 |
|---|---:|---:|---:|
| modal-trained | 264 px | 379 px | 0.7626 |
| amodal, permissive bridging | 277 px | 400 px | 0.7634 |
| **amodal, conservative bridging** | **276 px** | 395 px | 0.7570 |
| *amodal labels (ceiling)* | *300 px* | *396 px* | — |

**+4.5% mean component length**, closing about a third of the gap to the label ceiling, for
about 0.6 F1. Precision falls and recall rises — the model marks trunk the camera could not
see, which modal labels do not credit.

The permissive run bridged roughly twice as much area, much of it invented; retraining on
conservative limits gives **+4.5% against +4.9%**, statistically the same. The benefit comes
from bridging *real* occlusions; the invented spans contributed nothing.

---

## 6 · The preprocessing comparison is not yet valid

Three extraction arms (R port / fixed-metre / native+multiscale) × two architectures were
run, and **all six had validation F1 exactly 0.0000**.

| arm | UNet | HRNet |
|---|---:|---:|
| A — R port | 0.0646 | 0.5038 |
| B — fixed-metre | 0.3635 | 0.6872 |
| C — native + multiscale | 0.1227 | 0.1680 |

Bachsee_north was used as the validation site. It is 100% amber pixels (November beech
canopy) against ~20% elsewhere, and a previous holdout of it scored exactly 0.0000 — a fact
recorded in the study's own spec, in the section on risks, about the *test* slot. (Measured
since: the labels are correctly registered and the stems are clearly separable in colour —
CIELAB separation 1.75, second best in the corpus. Trained on its own west half it reaches
F1 0.62 on its east half, so the site is learnable and the failure is domain shift. See
`scale-augmentation-loso.md`.)

With validation pinned at zero, checkpoint selection and the LR schedule run on a signal
carrying no information. **The 10× spread across arms is selection luck, not extraction
quality.** These numbers are reported here only so they are not mistaken for results later.

**The lesson, now in the experiment skill: a site that cannot be tested on cannot be
validated on either.** Validation is not a spare slot for awkward data.

The corrected fold draws validation from held-out blocks of the training sites (Campus,
10,685 m², 400 tiles at 4.55% stem coverage — real signal) and drops Bachsee_north entirely.

### A structural limit found on the way

**Native-1024 tiling cannot supply in-domain validation on this corpus.** A 1024 px tile at
Campus's 6.39 cm/px covers 65 m and needs a 46.2 m buffer; at Kaufland, 21.4 m and 15.2 m.
No block size leaves usable ground on either site. The same collision killed site-halving
for arm C. Native tiling and small AOIs are incompatible, and that is a property of the
corpus, not a bug.

---

## 7 · Where this leaves things

**Best measured configuration: HRNet w18, from scratch** — 0.8020 within-site, 0.7253 on a
genuinely unseen orthomosaic. It leads on every split, degrades least, and needs no
pretrained weights.

Open, in rough order of value:

1. **The preprocessing comparison**, on the corrected fold. Tooling and spec are ready.
2. **Anti-aliasing 2×2.** The Analyzer resamples two different ways —
   `utils/Prediction.py::_resize_batch` uses `anti_aliasing=False`, its stream path uses
   GDAL cubic, which filters. They disagree whenever a tile is downsampled, which happens on
   fine-resolution orthomosaics. Datasets are built; the experiment is two trainings.
3. **Seed replicates.** Almost everything here is n=1. Differences under ~1 point are
   directional only.
4. **A config mechanism** for `lr` and `seed`, neither currently exposed — a prerequisite
   for 1 and 3.

Two caveats that apply throughout: single runs, and a corpus where **site identity has
dominated every comparison** — larger than architecture, pretraining, or preprocessing.

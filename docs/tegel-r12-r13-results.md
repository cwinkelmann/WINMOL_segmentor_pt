# Revier 12 / 13 (Tegel) — zero-shot, fine-tune and native-resolution results

Design and the frozen test set were fixed before any model ran —
[`superpowers/specs/2026-08-16-tegel-r12-r13-design.md`](superpowers/specs/2026-08-16-tegel-r12-r13-design.md).
All numbers below are on that same held-out ground.

**Headline: existing beech models already reach F1 0.76 on this new survey with no Tegel
data at all.** Training on Tegel adds +3.5 to +6.2 F1. Fine-tuning from the beech model is
worth nothing over training from scratch. And **native-resolution extraction loses to
2.93 cm on every seed and both plots** through the deployment path — settled below.

## The data

Two July-2025 surveys, digitised as five widely separated sample plots.

| plot | polygons | stem area | AOI | role |
|---|---:|---:|---:|---|
| R12-P1 | 178 | 215 m² | 14,913 m² | val |
| R12-P2 | 552 | 891 m² | 23,080 m² | train |
| R12-P3 | 299 | 501 m² | 8,692 m² | **test (frozen)** |
| R13-P1 | 232 | 248 m² | 8,865 m² | train |
| R13-P2 | 123 | 141 m² | 13,436 m² | **test (frozen)** |
| total | **1,384** | **1,996 m² = 0.20 ha** | | |

That is a **20% increase on the entire pre-existing corpus** (0.98 ha), and it is far more
mixed than the beech sites:

| species | polygons | | species | polygons |
|---|---:|---|---|---:|
| beech | 897 | | hornbeam | 12 |
| pine | **153** | | poplar (3 spellings) | 16 |
| *(no species)* | 121 | | spruce | 7 |
| birch | 120 | | silver fir | 1 |
| ash | 41 | | oak | 16 |

**153 pine polygons**, in a corpus that previously had no annotated pine stand. Note
`POPLAR` / `WHITE POPLAR` / `SILVER POPLAR` are three spellings of one group — the same
naming-variant trap the older corpus has with `RBU`/`rBU`/`RBu`.

Orthomosaics: R12 1.20 cm/px (339k × 334k px, 12 GB, RGBA), R13 1.28 cm/px (393k × 335k px,
19 GB). Both finer than anything else held — Kaufland's 2.09 cm/px was previously the finest.

## Registration verified first

Labels are `EPSG:25833`, orthomosaics `EPSG:32633` — the exact datum mismatch that made
Bachsee_north look like a dead site. Checked before anything was trained or scored, on each
orthomosaic's own unrotated grid:

| plot | contrast at zero shift | bright peak | verdict |
|---|---:|---|---|
| R12-P2 | +1.205 | (0.00, 0.00) m | registered |
| R12-P3 | +1.094 | (0.00, 0.00) m | registered |
| R13-P1 | +0.894 | (0.00, 0.00) m | registered |
| R13-P2 | **+1.748** | (0.00, 0.00) m | registered |

All zero-offset, and **stem contrast is higher than anywhere in the beech corpus**
(Kaufland, the best, was +0.89). This is high-quality data.

## Results — per plot, never pooled

Test tiles cut at 19.512 m / 666 px (2.9297 cm/px), scored at the 1.00× centre window.
Three seeds per trained arm; the value is the mean.

| arm | R12-P3 | R13-P2 | scored at |
|---|---:|---:|---|
| zero-shot — beech UNet (jitter) | 0.7636 | 0.6081 | 2.93 cm |
| zero-shot — beech HRNet (jitter) | 0.7624 | 0.6313 | 2.93 cm |
| **Tegel from scratch** | **0.7988** | 0.6696 | 2.93 cm |
| **Tegel fine-tuned from beech** | **0.8031** | 0.6719 | 2.93 cm |
| Tegel native | 0.7861 | **0.6798** | *1.20 / 1.28 cm — different exam* |

Per seed (R12-P3 / R13-P2): scratch 0.7987·0.7970·0.8008 / 0.6659·0.6704·0.6725;
fine-tune 0.8001·0.7965·0.8128 / 0.6676·0.6604·0.6877;
native 0.7928·0.7688·0.7967 / 0.6909·0.6559·0.6925.

### Zero-shot already works

**0.76 on R12-P3 without a single Tegel tile in training**, against 0.78–0.79 for the beech
models on their own within-site test data. A genuinely new survey, a different season, a
mixed stand with pine and birch, and imagery 2.4× finer than the training scale — and the
model transfers. That is what matching the training GSD to the serving GSD buys, and it is
the strongest evidence yet for that rule.

R13-P2 is harder (0.61–0.63 zero-shot). It is the most mixed plot: 51% birch, 20% beech,
and 15% with no species recorded.

### Training on Tegel is worth +3.5 to +6.2 F1

Scratch minus zero-shot: **+3.5** on R12-P3, **+6.2** on R13-P2. The larger gain on the
harder, birch-dominated plot is the expected shape — the corpus had no birch stand.

### Fine-tuning buys nothing over training from scratch

**+0.4 F1 (R12-P3) and +0.2 F1 (R13-P2)**, both inside the per-seed spread. With 2,330
Tegel tiles, the beech initialisation is not doing measurable work. This is a useful
negative: for a survey of this size, a clean run is as good as a transfer, and simpler.
`--init-weights` was added for this arm and loads all 31,036,673 parameters or refuses.

### Native resolution — SETTLED: it loses, on both plots, on every seed

Resolved by running both models through the **WINMOL Analyzer**, each at its own matched
`tile_size` (15 m for the 2.93 cm models; 6.14 m on R12 and 6.55 m on R13 for the native
models, since 512 source px is 6.14 m at 1.20 cm/px). Both predictions were scored
identically: resampled to one 2.93 cm reference grid, masked to the plot AOI shrunk by 2 m,
via `scripts/score_stem_map.py`. Three seeds each.

| plot | 2.93 cm | native | delta | seeds favouring native |
|---|---:|---:|---:|:--:|
| R12-P3 | **0.7832** | 0.7653 | **-0.0178** | 0/3 |
| R13-P2 | **0.5840** | 0.5482 | **-0.0358** | 0/3 |

Per seed — R12-P3: 0.7832/0.7786/0.7877 against 0.7710/0.7464/0.7786.
R13-P2: 0.5742/0.5952/0.5826 against 0.5506/0.5343/0.5598.
**Native loses all six pairings.**

**The mechanism is precision, and it is consistent.** Native has *higher* recall on all six
runs and *lower* precision on all six:

| | recall | precision |
|---|---|---|
| R12-P3 | native 0.79–0.85 vs 0.79–0.85 | native **0.71–0.73** vs 0.73–0.77 |
| R13-P2 | native **0.70–0.74** vs 0.67–0.68 | native **0.43–0.45** vs 0.50–0.54 |

More pixels per stem finds more stem — and more things that merely look like stem. The net
is negative at both plots.

It is also far more expensive: at 6.14 m tiles the Analyzer processed **676 tiles per plot
against 11**, roughly 60× the inference for a worse result.

**Recommendation: extract at the serving scale (2.93 cm/px), not native**, even where the
imagery is 2.4× finer and even though the Analyzer's `tile_size` can be changed to match.

### New model vs the published WINMOL model, through the Analyzer

All four models run through the Analyzer at `tile_size 15` and scored identically —
resampled to one 2.93 cm reference grid, masked to the plot AOI shrunk by 2 m.

| model | R12-P3 | R13-P2 |
|---|---:|---:|
| **published** — Reder `SpecDS_Beech_512` (2023-02-28) | 0.7435 | 0.5715 |
| **ours, beech corpus** — BeechScale666 UNet, no Tegel data | 0.7404 | 0.5252 |
| **ours, Tegel from scratch** (3-seed mean) | **0.7832** | 0.5840 |
| **ours, Tegel fine-tuned** | 0.7811 | **0.5930** |

**The published model out-transfers our own beech-corpus model.** It matches it on R12-P3
(0.7435 vs 0.7404) and beats it by **4.6 F1** on R13-P2 (0.5715 vs 0.5252). Our beech models
are trained on four sites at the correct serving scale, with jitter, and still do not
transfer better than the published one to a new survey. The advantage we measured
*within* our corpus does not carry to new ground.

Two things this does not mean. It is not a like-for-like architecture comparison — the
published model is a different training corpus (including sites we do not hold, such as
Quesenbank) as well as a different pipeline. And it is a single plot pair; the two plots
disagree in magnitude.

**What does help is local data.** Training on Tegel beats the published model by **+4.0 F1**
on R12-P3 and **+2.2** on R13-P2. The gain comes from labelling the survey you intend to
run on, not from a better-scaled general corpus.

### Cross-validation of the harness

The world-space evaluator built for this comparison (`scripts/plot_inference.py`) and the
Analyzer are independent implementations — different tiling, different stitching, different
codebase. On the same model and plot they agree to **0.001 F1** (0.7842 vs 0.7832), which
is why the numbers above can be trusted.

### Tile-level F1 flatters — by a lot on the sparse plot

The same models score very differently depending on whether you measure on sampled tiles or
over the whole AOI:

| | tile-level | world-space, whole AOI |
|---|---:|---:|
| 2.93 cm @ R12-P3 | 0.7988 | 0.7832 |
| 2.93 cm @ R13-P2 | 0.6696 | **0.5840** |

Training tiles are sampled with a minimum stem fraction, so tile-level scoring
over-represents stem-rich ground and never asks the model about the empty areas where its
false positives live. On R13-P2 that is worth **8.6 F1** of optimism. Deployment numbers
should come from AOI-masked world-space scoring.

### The superseded framing

The tile-level table above scores the native arm on its own 1.20/1.28 cm tiles, where a
stem is ~2.4× thicker in pixels — a different exam, and not a valid comparison. On that
easier exam native appeared to win R13-P2 and lose R12-P3. The world-space result above
supersedes it: on a common grid native loses both.

## Reproducibility: the whole pipeline rebuilt and retrained identically

After the NAS mount on T14 dropped and was restored, both datasets were re-extracted and
all six models retrained from scratch. Every run reproduced its original to four decimal
places:

| seed | 2.93 cm rerun | 2.93 cm original | native rerun | native original |
|---|---:|---:|---:|---:|
| s1 | 0.7727 | 0.7727 | 0.7651 | 0.7651 |
| s2 | 0.7726 | 0.7726 | 0.7393 | 0.7393 |
| s3 | 0.7770 | 0.7770 | 0.7688 | 0.7688 |

Precision and recall match as well, 6/6.

This is a stronger check than a rerun of the same command, because several things changed
between the two passes and all had to be right at once:

- **A different storage path** — the NAS was remounted with `cifs-utils` freshly installed
  and different mount options (SMB 3.1.1). A resampling or byte-level difference in how the
  imagery was read would have moved the numbers.
- **A different extraction code path** — the rerun used `scripts/extract_parallel.py`, five
  concurrent processes merged with renumbering, against the original's single-process
  `make_splits`. A tile-index collision or a dropped plot in the merge would have shown up
  here.
- **Different hosts** for tiling and training than the first pass.

Before training, the tile sets were also compared directly: identical centres and rotations
in every split, 2330/500/1000 and 2400/500/1000.

So the parallel extractor is verified against the serial one not only on tile counts and
coordinates but on the models those tiles produce — and the ranking below holds on freshly
extracted data.

## Deviation from the pre-registered spec

The spec said the fine-tune would use strong hue augmentation
(`--aug-hue-shift 90 … --aug-hsv-p 0.9`). **It did not** — all three trained arms ran with
the default colour augmentation. The reason is that train and test here are the same
survey and the same season, where the strong setting is measured to *cost* 1.7 F1; its
+64.6 F1 benefit is out-of-domain. Applying it to only some arms would also have confounded
the native comparison.

Stated because the spec was written first and this differs from it. Untested consequence:
these Tegel models will likely transfer worse to a different season than a
strong-augmentation version would.

## Caveats

- **Two test plots.** 422 polygons. Reported separately throughout; the two disagree about
  the native arm, which is exactly why they are not pooled.
- **Mixed species.** These stands are 65% beech, 11% pine, 9% birch. A zero-shot gap may be
  species rather than site.
- **Season.** July, against a beech corpus that is mostly autumn and winter. Kaufland
  (July) is in the training corpus, which likely helps.
- **Colour augmentation left at defaults** for all arms, since train and test are the same
  season here. The strong setting costs 1.7 F1 in-domain and its benefit is out-of-domain.
- **Train/test share Reviere.** R12-P2 trains and R12-P3 tests; plot centroids are ~600 m
  apart. Leak-free (no shared ground) but not a new-forest claim.

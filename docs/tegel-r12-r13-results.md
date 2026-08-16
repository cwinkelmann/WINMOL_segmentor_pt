# Revier 12 / 13 (Tegel) — zero-shot, fine-tune and native-resolution results

Design and the frozen test set were fixed before any model ran —
[`superpowers/specs/2026-08-16-tegel-r12-r13-design.md`](superpowers/specs/2026-08-16-tegel-r12-r13-design.md).
All numbers below are on that same held-out ground.

**Headline: existing beech models already reach F1 0.76 on this new survey with no Tegel
data at all.** Training on Tegel adds +3.5 to +6.2 F1. Fine-tuning from the beech model is
worth nothing over training from scratch. Native-resolution extraction is **not** settled by
these numbers and is discussed honestly below.

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

### Native resolution — NOT settled by this table

The native arm trains and is scored at 1.20/1.28 cm/px, where a stem is ~2.4× thicker in
pixels than at 2.93 cm/px. **Pixel F1 across two different ground resolutions is not a
common yardstick** — that is the first rule in `.claude/skills/winmol-experiment`, and it
applies here in full.

What can be said: native wins on R13-P2 (+1.0 F1 over scratch) and loses on R12-P3
(−1.3 F1) *while playing the easier exam*. That is weak evidence **against** native being
clearly better, not evidence for it.

Settling it requires both models run over the same test-plot ground in world space, masked
to the plot AOIs and rasterised to one common reference grid — the same rule used for any
full-orthomosaic claim. That is built but not yet run; until it is, no ranking between
native and 2.93 cm should be quoted.

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

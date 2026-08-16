# Revier 12 / 13 (Tegel) — zero-shot benchmark and fine-tune, experiment design

**Question.** How do our existing beech models do on a genuinely new survey, and how much
does fine-tuning on part of it help?

**Written before any model was run.** The test set below is frozen and is the only thing
either benchmark is scored on.

---

## The data

Two July-2025 surveys, digitised as widely separated **sample plots** (Probekreise) rather
than one contiguous area. Labels are `EPSG:25833`; the orthomosaics are `EPSG:32633`.

| plot | polygons | stem area | AOI | centroid (E, N) |
|---|---:|---:|---:|---|
| R12-P1 | 178 | 215 m² | 14,913 m² | 381952, 5829658 |
| R12-P2 | 552 | 891 m² | 23,080 m² | 380666, 5828829 |
| R12-P3 | 299 | 501 m² | 8,692 m² | 381376, 5828926 |
| R13-P1 | 232 | 248 m² | 8,865 m² | 377121, 5827521 |
| R13-P2 | 123 | 141 m² | 13,436 m² | 374525, 5827975 |
| **total** | **1,384** | **1,996 m² = 0.20 ha** | | |

That is a **20% increase on the entire existing corpus** (0.98 ha), and unlike it these
stands are genuinely mixed:

| species | polygons |
|---|---:|
| beech | 897 |
| pine | **153** |
| birch | 112 |
| ash | 41 |
| oak | 11 |
| poplar | 4 |
| unlabelled | 121 |

**153 pine polygons** where the corpus previously had none in a pine stand.

Orthomosaics (on carrot): R12 `result_Res1.2_webp.tif`, 1.20 cm/px, 339k × 334k px, 12 GB,
RGBA. R13 `result_Res1.3_COG.tif`, 1.28 cm/px, 393k × 335k px, 19 GB, RGB. Both are finer
than anything in the beech corpus (Kaufland, 2.09 cm/px, was the finest).

## The yardstick — frozen now

**TEST = R12-P3 + R13-P2.** 422 polygons, 642 m² of stem, one plot from each Revier.
Never trained on, never validated on, used for every benchmark in this experiment.

| split | plots | polygons | stem area |
|---|---|---:|---:|
| train | R12-P2, R13-P1 | 784 | 1,139 m² |
| val | R12-P1 | 178 | 215 m² |
| **test (frozen)** | **R12-P3, R13-P2** | **422** | **642 m²** |

Chosen so that both Reviere appear in train and in test, and validation is a whole held-out
plot rather than a slice of a training one. Plot centroids are 600 m to 7 km apart, so a
plot-level split is leak-free by construction — no buffer argument needed. It is verified
numerically anyway.

**Metric:** pixel F1, precision and recall reported separately, on tiles cut inside the
test plots' AOIs. Recall and precision separately because every domain-shift failure
measured in this project has been a recall collapse with precision intact.

## Extraction

`--extent 19.512 --tile-px 666` → **2.9297 cm/px**, identical to `BeechScale666` and to
the Analyzer's default serving scale (`tile_size 15 / 512`). 666 px so the multiscale crop
range (394–666) is available, per the scale-jitter result. Scoring happens at the 1.00×
centre 512 window, the deployment point.

Inward buffer is `19.512 · √2/2 = 13.80 m` per plot, which the smallest AOI (R12-P3,
8,692 m²) still clears.

## Arms

1. **Zero-shot** — existing models applied unchanged: the BeechScale666 UNet and HRNet
   (fixed and jitter), and the published Reder beech model for reference.
2. **Fine-tuned** — the best zero-shot architecture, fine-tuned on the train plots.

Both scored on the identical frozen test set, so the comparison is a like-for-like delta.

## Checks required before any number is believed

- **Registration.** Labels and orthos are on different datums (25833 vs 32633), the exact
  pattern that made Bachsee_north look like a dead site. `scripts/check_registration.py`
  must show the stems peaking at zero offset on each ortho's own grid before anything is
  trained or scored.
- **Leak-freedom**, numerically, from `tiles.jsonl`.
- **Colour augmentation.** These are July acquisitions against a corpus whose beech sites
  are October, November, February and July. Strong hue augmentation
  (`--aug-hue-shift 90 --aug-sat-shift 60 --aug-val-shift 30 --aug-hsv-p 0.9`) is the
  measured default now; the fine-tune uses it.

## Declared confounds

- **Mixed species.** The corpus models are trained on beech-dominated stands; these plots
  are 65% beech, 11% pine, 8% birch. A zero-shot drop may be species, not site.
- **Resolution.** 1.20–1.28 cm/px native, resampled 2.4× down to reach 2.93 cm/px. No other
  corpus site required that much downsampling.
- **Small test set.** 422 polygons across two plots. Per-plot numbers are reported
  separately, never pooled into a single headline.
- **Season.** July, against a beech corpus that is mostly autumn/winter.

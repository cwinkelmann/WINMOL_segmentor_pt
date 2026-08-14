---
title: "WINMOL stem segmentation — method corrections and scale-augmentation results"
author: "Christian Winkelmann"
date: "14 August 2026"
toc: true
toc-depth: 2
numbersections: true
geometry: margin=2.4cm
fontsize: 10pt
linkcolor: RoyalBlue
urlcolor: RoyalBlue
---

# Executive summary

Two independent bodies of work, in the order they should be read.

## Part I — Corrections to the published WINMOL method

Read directly from `WINMOL_segmentor` (R) and `WINMOL_Analyzer` (Python), with every value
quoted from source or measured by running it. Five items change how the published results
should be read, and three of them are places where the papers and the code disagree.

| # | Finding | Consequence |
|---|---|---|
| 1 | **The composite F1/BCE loss does not train.** `F1Score_loss` calls the *metric*, which applies `k_round`; a rounded value has zero derivative almost everywhere. | The model trains on **plain BCE**. The term appears in the reported loss and is absent from the optimisation. Verified three ways, including an 8-epoch R run agreeing with pure BCE to **eight decimal places**. |
| 2 | **The split is a random 80/20 over tiles**, drawn per stage — not site-level, not spatial. With 100× oversampling, tiles overlap. | Early stopping, the LR schedule and checkpoint selection all read an optimistic validation signal. |
| 3 | **Tile side length is 15 m, not ≈15.4 m**, fixing effective GSD at **2.93 cm/px**. | Load-bearing: matching training GSD to it was worth **+14.6 F1** in our measurements. |
| 4 | **Node spacing is 50 cm, not the documented 25 cm** (`min(min_length, 0.5)`). | Measured 0.47–0.50 m on real output across four orthomosaics. |
| 5 | **A transposed digit in the decoder** (`265` filters where the symmetric ladder gives 256). | The published `.hdf5` is 31,140,642 parameters; the described architecture gives 31,036,673. Not reproducible from the paper without the typo. |

Plus two coordinate traps in the vectoriser that produce plausible geometry rather than an
error — the skeletonisation pad is never removed, and the EDT diameter branch reads pixel
`(row, col)` as easting/northing. Full detail with file and line references in Part II.

**A reproduction worth noting.** Trained single-stage on the published SpecDS and scored on
the published TestDS with our own metric code: **0.7581**, which matches the 2022 paper's
*best two-stage pretrained* result (75.6%) without any GenDS pretraining. The published
beech model scores **0.6812** on that same TestDS — below the paper's figure, unexplained,
and worth raising with the author.

## Part II — Scale augmentation

The Analyzer's effective resolution is `tile_size / 512`, a user-set knob most people never
touch deliberately. Jittering the training footprint ±30% makes a model robust to it:

| | fixed spread | jitter spread | improvement | cost at 1.00× |
|---|---:|---:|---:|---:|
| HRNet, within-site | 0.0295 | 0.0207 | −0.0087 (t −6.0) | −0.0011 (noise) |
| UNet, within-site | 0.0334 | 0.0146 | −0.0188 (t −57.0) | **+0.0020** (3/3 seeds) |
| UNet, clean site holdout | 0.1231 | 0.1116 | −0.0115 | **+0.0328** (3/3 seeds) |

**Recommendation: enable it by default.** It costs nothing at the deployment scale, removes
30–56% of the degradation when the tile size moves, and on the one clean site holdout where
segmentation works at all it is worth **+3.3 F1** at the deployment scale.

It does **not** replace matching the training scale to the serving scale — that is worth
14.6 F1 against this 0.9–3.3. Do both.

## What is not settled

- **Bachsee_north returns F1 < 0.13 for every model ever trained here.** Long recorded as a
  colour-domain failure; the imagery shows a closed canopy with the stems beneath it, and
  stem-vs-background contrast there is **−0.17** of a tile standard deviation against
  **+0.89** at Kaufland. Whether the labels are also misregistered against this
  orthomosaic — its annotations are EPSG:25833 and its ortho EPSG:32633, a datum pair that
  diverges ~0.5 m — is **under test and not yet established**. A first attempt to measure
  it in tile-pixel space was invalid, because tiles are cut at random rotations and a
  constant world offset therefore appears at a different pixel offset in every tile.
- **The site holdout rests on one informative fold.** Of four beech sites, two are the same
  forest 4.4 years apart (33% footprint overlap) and one is Bachsee_north.
- **Label scarcity.** Under 1 ha of stem is labelled in total; beech is near exhausted and
  pine has none.

\newpage

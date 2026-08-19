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

**Recommendation: enable it by default.** It costs nothing at the deployment scale and
removes 30–56% of the degradation when the tile size moves.

On the one clean site holdout it also *gained* accuracy — but treat that number with care:
the per-seed differences were **+1.1, +3.3 and +5.5 F1** (mean +3.3, 3/3 positive, but
|t| = 2.6, **below the |t| ≥ 3 bar used elsewhere in this report**). One plot, one
architecture, three seeds. The direction is consistent; the magnitude is not well
determined, and it should not be quoted as "+3.3" without that range.

It does **not** replace matching the training scale to the serving scale — that is worth
14.6 F1 against this 0.9–3.3. Do both.

## Part IV — Deployed on the Tegel surveys

Both orthomosaics (16.3 and 21.6 km²) run end to end through the Analyzer with our model
and with the published one: **8,323 / 25,238 stems** against the published model's
**6,483 / 20,141**, plus a diameter every 50 cm along each stem. Restricted to the frozen
test plots the full run reproduces the plot-level scores to within 0.5 F1, so the pipeline
is consistent from tile to full survey.

Counts alone are not evidence of quality — outside the five annotated plots nothing was
digitised, so extra detections there cannot be judged.

## Part III — Bachsee_north is not a bad site

Every model trained here scores F1 < 0.13 on Bachsee_north, and the corpus notes have long
attributed this to its amber November canopy. Three explanations were tested; the first two
are wrong.

| checked | result |
|---|---|
| Are the labels misregistered? Annotations are EPSG:25833, ortho EPSG:32633 — a datum pair diverging ~0.5 m. | **No.** On the orthomosaic's own grid every beech site peaks at **zero** offset; residual under 4 cm. |
| Are the stems invisible under closed canopy? | **No.** That came from a *luminance-only* statistic. In CIELAB, Bachsee's stem/background separation is **1.75** — second highest in the corpus, above Campus's **1.16**, and Campus scores F1 0.53. |
| Is the site learnable at all? | **Yes.** Trained on 252 tiles from its own west half and tested on the spatially separated east half: **F1 0.6203**. |

![Colour domain by site — Bachsee_north is the only late-autumn acquisition](figures/loso-site-domains.png)

![One labelled stem per panel: bare, then outlined. Bachsee's stems are visible](figures/bachsee-closeups.png)

**252 of its own tiles beat 3,588 tiles from every other site by 53 F1 points.** The data
is sound; the failure is **domain shift**.

Re-running the folds with Bachsee **excluded from training** (18 runs) confirms it belongs
there: overall **-0.045 F1**, 5 of 18 paired comparisons positive. The damage is ordered by
phenological distance from Bachsee's November amber — Campus (October) **-0.095**,
Campus_Oberheide (leaf-off February) **-0.045**, Kaufland (high summer) **+0.005** — and
the two Campus folds lose the identical 800 tiles, so tile count does not explain the
ordering. The out-of-domain site is the corpus's phenological diversity, not noise. Bachsee is the corpus's only autumn-phenology
site, so when it is held out nothing in training has shown the model a stem against orange
foliage. **Strong hue augmentation fixes it.** Randomising hue over the full circle
(`--aug-hue-shift 90 --aug-sat-shift 60 --aug-hsv-p 0.9`) takes the Bachsee holdout from
**F1 0.042 to 0.688** (3/3 seeds) for **−1.7 F1** in domain (Kaufland, 3/3 seeds). The base
arm predicts almost nothing there — precision 0.87–1.00, recall 0.00–0.03; the strong arm
reaches recall 0.58–0.66. This retires the standing advice that a site holdout is
impossible on this corpus. It is an **interaction, not a dose**: a full hue circle at the
default probability reaches only 0.261, and half the hue range at full probability only
0.177, while both together reach 0.688. Do not soften the setting.

An earlier claim in this repo that fold F1 is monotone in stem contrast came from the
luminance-only statistic and is **withdrawn**.

## What is not settled

- **The site holdout rests on one informative fold.** Of four beech sites, two are the
  same forest 4.4 years apart (33% footprint overlap) and one is Bachsee_north, which is
  outside every training domain (below).
- **Label scarcity.** Under 1 ha of stem is labelled in total; beech is near exhausted and
  pine has none.

\newpage

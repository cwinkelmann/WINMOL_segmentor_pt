The goal is to build a segmentation model which works the best on unseen data.


There are a couple of things to keep in mind. 

Those are: different tree species look different, there are different ground sampling distances

For instance on 20220212_Barnekow_5 we have very highly resolved images, see @./docs/training-data-from-annotations.md for it. 20171016_EW_WW_Campus is rather low resoltion.

Assuming we would always resample 15m tiles into 512px we probably reduce performance for Barnekow. We might even loose performance on 20171016_EW_WW_Campus too. SOme insights are collected here @/Users/christian/work/work/WINMOL_segmentor_pt/docs/preprocessing-comparison.md

Are the implementation modernisations benefiting or hurting performance?, can we test that on our newly retrieved orthomosaics? TODO: run a truly faithfull reimplementation of the R code and compare it to the current implementation.

------
Section for ideas of Claude
------
How should we train a model? 
can pretraining on synthetic data help ( @ /Users/christian/work/work/WINMOL_segmentor_pt/docs/synthetic-pretraining-beech.md )

## The GSD spread, measured

| site | species | cm/px | AOI m² | resample to 15 m / 512 px |
|---|---|---:|---:|---|
| Barnekow_5 | spruce | 1.58 | 27,788 | ×1.85 **down** |
| Bremerhagen_3 | spruce | 1.72 | 2,566 | ×1.70 **down** |
| Barnekow_3 | spruce | 1.76 | 15,516 | ×1.67 **down** |
| Kaufland | beech | 2.09 | 5,393 | ×1.40 **down** |
| Campus_Oberheide | beech | 3.36 | 111,631 | ×0.87 up |
| Bachsee_north | beech | 4.08 | 31,527 | ×0.72 up |
| Campus | beech | 6.39 | 118,808 | ×0.46 up |

A 4× spread with the target in the middle. Your intuition is right that a fixed
15 m/512 px target discards detail at Barnekow and invents pixels at Campus — but the
weighting is lopsided. The two sites that dominate by area (230,439 m² of ~303,000 m²)
are the ones being *upsampled*; only 51,263 m² is downsampled, and most of that is spruce.

## Training scale is the largest effect measured in this project

The Analyzer's effective resolution is `tile_size / 512`, independent of the
orthomosaic's own GSD. Default `tile_size = 15` → 2.93 cm/px. On Kaufland, AOI-masked,
same architecture and split family:

| trained at | served at | F1 |
|---|---|---:|
| 2.00 cm/px | 2.93 cm/px | 0.7389 |
| 2.00 cm/px | 2.00 cm/px | 0.8091 |
| **2.93 cm/px** | **2.93 cm/px** | **0.8844** |

**+14.6 F1 from matching the serving scale** — larger than every architecture and
pretraining effect combined. Per-site optimality is the wrong target: the Analyzer serves
every site at one scale, so that is the scale to optimise.

## The architecture gap collapses at the coarser scale

Same corpus, same block splits, only the footprint changed:

| model | 10.24 m (2.00 cm/px) | 15 m (2.93 cm/px) |
|---|---:|---:|
| HRNet + ImageNet | 0.7298 | **0.8011** |
| SegFormer mit_b2 | 0.7153 | 0.7974 |
| HRNet, scratch | 0.7197 | 0.7866 |
| UNet (classic) | 0.6877 | 0.7862 |

At 10.24 m the classic UNet trailed HRNet by 3.2 points; at 15 m it trails by **0.04**.
That reframes the earlier architecture study — much of what it measured was "which
architecture copes with tiles that are too fine", not "which segments stems better". **If
we serve at 15 m the classic UNet is not the liability it looked like**, which matters
because it is the only architecture with a Keras mirror and the widest deployment surface.

## Answering the three questions

**Does synthetic pretraining help?** No. Two-stage runs against their own baseline: SD
imagery −0.2 (HRNet) / −0.4 (UNet), Blender geometry −1.4. Stage 1 learns the synthetic
sets easily (val F1 0.887–0.903); the features do not transfer. Same for ImageNet (−0.1),
DINOv3 (−7.3) and ConvNeXt-V2 FCMAE (−10.4) at equal parameter count. Notably, plain
supervised ImageNet beat both self-supervised regimes by 7–10 points — if any pretraining
is used, use that one.

**Do the modernisations help or hurt?** The extraction rewrite is the clear win:
fixed-metre sampling beats the R port by +8.7 to +26.5 depending on architecture, entirely
in recall, because R's acceptance rule counts the whole area of every polygon touching a
footprint rather than the part inside it. Bigger models, transformer encoders and
self-supervised weights are all neutral or negative here.

**Can we test on the new orthomosaics?** Only qualitatively — R12/R13 (Tegel) have no
annotations. The beech model produced 2,467 stems on the 1 km R13 centre crop in 160 s,
which shows the pipeline works but cannot score it. A number needs either annotations
there or a leave-one-site-out fold on the annotated corpus.

## What I would do next, in order

1. **Scale jitter instead of a fixed scale.** Train at 15 m but vary the footprint ±30%
   per sample. The scale sweep measured 5.2 F1 lost across ±30% zoom, asymmetric (−5.2
   out, −1.9 in); jitter should flatten that curve and is the direct answer to a 4× GSD
   spread. Cheap — a sampler flag plus a loader change, no new data.
2. **Native-resolution crops, properly powered.** The earlier native+multiscale arm lost,
   but trained on 1,185 tiles against 2,700. That confound was never separated; rebuild at
   matched tile count before concluding.
3. **Separate the species question from the resolution question.** Every spruce site is
   1.6–1.8 cm/px and beech spans 2.1–6.4, so species and GSD are confounded corpus-wide.
   Worth measuring how much of any spruce/beech gap is really a resolution gap.
4. **Annotate at the resolution extremes, not uniformly.** Kaufland is the only fine beech
   site and at 5,393 m² it is too small to block-split at 15 m. One more fine-resolution
   beech site would buy more scale robustness than more Campus.
5. **Leave-one-site-out before any deployment claim.** Everything above is a single fold.
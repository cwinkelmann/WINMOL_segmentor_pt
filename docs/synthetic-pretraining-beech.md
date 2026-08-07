# Does synthetic pretraining help a beech stem segmenter?

Short answer: yes, by about **+1.1 F1** on a leak-free split — but from a single run per
arm, so treat it as a direction, not a measured effect size.

## Why the question matters here

The WINMOL corpus contains **9,802 m² of digitized stem in total** — under one hectare,
across 3,842 polygons and 1,777 trees. Usable beech is roughly a third of that, and only
**one** unannotated beech orthomosaic exists in the corpus to expand it with. Labelling
is close to exhausted for this species.

A generator that produces unlimited perfectly-labelled tiles is therefore not a side
experiment. It is one of the few remaining sources of signal.

## Setup

Two arms, identical apart from stage 1, evaluated on the same held-out tiles:

| arm | stage 1 | stage 2 |
|---|---|---|
| A | — | beech tiles |
| B | 3,000 synthetic tiles | the same beech tiles |

UNet, 30 epochs, batch 16, fixed `--val-data-dir` and `--test-data-dir` so neither arm
re-splits. Two-stage resets the learning rate between stages, so stage 2 does not inherit
stage 1's decayed rate.

Real tiles are sampled at `--extent 10.24`, putting them at **512 px / 10.24 m = 2 cm/px**
— exactly the synthetic generator's `gsd_m_per_px`. Sharing a ground resolution between
the two stages is the point of pretraining; the R script's 15 m footprint would have
introduced a 1.5× scale gap for no reason.

| split | tiles | mean stem coverage |
|---|---:|---:|
| train | 3,084 (Campus 2,500 + Kaufland 584) | 5.14% |
| val | 500 (Campus) | 2.72% |
| test | 570 (Campus 500 + Kaufland 70) | 4.87% |

## Result

Three architectures, each with and without the synthetic stage, on the same leak-free split:

| architecture | params | beech only | synthetic pretrain | gain |
|---|---:|---:|---:|---:|
| UNet (in-repo) | ~31M | 0.7638 | 0.7748 | **+1.10** |
| SegFormer (mit_b0) | 3.7M | 0.7788 | 0.7833 | +0.45 |
| **HRNet (w18)** | 16.1M | **0.7856** | **0.7870** | +0.14 |

Synthetic pretraining helps every architecture. But **the gain shrinks monotonically as the
baseline strengthens** — +1.10 points for the weakest, +0.14 for the strongest.

Two consequences worth stating plainly:

- **HRNet with no pretraining at all (0.7856) beats UNet with it (0.7748).** On this corpus,
  changing architecture buys more than synthetic pretraining does.
- The two do not stack. Synthetic data appears to substitute for model capacity here rather
  than supplying information a stronger model lacks.

SegFormer is notable separately: 0.7788 from **3.7M parameters**, above the ~31M UNet, on a
corpus with under a hectare of label. The smallest model in the comparison beats the incumbent.

For reference, the published R model scores 0.760 on its own TestDS, so none of these are
being flattered by a weak baseline.

**One run per arm.** HRNet's +0.14 is indistinguishable from seed noise, and even +1.10 is
suggestive rather than established. The architecture ranking spans a wider range and is more
likely real. Replicating across seeds would settle both, and needs a `--seed` CLI flag on
`run_train` first — it lives in `TrainConfig` and is not currently exposed.

A four-page PDF of this study is at `docs/assets/synthetic-pretraining-study.pdf`, rendered by
`scripts/report_synth_pretraining.py` from `docs/assets/synth-pretraining-results.json`. Every
number there is copied from a run's own `test_results.md`; the script computes no metrics, so
the report cannot drift from what training reported.

## Two earlier attempts that were wrong, and why

Both are worth recording, because both produced numbers that looked like results.

### Holding out a whole site gave F1 exactly 0.0000 — for both arms

The first design held out Bachsee_north. Both arms scored 0.0000 with peak output
probability 0.0024 — the models never fired at all.

The labels were fine: at native resolution the annotated stems are visible in that site's
own orthomosaic and correctly aligned. The problem was colour.

| set | mean RGB | amber pixels |
|---|---|---:|
| Campus (train) | 0.428 / 0.444 / 0.377 | 19.8% |
| Kaufland (train) | 0.531 / 0.631 / 0.461 | — (66% green) |
| **Bachsee_north (test)** | 0.478 / 0.279 / 0.126 | **100%** |
| Synthetic | 0.336 / 0.309 / 0.312 | 20.8% |

Bachsee_north is a November beech canopy in full autumn colour — a domain nothing else
covers, synthetic included. With three beech sites, holding one out removes an entire
acquisition: its phenology, its colour cast, its GSD. What comes back measures domain
transfer, not segmentation.

Hence `--block-size` / `--split` in `sample_training_tiles.py`: cut each AOI into blocks,
assign whole blocks to train/val/test, shrink each by the footprint half-diagonal so no
tile can straddle a boundary at any rotation. Every split spans the whole site, and no
pixel is shared.

Bachsee_north remains a legitimate *hard-transfer* benchmark. It is not a fair test set.

### The stem filter tested the wrong square

Acceptance measured stems inside a world-space footprint rotated counter-clockwise, but
`PIL.Image.rotate` turns the *image* counter-clockwise, which samples the world square
rotated the other way. The two coincide only at 0° and 90°.

Image and mask still rotated together, so labels stayed registered and an overlay of
sampled tiles looked perfect — which is exactly why this survived a visual check. Only
the *selection* was wrong: **3.2% of written tiles fell below the 0.5% stem floor and 21
were completely empty.**

The floor is now applied to the finished mask, which is exact by construction. After the
fix: 0 empty masks, 4 of 4,154 marginally under floor (the final resize to 512 shaving
borderline tiles).

The pre-existing floor test could not catch this — its stem fixture is dense enough that
every square contains stems whichever way it turns. `test_sparse_stems_never_yield_a_tile
_below_the_floor` spaces stems ~8 m apart so the rotation sign decides the outcome.

## Caveats on the synthetic data itself

- **24% of synthetic tiles have >10% violet pixels** — a colour no orthomosaic produces.
- The generator renders **no vegetation green at all** (0.0%, against 66% for a leaf-on
  July site). Its palette matches autumn and leaf-off scenes; it does not cover leaf-on.
- Label statistics match well, which is likely why it helps despite the above: synthetic
  stem coverage averages 7.0% against 6.2% for real tiles.

So the +1.1 here is measured on autumn/leaf-off-ish sites. A leaf-on target would
probably benefit less until the generator grows a summer palette.

## Reproducing

```bash
# 1 — build the leak-free block split (2 cm/px, matching the generator)
python scripts/sample_training_tiles.py --ortho <site>_ortho.tif --stems <site>.shp \
  --aoi <site>_AOE.shp --out <DS>/train --extent 10.24 \
  --block-size 60 --split train --split-seed 1
#     ... repeat with --split val and --split test, same --split-seed

# 2 — arm A
python -m training.run_train --data-dir <DS>/train --val-data-dir <DS>/val \
  --test-data-dir <DS>/test --arch unet --epochs 30 --batch-size 16 --device cuda \
  --no-cache-dataset --num-workers 8 --out-dir runs/A

# 3 — arm B: synthetic stage 1, same beech stage 2
python -m training.run_train --gen-data-dir <SYNTH> --spec-data-dir <DS>/train \
  --test-data-dir <DS>/test --arch unet --epochs 30 --batch-size 16 --device cuda \
  --no-cache-dataset --num-workers 8 --out-dir runs/B
```

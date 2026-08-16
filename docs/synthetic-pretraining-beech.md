# Does synthetic pretraining help a beech stem segmenter?

> **Correction (2026-08-14).** This document's explanation of the F1 0.0000 site holdout —
> that Bachsee_north's amber canopy is an unseen colour domain and its labels may be
> unusable — is **wrong**. Its labels are correctly registered, its stems are the
> second-most separable in the corpus in CIELAB, it reaches F1 0.62 trained on its own west
> half, and **strong hue augmentation takes the holdout from 0.042 to 0.688**. The
> synthetic-pretraining measurements below stand; the diagnosis does not.
> See [`scale-augmentation-loso.md`](scale-augmentation-loso.md).

Short answer: **no.** This page originally reported +1.1 F1; a later, corrected repeat
measured **−0.2 to −1.4**. Both sets of runs are kept below, with the reason the first
set is void.

> **The runs on this page predate a fix to `run_two_stage`.** Stage 2 built its loaders
> without passing `--val-data-dir`, silently falling back to a random split of the
> training tiles — so stage-2 checkpoints were selected on a different, easier signal
> than the single-stage baselines they were compared against. Fixed in `a1ca11d` and
> pinned by `tests/test_two_stage.py::test_two_stage_stage2_validates_on_the_given_val_dir`.
>
> Repeating the comparison with the fix, on the fixed-metre corpus:
>
> | stage-1 source | architecture | Δ vs no pretraining |
> |---|---|---:|
> | Stable-Diffusion imagery | HRNet | −0.2 |
> | Stable-Diffusion imagery | UNet | −0.4 |
> | Blender geometry | HRNet | −1.4 |
>
> Stage 1 still learns the synthetic sets easily (val F1 0.887–0.903). The features do not
> transfer. **Treat the numbers below as the record of how the question was first
> answered, not as the answer.**


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

The two architectures that matter for deployment, each with and without the synthetic
stage, on the same split:

| architecture | params | beech only | synthetic pretrain | gain |
|---|---:|---:|---:|---:|
| UNet (in-repo) | ~31M | 0.7638 | 0.7748 | +1.10 |
| **HRNet w18** | 16.1M | **0.7856** | **0.7870** | +0.14 |

**The gain tracks how weak the model is.** UNet, the weaker of the two, gains +1.1; HRNet,
the stronger, gains +0.1. Synthetic data compensates for a data-starved model rather than
adding something a good one lacks. Transformer and ViT variants were also run and are
summarised in the appendix; they do not change this conclusion.

**Architecture buys more than synthetic data.** UNet → HRNet is +2.2 points; the best
synthetic gain on a working model is +1.1.

For reference the published R model scores 0.760 on its own TestDS, so these baselines are
not weak.

**None of these models had pretrained weights** — `encoder_weights=None` throughout, so the
synthetic comparison stayed unconfounded.

**One run per arm.** HRNet's +0.14 is indistinguishable from seed noise. The architecture
ranking spans a wider range and is more likely real. Replicates need a `--seed` CLI flag on
`run_train`, which currently lives in `TrainConfig` unexposed.

A four-page PDF is at `docs/assets/synthetic-pretraining-study.pdf`, rendered by
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

---

## Appendix — transformer and ViT variants

Run alongside the two above and excluded from the main table because neither is a
deployment candidate and both muddy the synthetic question with a capacity effect.

| architecture | params | beech only | synthetic pretrain | gain |
|---|---:|---:|---:|---:|
| DPT (ViT-base) | 122.1M | 0.4917 | 0.5361 | +4.44 |
| SegFormer mit_b0 | 3.7M | 0.7788 | 0.7833 | +0.45 |
| SegFormer mit_b2 | 24.7M | 0.7806 | 0.7588 | −2.18 |

Three points, briefly:

- **Capacity without pretrained weights is a liability.** 3.7M works, 16.1M is the sweet
  spot, 24.7M starts to overfit and 122M fails outright. A ViT has almost no convolutional
  prior, so with ~3,000 tiles there is nothing to fall back on. (DPT ran at batch 8 rather
  than 16 for memory — a real confound, though not one that explains a 0.29 F1 gap.)
- **DPT is train-only.** It cannot meet the ONNX contract: its ViT encoder is fixed at 384,
  `dynamic_img_size=True` interpolates position embeddings with antialiased bicubic, and
  `aten::_upsample_bicubic2d_aa` has no ONNX lowering at opset 17–20. Exporting at native
  384 produces fixed spatial dims the contract rejects, at ~485 MB.
  `test_dpt_cannot_yet_export_onnx` pins this.
- **SegFormer never beat HRNet on a deployment-relevant test.** Later runs at the
  Analyzer's own scale put mit_b2 at 0.7974 against HRNet's 0.8011, inside noise, at 1.5×
  the parameters. There is no case for carrying it further.

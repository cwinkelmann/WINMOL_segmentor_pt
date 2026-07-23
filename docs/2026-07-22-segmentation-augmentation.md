# Sample-mixing augmentation on SpecDS_ready — mosaic / copy-paste / cutout

Run 2026-07-22. UNet, 20 epochs, batch 8, seed 1, `--no-cache-dataset --num-workers 4`, CUDA.
Identical in every respect except the augmentation flags, so the seeded 80/20 tile split is the
same set of tiles for all four arms. Implementation: `training/mix_augment.py`
(`--aug-mosaic-p / --aug-copypaste-p / --aug-cutout-p`).

## Result: none of them helped

| arm | flags | val F1 | precision | recall | loss |
|---|---|--:|--:|--:|--:|
| **baseline** | — | **0.9079** | 0.9090 | 0.9067 | 0.1363 |
| mosaic | `--aug-mosaic-p 0.5` | 0.9054 | 0.9056 | 0.9052 | 0.1399 |
| copy-paste | `--aug-copypaste-p 0.5` | 0.9014 | 0.9099 | 0.8931 | 0.1452 |
| combo | all three at 0.3 | 0.8975 | 0.9056 | 0.8896 | 0.1508 |

The ordering is monotone in how much mixing is applied: the more the training distribution is
perturbed, the *worse* the score, and the loss ranks the same way. Precision is flat (0.906–0.910)
while recall carries the whole difference (0.9067 → 0.8896), i.e. the mixed-up models find fewer
stems rather than inventing more.

## What this does and does not show

**It does not show that these augmentations are useless.** The measurement is a random *tile*
split of `SpecDS_ready`, so validation tiles come from the same orthomosaics as training tiles
(see `BUGS.md` #2). Augmentation buys generalization to *new* scenes — precisely the axis this split
cannot see. Evidence from this repo that the axis matters: in
`results/bamforests_experiments/COMBINED_SITESPLIT.md` the same models score val F1 0.846–0.852 but
cross-site TestDS F1 0.747–0.815, and the random-split report concluded augmentation was worth
**+0.09 to +0.15 F1** on a *held-out* set.

So the honest reading is: **on in-distribution data, sample mixing costs a little accuracy and
buys nothing measurable.** Whether it helps out-of-distribution is untested here.

Two further confounds worth stating:
- **20 epochs may be too few.** Heavier augmentation slows convergence; a fixed epoch budget
  favours the least-augmented arm. The baseline may simply have converged furthest.
- **Single seed.** The spread (0.8975–0.9079) is ~1 pp, which is within the range seed noise can
  produce; no arm is separated by more than that from its neighbour.

## What to do instead

1. **Re-run on a scene-grouped or site-held-out split** — the only setting where these
   augmentations could show a benefit. This is the same fix `BUGS.md` #1/#2 call for.
2. If the goal is more *stems* rather than better pixels, note the error budget in
   `learned_vectorization/README.md`: mask quality moves the analyzer's stem count by **2.04×**,
   so recall on real (not augmented) scenes is what matters.

## Historical note

The 2021/2022 WINMOL paper's stage-1 pre-training is itself a copy–paste augmentation (stems pasted
onto backgrounds, 10/50/100 augmentations per annotated stem). It reported a gain of ~1–3 F1 points
from that pre-training (72.6% baseline → 73.9/74.3/75.6%) — evaluated on a *different* windthrow,
i.e. exactly the out-of-distribution setting this run does not measure.

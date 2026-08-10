---
name: winmol-experiment
description: Use when running or comparing any WINMOL training experiment — architecture comparisons, preprocessing variants, augmentation studies, pretraining arms, or any claim that one configuration beats another. Enforces a common yardstick, leak-free splits, sane metrics, and honest reporting.
---

# Running a WINMOL experiment

Every rule here exists because its absence already produced a wrong result in this repo.
They are cheap to follow and expensive to skip.

## Before running anything

**1. Name the yardstick, in writing, before training.**

Two configurations trained on different data are scored on different exams. Write down
what all arms will be measured on and confirm no arm's own preprocessing appears in it.

> *Happened here:* a modal-trained and an amodal-trained model were each scored against
> their own labels. The modal target is strictly easier, so the numbers were not
> comparable. Separately, three preprocessing arms each produced a different test set
> (15 m / 10.24 m / native-px footprints) — comparing their F1 would have been meaningless.

For full-pipeline claims the yardstick is **full-orthomosaic inference masked to the
AOI**. Outside the windthrow polygon stems are real but undigitised, so scoring the whole
raster counts correct detections as false positives — and penalises the *better* model
hardest, inverting the result.

**2. Declare what differs between arms, and hold everything else fixed.**

Seed, epochs, batch size, optimiser, geometry repair, and the held-out site are constant.
If something else must differ, it is a confound and goes in the report.

> *Happened here:* arm C was nearly launched at batch 8 (larger tiles) while arms A and B
> used 16 — two differences at once. Also: native tiling yields 1,749 training tiles
> against 3,600, which is a real alternative explanation for any result.

**3. Verify the split is leak-free numerically, not by assertion.**

Use the tile provenance (`tiles.jsonl`) to compute the closest centre-to-centre distance
between splits and compare it against `2 × extent × √2 / 2`, below which two tiles can
overlap at all.

> *Happened here:* a tile-level split with 100× oversampling put near-duplicate tiles on
> both sides. The verified halved split showed 34.20 m closest approach against a 14.48 m
> overlap threshold — that is what a leak-free claim looks like.

## Choosing a split

| situation | use | why |
|---|---|---|
| few sites (≤5) | `make_splits.py --strategy halve` | holding a site out removes an entire acquisition; the score then measures domain transfer, not segmentation |
| enough sites (≥6) | `--strategy sites` | the stronger claim: nothing shared |
| any | never a random tile split | oversampled tiles overlap |

> *Happened here:* holding out Bachsee_north gave **F1 exactly 0.0000** for every arm. It
> is 100% amber pixels (November beech canopy) against ~20% elsewhere — a colour domain
> nothing in training covered.

## Cross-validation across data sources

A single held-out site is one sample of "a site we have not seen". With 8 annotated sites
the corpus supports **leave-one-site-out**, and that is the default for any claim meant to
generalise.

**Fold by data source, not by tile.** The sources that matter here, in decreasing order of
how much they move results:

| axis | folds | when |
|---|---|---|
| **site** | one per annotated site | the default; site variance dominates everything |
| **species group** | beech / spruce / pine | when the claim is about a species model |
| **acquisition date** | leaf-on vs leaf-off | phenology is the largest colour effect measured |
| **resolution band** | ≤2.5 cm / 2.5–5 / >5 cm | when the claim is about scale robustness |

Rules that make the result mean something:

- **Report every fold.** With 4–8 folds the mean is fragile, and one fold returning zero
  (Bachsee_north does) would otherwise vanish into it.
- **Pair the arms.** Every arm sees identical folds, so the site effect cancels and the
  comparison is far tighter than unpaired runs.
- **State superiority as a fold count**, not a mean threshold: "positive in ≥3 of 4 folds"
  survives one bad fold; "mean > 0" does not.
- **Do not pool folds into one score.** Sites differ in size, density and difficulty, so a
  pooled number is dominated by whichever site contributed most tiles.
- **A fold where every arm scores ~0 is a tie, not a failure** — report it separately
  rather than averaging it in. It is also the most informative fold if one arm survives it.

Folds are expensive: arms × architectures × folds. Decide the fold axis before building
datasets, because each fold needs its own training set.

## Metrics

**Sanity-check a metric's range and direction before trusting a number from it.**

- AP must lie in [0, 1]. Integrating precision over *descending* recall gives a negative
  value — that is how a real bug announced itself here (AP −0.67).
- Comparing file sizes to measure model size breaks when the exporter writes external
  data: a 13 KB graph stub was compared against a 2.4 MB self-contained file.
- Skeleton endpoint counting is dominated by spurs off ragged traced outlines (reference
  masks scored ~33 endpoints/tile against a prediction's ~6). It reported the *better*
  model as worse. Use mean component major-axis length for continuity.

**Pin any metric you rely on with a test.** A metric that silently lies is worse than no
metric, because it produces a confident wrong conclusion.

**Choose metrics that survive a target change.** F1 cannot compare arms trained on
different labels; continuity, computed from the prediction alone, can.

## Claims and statistics

- **State n.** Single runs are the default here and that is fine, but a single run cannot
  support a claim about a ~0.5-point difference. Say "direction, not effect size".
- **Prefer paired designs.** Leave-one-site-out with per-fold reporting beats unpaired
  seeds, because site variance has dominated every comparison in this project.
- **Report per-fold values, never only a mean.** One catastrophic fold must not hide
  inside an average.
- **Recall and precision separately.** Every domain-shift failure here has been a recall
  collapse with precision intact — an under-reporting model, which is the benign mode and
  invisible in F1 alone.

## Figures

- **Random samples by default**, with the seed stated.
- If a figure is selected (largest difference, best cases), **label it as selected and
  show a random sample alongside it**. A selected figure alone misrepresents.
- Show failures. Overlay missed ground truth in a distinct colour rather than only hits.
- Outlines beat fills when the question is "is the label right" — a fill hides the
  evidence underneath it.

## Reporting

- Results go to **JSON, copied verbatim from each run's `test_results.md`**. Reports and
  PDFs render from that JSON and compute no metrics, so they cannot drift from what
  training actually reported. Adding an arm is then a data edit.
- **Record what was refused or dropped**, not just what succeeded — refused bridges,
  rejected tiles, repaired geometries. A high refusal rate is information about the data.
- **Write down the caveats you already know.** Resampling factors, tile-count differences,
  species composition, phenology. If it would change how someone reads the number, it
  belongs beside the number.

## Failing loudly

Silent partial success is the enemy. An empty split, a site contributing zero tiles, or an
invalid geometry aborting mid-run must raise with a message naming what to change.

> *Happened here:* Kaufland's val split produced **zero tiles** because its two blocks
> vanished under the buffer. The driver moved on without comment and it was found only by
> reading logs.

## Reproducibility

- Every dataset carries `tiles.jsonl` — source ortho, world centre, rotation, GSD, stem
  fraction. Without it a suspicious label cannot be traced back to the ground.
- `scripts/locate_tile.py --data-dir <DS> --tile N --crop out.png` re-cuts the footprint
  at native resolution, which is usually what settles a label question.
- Prefer a config file over CLI flags for `lr`, `seed` and per-architecture overrides, and
  commit the config next to the results.

# Scale jitter under a site holdout — four folds, reported individually

Follow-up to [`scale-augmentation-results.md`](scale-augmentation-results.md), which
measured scale jitter on splits that **share sites**. This asks whether the robustness
survives when the test site is absent from training. UNet, 3 paired seeds per arm per
fold, 24 runs. Numbers copied verbatim from [`assets/loso-*.json`](assets).

**Answer: it survives, but the evidence is thinner than the within-site result and one
fold is uninformative.** Jitter's scale curve is flatter in **4 of 4 folds**; only one
fold reaches the usual significance bar on its own.

## Two of these folds are not site holdouts

`tiles.jsonl` says **33% of Campus tiles overlap a Campus_Oberheide footprint** (centroids
183 m apart). They are the same forest imaged 2017-10-16 and 2022-02-26, and windthrown
stems do not move between acquisitions — so those are largely the same physical stems.

| fold | test tiles | nearest training site | what it measures |
|---|---:|---:|---|
| **Bachsee_north** | 800 | **5.68 km** | new-site transfer |
| **Kaufland** | 388 | **3.56 km** | new-site transfer |
| Campus | 800 | 0 m (33% overlap) | 4.4-year acquisition gap |
| Campus_Oberheide | 800 | 0 m (30% overlap) | 4.4-year acquisition gap |

Reported under those two headings, never pooled.

## Every fold, no mean

`spread` = each seed's peak-minus-worst F1 across 0.77× / 1.00× / 1.30×. Lower is more
scale-robust; the paired column counts seeds where jitter is flatter.

| fold | F1@1.00× fixed | jitter | spread fixed | spread jitter | flatter | t |
|---|---:|---:|---:|---:|:--:|---:|
| Bachsee_north | 0.0945 | 0.0434 | 0.0940 | 0.0778 | 1/3 | −0.15 |
| Kaufland | 0.6737 | **0.7065** | 0.1231 | 0.1116 | 2/3 | −1.31 |
| Campus | 0.5263 | 0.5290 | 0.1125 | 0.0886 | 2/3 | −1.41 |
| Campus_Oberheide | 0.6793 | 0.6792 | 0.0408 | **0.0272** | 3/3 | −5.52 |

**Direction is unanimous — all four mean spreads favour jitter — but only
Campus_Oberheide clears |t| ≥ 3 with unanimous seeds.** State this as a fold count, not a
mean: *flatter in 4/4 folds, individually significant in 1/4.*

### Bachsee_north is a dead fold

F1 **0.03–0.13** for every arm and seed. It is 100% amber pixels (November beech canopy)
against ~20% elsewhere, a colour domain nothing in training covers — the same collapse
recorded in `docs/process.md`. Neither arm segments it, so the fold is a **tie**, not
evidence for or against jitter, and its per-scale differences are ratios of noise. Had
these four folds been averaged, this one would have moved the mean while carrying no
information.

### Kaufland is the informative clean holdout

The only true spatial holdout where the model works at all, and jitter wins **at every
scale on every seed**:

| ratio | GSD | fixed | jitter | diff | sign | t |
|---:|---:|---:|---:|---:|:--:|---:|
| 0.77× | 2.254 | 0.5786 | 0.6237 | **+0.0451** | 3/3 | 4.37 |
| 1.00× | 2.930 | 0.6737 | 0.7065 | **+0.0328** | 3/3 | 2.61 |
| 1.30× | 3.811 | 0.7017 | 0.7353 | **+0.0336** | 3/3 | 9.58 |

Here jitter is worth **+3.3 F1 at the deployment scale** — an order of magnitude more than
the +0.2 measured within-site. Note the curve *rises* with coarseness on this fold, unlike
every other: Kaufland is the corpus's finest orthomosaic (2.09 cm/px native) and its
densest (stem fraction 0.063 against 0.028–0.038), so its peak sits outside the sampled
range. Its spread is therefore a lower bound.

## What the holdout costs

| | within-site (pooled blocks) | site holdout |
|---|---:|---:|
| F1 @ 1.00× | 0.78 | 0.04 – 0.71 |

Between 7 and 74 F1 points, depending entirely on which site. Any single-fold number here
would have been arbitrary — which is the argument for reporting all four.

## Caveats

- **n = 3 seeds per arm per fold.** Under domain shift the within-arm seed spread reaches
  **12 F1 points** (Campus fold), against 1–2 within-site. Absolute F1 differences below
  ~3 points are unreadable at this n; the spread statistic is within-model and survives
  better, which is why the claim rests on it.
- **Only 2 of 4 folds are true spatial holdouts**, and one of those is dead. The
  new-site evidence is effectively **one fold** — strong, but one.
- **Train sizes differ by fold** (2788–4000 tiles), inherent to leave-one-site-out.
- **Val comes from one site** in the Campus and Campus_Oberheide folds, since only those
  two sites are large enough to block-split at a 19.512 m footprint.
- **Corpus limit.** Four beech sites, two of them the same ground. A stronger claim needs
  annotated sites this corpus does not have — see [[winmol-label-scarcity]].

## Verdict

The within-site recommendation stands and is mildly strengthened: jitter never hurt in any
fold, flattened the curve in all four, and on the one clean holdout where segmentation
works at all it was worth +3.3 F1 at the deployment scale. But calling this "validated
across sites" would overstate a corpus with one informative holdout fold.

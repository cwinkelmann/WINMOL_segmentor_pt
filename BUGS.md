# Known bugs & open questions

Running list of things that look wrong, unexplained, or worth re-checking. Each entry separates
what is **measured** from what is **hypothesis**, so nobody inherits a guess as a fact.

---

## 1. The ~98% stage-1 (GenDS) F1 in the 2021 paper / docs is not reproducible, and looks like
   tile-level data leakage in GenDS100

**Status:** open. Raised by Christian (2026-07-22): *"I can't understand where the 98% F1 score for
stage 1 comes from which is documented in the 2021 paper and in the docs, aside from data leakage
in GenDS100."*

**What the paper actually reports.** The paper is Reder, Mund, Albert, Miranda, *Detection of
Windthrown Tree Stems on UAV-Orthomosaics Using U-Net Convolutional Networks*, Remote Sens. 2022,
14(1), 75 (doi 10.3390/rs14010075). Its headline results are **F1 73.9% (S1Mod10), 74.3%
(S1Mod50), 75.6% (S1Mod100)** against a non-pre-trained baseline of **72.6%** — i.e. **~74–76%,
not 98%**. Those are the stage-2 models evaluated on the *specific* windthrow dataset.
(MDPI and ResearchGate both return HTTP 403 to automated fetches, so this comes from indexed
abstract/summary text; the full Methods section has **not** been read. Anything below about the
split is therefore inference, not quotation.)

**Corrected: what the 10/50/100 suffix means.** An earlier version of this entry inferred from the
on-disk layout that the suffix meant *tiles per scene*. That was wrong. Per the paper, GenDS is a
**synthetic** dataset: the network was *"pre-trained with generic datasets, randomly combining stems
and background samples in a copy–paste augmentation"*, with **10, 50 and 100 augmentations per
annotated windthrown stem**. The disk layout is consistent with that reading: `GenDS10` is
**454 groups × exactly 10 tiles** (4540 tiles, `train_<a>_<b>.jpeg`), i.e. 454 annotated stems ×
10 augmentations each — not 454 scenes.

**Why leakage is still the likely explanation for a ~98% stage-1 number — and now more so:**

1. **With 100 augmentations per annotated stem, the same stem appears in 100 tiles.** A random
   split puts augmentations of the *same* annotated stem on both sides, so the validation set is
   near-copies of the training set. That is textbook leakage, and it gets worse as the augmentation
   count rises (10 → 50 → 100).
2. **The R training split is a plain random shuffle, with no grouping.** `input_pipeline.R` uses
   `dataset_shuffle(...)` with no grouping by the `<a>` field, so nothing prevents (1).
3. **Exact duplication across dataset folders is real here, not theoretical.** `SpecDS_local` is a
   strict subset of `SpecDS_ready` — all 410 training images are **byte-identical**
   (verified by md5). Any evaluation mixing those folders is scoring on training data.
4. **This repo's own benchmarks show how large the split-geometry effect is.** In
   `results/bamforests_experiments/`, the same models score val F1 **0.846–0.852** but held-out
   **cross-site** TestDS F1 **0.747–0.815** — a 0.04–0.10 drop caused purely by whether the split
   respects scene/site boundaries.
5. **Today's SpecDS run reproduces the optimistic regime:** UNet, 20 epochs, random *tile* split on
   `SpecDS_ready` → val F1 **0.9079**. High, on a split that does not respect scene boundaries.

**Hypothesis (not yet verified):** the ~98% is the **stage-1 validation F1 on GenDS itself** —
validating on copy-paste augmentations of the same annotated stems used for training — and is
therefore a memorization score, not a generalization estimate. It is consistent with the paper
reporting only 73.9–75.6% for the models that were actually evaluated on real windthrow data: the
98% would never have been the headline number, which is why it does not appear in the abstract.
A ~98% F1 on *real* stems would also be flatly inconsistent with everything measured in this repo
(best cross-site F1 ≈ 0.81).

**How to settle it definitively (cheap, ~1 h):** retrain stage 1 on `GenDS10` twice, everything else
identical — once with the current random tile split, once with a **grouped split** on the `<a>`
field of `train_<a>_<b>.jpeg`, so all 10 augmentations of one annotated stem land on the same side.
`tile_key()` in `learned_vectorization/build_teacher_dataset.py` already parses these names. If the
grouped split drops F1 substantially, leakage is confirmed and the gap quantifies it. The effect
should scale with the augmentation count, so `GenDS100` would show it strongest.

**Blockers:**
- `GenDS100` is not on this host. `/home/christian/hnee/WINMOL_segmentor/datasets/` has only
  `GenDS10`, `SpecDS_ready`, `SpecDS_local`, `SpecDS_smoke`.
- The paper's full text could not be retrieved (MDPI + ResearchGate both 403 automated fetches),
  so the exact split protocol and the location of any 98% figure remain **unverified**. Someone
  with the PDF should check Section "Materials and Methods" / the training tables directly.

**Note.** The paper's own stage-1 pre-training *is* copy-paste augmentation of stems onto
backgrounds — the same idea re-implemented in `training/mix_augment.py` (2026-07-22), where at
p=0.5 on SpecDS it did **not** improve held-out F1 (0.9014 vs 0.9079 baseline).

---

## 2. Random-tile validation splits overstate held-out F1 throughout this repo

**Status:** open / methodological. Directly related to #1.

`training/dataset.py::train_val_split` shuffles tile IDs randomly. Where several tiles come from one
orthomosaic (all datasets here), validation shares scenes with training. Every "val F1" produced
this way is optimistic by roughly the 0.04–0.10 seen in the BAMFORESTS site-split comparison.

Not a bug in the code — it does what it says — but val F1 should not be quoted as a generalization
number without saying which split produced it. The site-split reports
(`results/bamforests_experiments/COMBINED_SITESPLIT.md`) are the honest ones.

**Possible fix:** add an optional `group_by` to `train_val_split` that groups on a filename field,
and make the benchmark scripts use it.

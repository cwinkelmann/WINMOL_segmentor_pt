# Known bugs & open questions

Running list of things that look wrong, unexplained, or worth re-checking. Each entry separates
what is **measured** from what is **hypothesis**, so nobody inherits a guess as a fact.

---

## 1. The ~98% stage-1 (GenDS) F1 in the 2021 paper / docs is not reproducible, and looks like
   tile-level data leakage in GenDS100

**Status:** open. Raised by Christian (2026-07-22): *"I can't understand where the 98% F1 score for
stage 1 comes from which is documented in the 2021 paper and in the docs, aside from data leakage
in GenDS100."*

**Why the leakage explanation is plausible — measured facts:**

1. **`GenDS10` is 454 scenes × exactly 10 tiles each** (4540 tiles; `ls | sed 's/^train_//;
   s/_[0-9]*\.jpeg$//' | sort | uniq -c` gives 10 for every one of the 454 prefixes). The tile
   naming is `train_<scene>_<index>.jpeg`. So the trailing number in the dataset name is
   **tiles per scene** — which makes `GenDS100` ~100 tiles drawn from the *same* source
   orthomosaics, i.e. ~10x denser sampling of the same scenes.
2. **The R training split is a plain random shuffle, not scene-aware.** `input_pipeline.R` uses
   `dataset_shuffle(...)` with no grouping by `<scene>`. So tiles cut from the *same* orthomosaic
   land in both train and validation. At 100 tiles/scene those tiles are near-duplicates of each
   other (adjacent/overlapping crops of one image), so validation is effectively measuring
   memorization of scenes it has already seen.
3. **Exact duplication across dataset folders is real here, not theoretical.** `SpecDS_local` is a
   strict subset of `SpecDS_ready` — all 410 training images are **byte-identical**
   (verified by md5). Any evaluation mixing those folders is scoring on training data.
4. **This repo's own benchmarks show how large the split-geometry effect is.** In
   `results/bamforests_experiments/`, the same models score val F1 **0.846–0.852** but held-out
   **cross-site** TestDS F1 **0.747–0.815** — a 0.04–0.10 drop caused purely by whether the split
   respects scene/site boundaries.
5. **Today's SpecDS run reproduces the optimistic regime:** UNet, 20 epochs, random *tile* split on
   `SpecDS_ready` → val F1 **0.9079**. High, on a split that does not respect scene boundaries.

**Hypothesis (not yet verified):** a ~0.98 F1 on GenDS100 is an *in-scene* number — train and
validation tiles come from the same orthomosaics — and is not a generalization estimate. It should
not be compared against, or cited alongside, cross-site numbers.

**How to settle it definitively (cheap, ~1 h):** retrain stage 1 on `GenDS10` twice with everything
else identical — once with the current random tile split, once with a **scene-grouped split**
(group by the `<scene>` field of the filename, so all 10 tiles of a scene fall on the same side).
`tile_key()` in `learned_vectorization/build_teacher_dataset.py` already parses these names. If the
grouped split drops F1 substantially, leakage is confirmed and the gap quantifies it. Running it on
GenDS100 (if it can be located — it is **not** on this host, only `GenDS10`) would test the
stronger claim directly, since the leakage should scale with tiles-per-scene.

**Blocker:** `GenDS100` is not present on this machine. `/home/christian/hnee/WINMOL_segmentor/
datasets/` has only `GenDS10`, `SpecDS_ready`, `SpecDS_local`, `SpecDS_smoke`.

**Caveat:** I have not read the 2021 paper, so I cannot confirm what protocol it reports — the
above explains how such a number could arise from this data layout, it does not prove that is what
happened.

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

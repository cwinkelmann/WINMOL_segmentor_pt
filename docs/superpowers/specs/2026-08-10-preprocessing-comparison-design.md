# Training-preprocessing comparison — design

**Goal.** Establish whether the tile-extraction variants written in this repo perform
better than, or at least no worse than, the original R generator
(`docs/reference/241202_Training_data_SpecDS.R`) — on a yardstick that neither variant's
own preprocessing can flatter.

**Status.** Design approved 2026-08-10. Non-inferiority margin −1.0 F1 points, agreed.

---

## Why a naive comparison fails

Each variant produces a *different* test set. The R port cuts 15 m footprints at
2.93 cm/px; the fixed-metre sampler cuts 10.24 m at 2 cm/px; native cuts 1024 source
pixels, which is 21 m of ground at Kaufland and 65 m at Campus. Comparing F1 across those
is comparing scores from three different exams.

This project has already produced two results that looked meaningful and were not, both
for this reason: a modal-trained and an amodal-trained model scored against their own
labels (the modal target is strictly easier), and a whole-site holdout that returned F1
exactly 0.0000 because the held-out site's colour domain appeared nowhere in training.

The design below exists to make that class of mistake impossible rather than to detect it
afterwards.

## The yardstick — full-orthomosaic inference

Every model is scored the way the WINMOL Analyzer runs, on a complete held-out
orthomosaic:

1. cut `ceil(tile_size / pixel_size)` source pixels, resize to 512 — this is
   `ExecutionPlan.py:95-97` in the Analyzer, reproduced exactly;
2. predict, stitch with overlap blending, threshold;
3. compare against the site's rasterised stem map from `rasterize_annotations.py`.

No variant's tiling choice appears anywhere in this path, so it cannot favour one arm. It
is also the deployment path, which means a gain measured here is a gain the Analyzer sees.

### Evaluation is masked to the AOI

**Metrics are computed only inside the windthrow polygon (`*_AOE.shp`).** Outside it the
stems are real but were never digitised, so scoring the whole raster would count correct
detections as false positives — and would penalise the *better* model hardest, inverting
the result. This is the single most important correctness detail in the design.

### Reported per evaluation

| metric | why |
|---|---|
| F1, precision, recall | the headline, at the Analyzer's default `tile_size=15` |
| AP | threshold-free; separates ranking quality from calibration |
| best-F1 threshold | calibration drift is large on unseen sites (0.036 vs 0.5 observed) |
| mean component length | fragmentation, which is what the vectoriser downstream consumes |

`tile_size` is swept over 10 / 12.5 / 15 / 17.5 / 20 m to measure scale robustness on the
same harness. The current model loses 5.2 F1 points across a ±30% zoom range, asymmetric:
−5.2 zooming out, −1.9 zooming in.

## The three arms

Extraction only. Amodal masks are excluded — they change *what* is predicted, not how
tiles are cut, and belong to a separate question that already has its own result.

| arm | extraction | training |
|---|---|---|
| **A** — R port | `r_sample_tiles.py`, 15 m → 512 px, literal −11 m buffer, R's acceptance rule | default loader |
| **B** — fixed-metre | `sample_training_tiles.py --extent 10.24`, every site normalised to 2 cm/px | default loader |
| **C** — native + multiscale | `--native-px 1024`, no resampling, each site keeps its own GSD | `--multiscale --crop-min-px 400 --crop-max-px 1024 --eval-tiling` |

Held constant across arms: stem polygons, geometry repair (`fix_geometries.py`), epochs,
batch size, optimiser, seed, and the held-out site per fold. Only extraction differs.

Arm A reproduces R's acceptance rule deliberately, including the part worth avoiding: it
sums the *whole* area of every polygon intersecting the footprint rather than the part
inside it, so a tile can qualify on a stem that mostly falls outside.

## Protocol

**Leave-one-site-out over the four beech sites**, 3 arms × 2 architectures × 4 folds =
**24 runs**, roughly 8 GPU-hours on carrot.

The `beech/` folder holds five orthomosaics; `20171123_EW_WW_Bachsee_south` has imagery but
no annotations at all and is therefore not a fold. Four annotated sites, four folds.

| fold | train | evaluate on full ortho |
|---|---|---|
| 1 | Oberheide, Bachsee_north, Kaufland | **Campus** |
| 2 | Campus, Bachsee_north, Kaufland | **Campus_Oberheide** |
| 3 | Campus, Oberheide, Kaufland | **Bachsee_north** |
| 4 | Campus, Oberheide, Bachsee_north | **Kaufland** |

Architectures: **UNet** (what the Analyzer ships, and what the Keras HDF5 drop-in
requires) and **HRNet** (the strongest model measured, 3.2–5.5 points ahead of UNet).
Running both answers whether preprocessing gains hold generally or only rescue the weaker
model — the recurring pattern with both synthetic pretraining and ImageNet weights.

Comparison is **paired per fold**: all arms see identical folds, so the site effect — the
dominant source of variance all session — cancels.

## Acceptance criteria

Stated as non-inferiority against arm A, on mean paired difference across the four folds:

- **non-inferior** if mean paired ΔF1 > **−1.0 points**
- **superior** if the paired difference is positive in **at least 3 of 4 folds**
- **inferior** otherwise

Per-fold values are reported alongside the mean, always. A single catastrophic fold must
not hide inside an average, and with n=4 the mean is fragile.

## Risks, designed for rather than discovered

**Bachsee_north may return ~0 for every arm.** It is 100% amber pixels (November beech
canopy) against ~20% elsewhere; a previous holdout of this site scored exactly 0.0000. In
a paired design that fold is a tie and contributes nothing — *unless* arm C survives it,
in which case it is the most valuable result in the study. Kept for that reason, and
reported separately rather than folded into the mean if every arm scores near zero.

**Campus_Oberheide is 37% beech** (40% spruce, 18% Douglas fir) despite living in the
`beech/` folder. Its fold is reported but flagged; it measures something closer to a mixed
stand.

**Native tiles are scarce.** A 1024 px tile at Campus's 6.39 cm/px covers 65 m and needs a
46 m inward buffer, leaving 22,927 m² usable against 36,542 m² at 10.24 m footprints. Arm
C therefore trains on fewer, larger tiles — the sampling moves from the generator into the
loader's crops. Tile counts per arm are reported, since "fewer tiles" is a confound if not
declared.

**Native tiling is incompatible with site-halving.** Campus's *halved* AOI (29,041 m²)
vanishes under the 46 m buffer. Arm C uses whole-site splits; all arms use the same folds,
so this does not bias the comparison, but it does mean the study cannot also report a
halved-split result.

**`seed` and `lr` are not exposed.** Both live in `TrainConfig`. A config mechanism is a
prerequisite for reproducibility and is task 1 of the implementation plan, not an
afterthought.

## Deliverables

1. `scripts/eval_orthomosaic.py` — the yardstick, with tests covering AOI masking, the
   Analyzer's tile-size arithmetic, and stitching.
2. `scripts/run_preprocessing_comparison.py` — builds the 12 datasets, runs 24 trainings,
   collects results into one JSON.
3. `docs/preprocessing-comparison.md` — per-fold table, the verdict against the criteria
   above, and the `tile_size` sweep.
4. A PDF via the existing `report_synth_pretraining.py` pattern: results read from JSON,
   never recomputed, so the report cannot drift from what training reported.

## Out of scope

Amodal masks; synthetic pretraining; architectures beyond UNet and HRNet; the spruce and
pine sites; any change to the ONNX contract.

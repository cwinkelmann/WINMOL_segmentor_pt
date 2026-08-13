# Working process

## Splits

**Standard:** no tile footprint may overlap between train, val and test.
Spatial autocorrelation inside a split is accepted and expected.

`make_splits.py --strategy blocks` satisfies this by construction: the AOI is cut into a
grid, whole blocks are dealt to splits, and each block is then shrunk by the footprint
half-diagonal. Two centres on opposite sides of a shared edge are therefore at least
`extent · √2` apart — the distance below which two rotated footprints can share a pixel.

**Verification is a pass/fail check, not a discussion.** `scripts/plot_splits.py` prints
the measured closest centre-to-centre distance per site and exits non-zero if a site is
missing from a split. Run it after every build; report the number once.

| | requirement |
|---|---|
| footprint overlap between splits | none — closest centre distance > `extent · √2` |
| spatial autocorrelation within a split | accepted |
| every site in every split | yes, unless the AOI is too small to cut (record in `splits.json`) |
| provenance | `tiles.jsonl` per split: site, centre, angle, GSD, stem fraction |

A site whose AOI cannot yield 12+ usable blocks goes wholesale into one split via
`whole_split`, recorded in `splits.json`. No further comment needed.

## Scale

**Train at the scale the Analyzer serves.** Effective GSD is `extent_m / tile_px` in
training and `tile_size / 512` in the Analyzer. These must match, or the model is served
at a different resolution than it learned.

Default `tile_size = 15` → **2.93 cm/px** → build with `--extent 15 --tile-px 512`.

Measured on Kaufland, AOI-masked, same model family:

| trained at | served at | F1 |
|---|---|---:|
| 2.00 cm/px | 2.93 cm/px (mismatched) | 0.7389 |
| 2.00 cm/px | 2.00 cm/px (matched) | 0.8091 |
| 2.93 cm/px | 2.93 cm/px (matched, default) | **0.8844** |

Ship the `tile_size` alongside any model that is not trained at 2.93 cm/px.

**Jitter the training footprint ±30%** — `--multiscale --crop-min-px 394 --crop-max-px 666`
on a 666 px source. Measured on two architectures, 3 paired seeds each: it flattens the
F1-vs-GSD curve by 0.87 F1 on HRNet and 1.88 on UNet, and costs nothing at the native
scale. See [`scale-augmentation-results.md`](scale-augmentation-results.md). Matching the
scale is still worth far more than being robust to it — do both.

## Evaluation

- Tile-level F1 for comparing training runs.
- **Full-orthomosaic inference masked to the AOI** for anything claiming deployment
  performance. Outside the windthrow polygon stems are real but undigitised; scoring the
  whole raster is meaningless (measured: 0.2974 vs 0.8844 on the same prediction).
- Report precision and recall separately.

## Reporting

- Numbers come from each run's own `test_results.md`, collected by
  `scripts/collect_results.py` into JSON. Reports render from that JSON.
- State the test set and the training scale beside every figure.
- One line per caveat, in a caveats section. Not inline.

## Pipeline order

`fix → sample → split`, enforced by `make_splits.py`. Geometry repair happens first
because GEOS aborts on the first set operation touching an invalid ring.

## Environments

| what | where |
|---|---|
| corpus | `/Volumes/storage/Datasets/Winmol/training_data/WINMOL_Trainings_Data/GIS` |
| derived tiles | `data/` in this repo, git-ignored, regenerable from `configs/*.json` |
| training | carrot `/raid/cwinkelmann/winmol` — check GPUs are free before launching |
| Analyzer | `~/hnee/WINMOL_Analyzer`, needs Python ≥ 3.10 |

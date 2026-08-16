# Data inventory

Every number here was measured from the files on 2026-08-14, not transcribed from earlier
notes. Rasters read with `rasterio`, vectors with `fiona`; areas are in each layer's own
CRS units (m²). Source tree: `/Volumes/storage/Datasets/Winmol/training_data/WINMOL_Trainings_Data/GIS`
(mounted on T14 as `/data/mnt/storage/...`).

## The headline number

**1.18 ha of labelled stem exists in total** — 0.98 ha across 8 annotated sites in the main
GIS tree, plus 0.20 ha from the Tegel Revier 12/13 surveys (section 6). 31 orthomosaics are
held; **21 of them have no stem annotations at all**. Labelling, not imagery, is the binding
constraint on this project.

## 1. Orthomosaics — all 29

Year is the acquisition date encoded in the filename (`YYYYMMDD`).

### Beech (10 rasters)

| acquisition | site | GSD | size | CRS |
|---|---|---:|---:|---|
| 2017-10-16 | EW_WW_Campus | 6.39 cm | 147 MPx | EPSG:32633 |
| 2017-11-14 | EW_WW_Bachsee_north | 4.08 cm | 214 MPx | EPSG:32633 |
| 2017-11-23 | EW_WW_Bachsee_south | 3.95 cm | 257 MPx | EPSG:32633 |
| 2021-07-06 | EW_WW_Kaufland | **2.09 cm** | 125 MPx | EPSG:25833 |
| 2022-02-26 | EW_WW_Campus_Oberheide | 3.36 cm | **1911 MPx** | EPSG:25833 |
| 2022-02-26 | EW_WW_Campus_Oberheide (clip) | 3.36 cm | 373 MPx | EPSG:25833 |
| 2023-11-06/07 | Altmann_Jauerling (AT) | 3.30 cm | 540 MPx | EPSG:3416 |
| 2024-07-19 | FR17203_Abt-128_129_30 | 3.52 cm | 280 MPx | **none** |
| 2025-03-06 | 172_03_Kohlleiten_Meislingeramt (AT) | 3.52 cm | 280 MPx | EPSG:3416 |
| 2025-03-27 | 171_4_Kraking_Windwurf | 3.99 cm | 76 MPx | EPSG:25833 |

### Spruce (13 rasters)

| acquisition | site | GSD | size | CRS |
|---|---|---:|---:|---|
| 2022-02-09 | Bremerhagen 1–8 | 1.23 – 1.84 cm | 302 – 802 MPx | EPSG:25833 |
| 2022-02-12 | Barnekow 1, 3, 5, 6 | 1.51 – 1.76 cm | 297 – 870 MPx | EPSG:25833 |
| 2022-02-12 | Barnekow 2, 4 (**IR**) | 1.75 / 2.07 cm | 70 / 141 MPx | EPSG:25833 |
| 2022-03-09 | EW_WW_Kaltes_Wasser_2 | 1.64 cm | 815 MPx | EPSG:25833 |
| 2022-03-09 | EW_WW_Ruheforst | 1.61 cm | 796 MPx | EPSG:25833 |

### Pine (3 rasters)

| acquisition | site | GSD | size | CRS |
|---|---|---:|---:|---|
| 2022-03-09 | EW_WW_Stromtrasse | 2.75 cm | 625 MPx | EPSG:25833 |
| 2022-03-29 | Alt_Madlitz_plot123 | 2.58 cm | 603 MPx | EPSG:25833 |
| 2022-03-29 | Alt_Madlitz_plot4 | 3.24 cm | 134 MPx | EPSG:25833 |

**GSD spans 1.23 – 6.39 cm/px, a 5.2× range.** The Analyzer serves at a fixed
`tile_size / 512` (2.93 cm/px by default), so most of this corpus is nowhere near the
serving scale unless resampled — which is what `--extent`/`--tile-px` extraction does.

## 2. Stem annotations — only 8 sites carry them

| acquisition | site | species | polygons | stem area | AOI | CRS |
|---|---|---|---:|---:|---:|---|
| 2017-10-16 | Campus | beech | 822 | 2,908 m² | 118,808 m² | 32633 |
| 2017-11-14 | Bachsee_north | beech | 136 | 451 m² | 31,527 m² | **25833** |
| 2021-07-06 | Kaufland | mixed | 125 | 250 m² | 5,393 m² | 25833 |
| 2022-02-26 | Campus_Oberheide | **mixed** | 490 | 1,917 m² | 111,631 m² | 25833 |
| 2022-02-09 | Bremerhagen_3 | spruce | 158 | 180 m² | 2,566 m² | 25833 |
| 2022-02-12 | Barnekow_3 | spruce/pine | 358 | 741 m² | 15,516 m² | 25833 |
| 2022-02-12 | Barnekow_5 | spruce/pine | 1,012 | 1,826 m² | 27,788 m² | 25833 |
| 2022-02-12 | Barnekow_6 | spruce | 741 | 1,528 m² | *none* | 25833 |
| | **total** | | **3,842** | **9,802 m² = 0.98 ha** | | |

A ninth file, `202171114_EW_WW_Bachsee_north`, is a **typo'd duplicate** of the 2017-11-14
layer — the same 136 polygons and 451 m². It has no matching orthomosaic. Ignore it; do
not "fix" the name and pair it, or those stems are counted twice.

### Species composition of the labelled polygons

| code | species | polygons |
|---|---|---:|
| GFI | Gemeine Fichte (Norway spruce) | 1,986 |
| RBU | Rotbuche (European beech) | 1,211 |
| GKI | Gemeine Kiefer (Scots pine) | 231 |
| DGL | Douglasie (Douglas fir) | 97 |
| GBI | Gemeine Birke (birch) | 10 |
| RGU | Roterle/other | 1 |
| — | no species attribute | ~295 |

**No pine site is annotated.** The 231 GKI polygons are pine stems inside the two Barnekow
spruce stands, not a pine acquisition — all three pine orthomosaics have zero labels.

**"Beech" sites are not pure beech.** Campus_Oberheide is GFI 199 / RBU 190 / DGL 92, so
the beech models in this repo are trained on a mixed stand that happens to sit in the beech
folder. Kaufland is RBU 110 / GBI 10 / DGL 5.

## 3. The four beech sites used for training

These are the sites behind every result in `scale-augmentation-results.md` and
`scale-augmentation-loso.md`.

| site | date | season | GSD | hue | saturation | notes |
|---|---|---|---:|---:|---:|---|
| Campus | 2017-10-16 | autumn | 6.39 cm | 15° | 0.19 | coarsest raster in the corpus |
| Bachsee_north | 2017-11-14 | **late autumn** | 4.08 cm | 25° | **0.73** | amber canopy; CRS mismatch vs its ortho |
| Campus_Oberheide | 2022-02-26 | leaf-off | 3.36 cm | 32° | 0.25 | 33% footprint overlap with Campus |
| Kaufland | 2021-07-06 | **high summer** | 2.09 cm | 98° | 0.32 | finest raster; smallest AOI |

Two facts that shaped the experiments:

- **Campus and Campus_Oberheide are the same forest**, imaged 4.4 years apart; 33% of
  Campus's tiles overlap a Campus_Oberheide footprint (centroids 183 m apart). They are not
  independent sites, so a "leave-one-site-out" fold on either is really a temporal holdout.
- **Bachsee_north is the only late-autumn acquisition**, at 2.4–3.8× the saturation of
  every other site. Held out it scores F1 ~0.09; trained on its own west half it reaches
  **0.62** on its east half. Removing it from training costs the other folds up to 9.5 F1.

## 4. Published datasets (Zenodo record 5682580, Reder et al.)

Verified byte-identical against the record.

| dataset | tiles | tile size | on disk | role |
|---|---:|---:|---:|---|
| GenDS10 | 4,540 | 2000×2000 | 2.3 GB | synthetic stage-1 pretraining |
| SpecDS | 454 | 313×313 | 12 MB | species-specific stage-2 |
| TestDS | 117 | 313×313 | 3.2 MB | the papers' test set |

Two caveats measured here. **GenDS10's 2000 px tiles carry roughly 500–650 px of real
information** — a round trip through 512 costs 1.45 grey levels against SpecDS's 3.73, so
they are largely interpolation; effective GSD ≈ 4.6 cm/px. And **TestDS contains 117 tiles,
not the 106 stems the 2022 paper describes**.

`GenDS10_512` (4,540 tiles, 736 MB) is our pre-resized copy: the loader's exact resize
applied once instead of every epoch, verified to reproduce the loader's array to 0.0000/255.

## 5. Datasets built in this repo

All carry `tiles.jsonl` (source ortho, world centre, rotation, GSD, stem fraction), which
is what makes any tile traceable back to the ground via `scripts/locate_tile.py`.

| dataset | tiles | tile px | extent | GSD | purpose |
|---|---:|---:|---:|---:|---|
| BeechScale666 | 3,388 / 400 / 400 | 666 | 19.512 m | 2.9297 cm | scale-augmentation arms |
| loso/site_all/Campus | 800 | 666 | 19.512 m | 2.9297 cm | LOSO fold source |
| loso/site_all/Campus_Oberheide | 800 | 666 | " | " | " |
| loso/site_all/Bachsee_north | 800 | 666 | " | " | " |
| loso/site_all/Kaufland | **388** | 666 | " | " | " |
| loso/site_tv/{Campus,Oberheide} | 1,600 + 300 each | 666 | " | " | block train/val |

Kaufland yields only 388 tiles because its AOI is 5,393 m² and a 19.512 m footprint holds
13.80 m in from every edge.

## 6. Tegel Revier 12 / 13 — added 2025-07, measured 2026-08-16

Two surveys held separately from the GIS tree above, digitised as five sample plots.
Detail and results in [`tegel-r12-r13-results.md`](tegel-r12-r13-results.md).

| ortho | acquisition | GSD | size | CRS | bands |
|---|---|---:|---:|---|---|
| Revier 12 `result_Res1.2_webp.tif` | 2025-07 | **1.20 cm** | 339k × 334k px, 12 GB | EPSG:32633 | RGBA |
| Revier 13 `result_Res1.3_COG.tif` | 2025-07 | **1.28 cm** | 393k × 335k px, 19 GB | EPSG:32633 | RGB |

| plot | polygons | stem area | AOI |
|---|---:|---:|---:|
| R12-P1 | 178 | 215 m² | 14,913 m² |
| R12-P2 | 552 | 891 m² | 23,080 m² |
| R12-P3 | 299 | 501 m² | 8,692 m² |
| R13-P1 | 232 | 248 m² | 8,865 m² |
| R13-P2 | 123 | 141 m² | 13,436 m² |
| **total** | **1,384** | **1,996 m² = 0.20 ha** | |

**This raises the corpus from 0.98 ha to 1.18 ha of labelled stem, +20%**, and adds the
first substantial pine (153 polygons), birch (120) and ash (41) annotations. Labels are
`EPSG:25833` against `EPSG:32633` orthos; registration verified at zero offset on all
plots. Species names are free text with the same variant problem as the older corpus
(`POPLAR` / `WHITE POPLAR` / `SILVER POPLAR`).

## 7. Defects to handle

- **Mixed CRS.** Campus and Bachsee north/south orthos are EPSG:32633; everything else is
  EPSG:25833 — the same UTM zone on different datums, so a silent mismatch shifts masks by
  metres rather than failing. `20171114_EW_WW_Bachsee_north` has annotations in 25833 and
  its ortho in 32633. Both scripts reproject explicitly, and this was **verified**: every
  beech site's polygons peak at zero offset on its own raster grid
  (`scripts/check_registration.py`), residual under 4 cm.
- **One orthomosaic has no CRS at all** (`20240719_FR17203_Abt-128_129_30.tif`). It is
  unannotated, so nothing depends on it yet.
- **`202171114_EW_WW_Bachsee_north`** — typo'd duplicate, see above.
- **Species case variants**: `RBU`/`rBU`/`RBu`, `GFI`/`GFi`/`gFI`, `DGL`/`DGl`/`dGL`, plus
  single-record typos `GIF` and `GFIU`. Match case-insensitively or silently drop records.
  ~295 polygons carry no species at all.
- **No DEM/DSM/CHM anywhere** in the corpus — height is not available as a channel.
- **Two Barnekow rasters are infrared** (2 and 4), not RGB; they are unannotated.
- **`gends10_ready`** is a locally derived copy that is not trusted; use the Zenodo
  originals.

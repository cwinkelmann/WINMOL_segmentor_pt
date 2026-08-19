#!/usr/bin/env bash
# Build the anti-aliasing 2x2 datasets.
#
# WHY: the Analyzer resamples two different ways. utils/Prediction.py::_resize_batch calls
# skimage order=3 with anti_aliasing=False; its stream path (Config default
# stream_prediction=True) uses rasterio Resampling.cubic, which filters. They disagree on
# every DOWNsampled tile. `--antialias on|off` reproduces both exactly.
#
# WHY SPRUCE, NOT BEECH: anti-aliasing only does anything when a tile is downsampled, and
# at the Analyzer's default scale (tile_size 15 / 512 = 2.93 cm/px) the beech orthos are
# too coarse to downsample at all --
#
#   Campus    6.39 cm/px -> x0.46 UPsample   (AA is a no-op)
#   Oberheide 3.36 cm/px -> x0.87 UPsample   (AA is a no-op)
#   Kaufland  2.09 cm/px -> x1.40 downsample, but only 2,190 m2 survives the buffer
#
# A beech 2x2 would have measured nothing. The spruce sites are 1.58-1.76 cm/px, so every
# split here downsamples x1.67-x1.85 with real area behind it. This changes the species,
# which is fine: the question is about a resampling filter, not about bark.
set -euo pipefail

PY=${PY:-.venv/bin/python}
GIS=${GIS:-/Volumes/storage/Datasets/Winmol/training_data/WINMOL_Trainings_Data/GIS}
OUT=${OUT:-$HOME/winmol_data}
S=scripts/sample_training_tiles.py

TRAIN_SITE=20220212_Barnekow_5      # 1.58 cm/px, x1.85 down, 18,985 m2 usable, 1012 stems
TEST_SITE=20220212_Barnekow_3       # 1.76 cm/px, x1.67 down, 10,749 m2 usable,  358 stems

o() { echo "$GIS/Orthomosaics/spruce/$1_ortho.tif"; }
s() { echo "$GIS/Training_Data/$1.shp"; }
a() { echo "$GIS/Polygone/$1_AOE.shp"; }

# Identical between arms except --antialias: same sites, same seeds, same block
# assignment. The pair differs by the resampling filter and nothing else.
for AA in on off; do
  D="$OUT/BeechAA_$AA"; rm -rf "$D"
  echo "=== building $D (antialias=$AA) ==="

  # train / val: leak-free block split of Barnekow_5 -- same --split-seed, so no block
  # feeds both sides.
  for SPLIT in train val; do
    $PY $S --ortho "$(o $TRAIN_SITE)" --stems "$(s $TRAIN_SITE)" --aoi "$(a $TRAIN_SITE)" \
           --out "$D/$SPLIT" --extent 15 --tile-px 512 --antialias "$AA" --seed 1 \
           --block-size 50 --split "$SPLIT" --split-fractions 0.7 0.3 0.0 --split-seed 1
  done

  # test: Barnekow_3 whole, an unseen site. Built in BOTH filters (each arm gets its own
  # copy) so every model can be scored on its own filter AND on the other one -- that
  # off-diagonal is what the Analyzer's two disagreeing paths actually cost.
  $PY $S --ortho "$(o $TEST_SITE)" --stems "$(s $TEST_SITE)" --aoi "$(a $TEST_SITE)" \
         --out "$D/test" --extent 15 --tile-px 512 --antialias "$AA" --seed 1

  for x in train val test; do
    printf '  %-6s %5s tiles\n' "$x" "$(ls "$D/$x/train" | wc -l)"
  done
done

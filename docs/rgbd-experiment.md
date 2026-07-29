# RGB vs RGBD experiment (synthetic depth)

Prereqs: `scripts/simulate_depth.py` (this plan) and `--rgbd`
(docs/superpowers/plans/2026-07-29-rgbd-input.md) are implemented.

## Data

    python scripts/simulate_depth.py --dataset <DS> --seed 1
    # leakage-control copy: same images, depth from the WRONG mask
    cp -R <DS> <DS>-mismatch && rm -rf <DS>-mismatch/depth
    python scripts/simulate_depth.py --dataset <DS>-mismatch --seed 1 --mismatch

## Runs (identical seed => identical train/val split; only the input differs)

    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgb \
        --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgbd --rgbd \
        --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS>-mismatch --out-dir output/exp-rgbd-mismatch \
        --rgbd --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps

## Reading the result

Compare val F1/IoU across the three runs:

| Outcome | Interpretation |
|---|---|
| rgbd > rgb, mismatch ~ rgb | depth is genuinely fused (what we hope for) |
| rgbd > rgb, mismatch > rgb | gain is label leakage through synthetic depth |
| rgbd ~ rgb | net ignores depth; check channel-4 first-conv weights |

Caveat: synthetic depth is derived from the masks, so even with dropout and
distractors the rgbd score is optimistically biased. This experiment validates
plumbing and fusion behavior, NOT real-sensor gains.

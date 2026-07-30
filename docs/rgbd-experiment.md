# RGB vs RGBD experiment (synthetic depth)

Prereqs: `scripts/simulate_depth.py` (this plan) and `--rgbd`
(docs/superpowers/plans/2026-07-29-rgbd-input.md) are implemented.

## Data

    python scripts/simulate_depth.py --dataset <DS> --seed 1
    # leakage-control copy: same images, depth from the WRONG mask
    cp -R <DS> <DS>-mismatch && rm -rf <DS>-mismatch/depth
    python scripts/simulate_depth.py --dataset <DS>-mismatch --seed 1 --mismatch

## Runs (TrainConfig seed defaults to 1 -- identical train/val split across runs; only the input differs. NOTE: no --seed CLI flag exists on main)

    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgb \
        --arch deeplabv3plus --encoder resnet34 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgbd --rgbd \
        --arch deeplabv3plus --encoder resnet34 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS>-mismatch --out-dir output/exp-rgbd-mismatch \
        --rgbd --arch deeplabv3plus --encoder resnet34 --epochs 50 --device mps

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

## Results — 2026-07-30 (thinkpad-t14, RTX 4080 SUPER)

deeplabv3plus/resnet34, seed 1 (default), 50 epochs w/ early stopping, SpecDS
(454 pairs, 80/20 split), held-out TestDS = beech/TestDS (117 tiles, synthetic
depth). Depth: scripts/simulate_depth.py --seed 1. ~7 min/arm.

| arm | TestDS F1 | P | R | loss |
|---|---|---|---|---|
| exp-rgb | 0.7381 | 0.7468 | 0.7296 | 0.3748 |
| exp-rgbd | **0.8915** | 0.9047 | 0.8787 | 0.1530 |
| exp-rgbd-mismatch | 0.7546 | 0.7746 | 0.7356 | 0.3525 |

Verdict per the table above: **rgbd > rgb (+0.153), mismatch ~ rgb (+0.017)**
=> the network genuinely fuses depth; the mismatch control learned to ignore
uninformative depth rather than exploiting a leakage shortcut. Absolute rgbd
numbers remain optimistic (test depth is mask-derived); real-sensor gains TBD.

Ops notes: torch 2.13 needs `onnxscript` for ONNX export (installed on t14);
its dynamo exporter emits opset 18 + external weights (model.onnx.data) — for
analyzer-bound models keep exporting from the torch 2.8 env until reconciled.

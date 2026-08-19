# Documentation index

What each document is, and **whether its conclusions still stand**. Several early results
were later refuted; those are marked so nobody re-cites them. When two documents disagree,
the one listed as *current* wins.

Citations marked **(private helper repo)** point at experiment tooling, pre-registered
specs and superseded results that were moved out when this repo was prepared for public
release. The measurements they produced are unaffected — the numbers quoted here were
copied verbatim from run output and still stand.

## Read these first

| document | covers | status |
|---|---|---|
| [`process.md`](process.md) | The rules: splits, scale, evaluation, reporting. The short one. | **current** |
| [`data-inventory.md`](data-inventory.md) | Every orthomosaic and annotation layer, measured. 1.18 ha labelled total. | **current** |
| [`reder-method-gaps-closed.md`](reder-method-gaps-closed.md) | Corrections to the published WINMOL method, read from R and Python source. | **current** |
| [`WINMOL-report.pdf`](WINMOL-report.pdf) | Everything above assembled for reading end to end. Rebuild with `scripts/build_report_pdf.sh`. | **current** |

## Results — current

| document | headline | status |
|---|---|---|
| [`scale-augmentation-results.md`](scale-augmentation-results.md) | ±30% footprint jitter flattens the GSD curve 0.87 F1 (HRNet) / 1.88 (UNet) at no cost at the serving scale. | **current** |
| [`scale-augmentation-loso.md`](scale-augmentation-loso.md) | Site holdouts; Bachsee_north explained; **strong hue augmentation takes it from F1 0.042 to 0.688**. | **current** |
| [`tegel-r12-r13-results.md`](tegel-r12-r13-results.md) | New Tegel survey: zero-shot 0.76, +3.5–6.2 F1 from training on it, fine-tuning adds nothing. | **current** |
| `unet_vs_ronneberger.md` *(private helper repo)* | Our UNet vs the paper and vs the R port, including the zero-gradient `k_round` loss. | **current** |
| [`2026-07-21-cpu-inference-speedup-results.md`](2026-07-21-cpu-inference-speedup-results.md) | CPU inference throughput work. | current |
| [`2026-07-02-segmentor-pytorch-onnx-design.md`](2026-07-02-segmentor-pytorch-onnx-design.md) | The cross-repo ONNX contract and package boundary. | current |

## Results — superseded in part

**These documents now live in the private helper repo.** They are listed here because the
corrections matter more than the documents: anyone reading them there must not re-cite the
voided numbers. Each carries a correction banner at its top.

| document *(all in the private helper repo)* | what is still good | what is void |
|---|---|---|
| `synthetic-pretraining-beech.md` | The synthetic-pretraining measurements themselves. | Its explanation of why a Bachsee holdout gives F1 0.0 (colour domain / label doubt). |
| `preprocessing-comparison.md` | The fixed-metre arms' relative ordering. | **Arm C (native + multiscale)** — Bachsee_north was the validation site, so checkpoint selection ran on a signal pinned near zero. |
| `session-report.md` | The narrative of what was run. | Same two items as above. |
| `data_training_concept.md` | The planning framing. | Any claim resting on the lost native arm. |
| `training-data-from-annotations.md` | The defect list — mixed CRS, typo'd duplicate layer, species variants. All verified. | The "one per split gives F1 0.0000" framing as evidence about the data. |

## Supporting

| document | covers |
|---|---|
| `FEATURES.md` *(private helper repo)* | What the package does, feature by feature. |
| [`report-front.md`](report-front.md) | Executive summary + PDF metadata. Not standalone — it is the first section of `WINMOL-report.pdf`. |

## Specifications

Written **before** their experiments, as pre-registrations. They are not updated to match
outcomes; deviations are recorded in the corresponding results document.

They live in the **private helper repo** under `specs/` — export, augmentation,
single-stage training, wandb, CPU inference, preprocessing comparison,
`2026-08-13-scale-augmentation-design.md`, `2026-08-16-tegel-r12-r13-design.md`.

The one specification kept in this repo is
[`2026-07-02-segmentor-pytorch-onnx-design.md`](2026-07-02-segmentor-pytorch-onnx-design.md),
because it specifies the interface `winmol_unet/contract.py` enforces.

## The four things most likely to mislead a newcomer

1. **Scale dominates.** Effective GSD is `tile_size / 512`, a user knob, not the
   orthomosaic's resolution. Matching it was worth **+14.6 F1**.
2. **Bachsee_north is not a bad site.** Its labels are correctly registered and its stems
   are the second-most separable in the corpus. It fails as a holdout because it is the
   only late-autumn acquisition — strong hue augmentation fixes it.
3. **F1 across different footprints is not one exam.** Comparing a native-resolution model
   with a fixed-metre one needs world-space scoring on a common grid
   (`scripts/plot_inference.py`), never their own tile sets.
4. **The noise floor is real.** Two runs with bit-identical gradients landed 2.0 F1 apart
   without `--deterministic`, 0.45–0.96 with it. Single-run gaps below that are unreadable.

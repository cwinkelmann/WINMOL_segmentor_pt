# How this U-Net differs from the original Ronneberger U-Net (PyTorch port)

The comparison table below is transferred from the R segmentor's
`WINMOL_segmentor/docs/unet_vs_ronneberger.md` (Ronneberger vs the R `model_UNet.R`),
with a third column added for **this PyTorch implementation** (`winmol_unet/model.py`).

> Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical
> Image Segmentation*, MICCAI. arXiv:1505.04597.

All three share the classic 5-level contracting/expansive encoder–decoder with skip
connections and the `64 → 128 → 256 → 512 → 1024` filter progression. This PyTorch port
is a *modernized adaptation of the R model, not a layer-exact clone* — see
"PyTorch port specifics" below.

## Summary table

| Aspect | Ronneberger 2015 (original) | R (`model_UNet.R`) | **This repo (`winmol_unet/model.py`)** |
|---|---|---|---|
| Input | 572×572×1 (grayscale) | 256×256×3 (RGB) | **512×512×3** (RGB; `contract.IMG_SIZE` / `IN_CHANNELS`) |
| Convolution padding | **`valid`** (unpadded) — maps shrink | **`same`** — size preserved | **`same`** (3×3, `padding=1`) |
| Output size vs input | Smaller (388 for 572) | Identical (256) | **Identical (512×512)** |
| Skip connections | encoder maps **cropped** before concat | direct concatenate | **direct concatenate** (`torch.cat([enc, up], dim=1)`) |
| Batch normalization | **None** | after every conv **and** every up-conv (22 BN) | **after every conv, but NOT after up-conv (18 BN)** |
| Normalization order | Conv → ReLU | Conv → ReLU → BN | **Conv → BN → ReLU** (modern default) |
| Conv bias | Yes | **`use_bias = FALSE`** | **`bias=False`** (BN β subsumes it) |
| Dropout | only at end of contracting path | one per block (0.1), between the two convs | **one per block (`Dropout2d`, 0.1), between the two convs** |
| Downsampling | 2×2 max-pool | 2×2 max-pool | **2×2 max-pool** |
| Upsampling | 2×2 up-conv halving channels | `conv2d_transpose` 2×2 s2 **+ BN** | **`ConvTranspose2d` 2×2 s2, `bias=False`, no BN after** |
| Depth (levels) | 5 (4 down / 4 up) | 5 | **5** |
| Activation | ReLU | ReLU | **ReLU** |
| Weight init | He / Gaussian √(2/N) | `he_normal` | **PyTorch default (Kaiming/He uniform)** |
| Output layer | 1×1 conv → 2ch, **softmax** | 1×1 conv → 1ch, **sigmoid** | **1×1 conv → 1ch; `forward` returns logits, sigmoid appended at export** |
| Task framing | multi-class (per-pixel softmax) | **binary** (single sigmoid) | **binary** (`OUT_CHANNELS = 1`) |
| Loss | weighted pixel-wise cross-entropy | `BCE + (1 − F1)` (hard-rounded F1) | **`BCEWithLogits + (1 − soft_F1)`** (differentiable F1) |
| Parameters | ~31M | ~31M | **31,036,673 (~31M)** |

## PyTorch port specifics (differences from the R baseline)

The R `model_UNet.R` is the reference the port adapts; these are the deliberate deltas
(also documented in `winmol_unet/model.py`):

1. **Normalization order — Conv→BN→ReLU vs R's Conv→ReLU→BN.** R fuses ReLU into
   `layer_conv_2d` then applies `batch_normalization`; the port uses the modern
   Conv→BN→ReLU. A genuinely different operation; both are common.
2. **No BatchNorm after ConvTranspose.** R inserts a BN after each of the 4
   up-convolutions (before the skip concat); the port omits them — **18 BN vs R's 22**.
3. **E7 decoder block uses 256 filters**, correcting R's `filters = 265` transposed-digit
   typo (`model_UNet.R:87,91`). The R baseline deliberately keeps the typo for faithful
   comparison; the port takes the intended 256.
4. **Input 512×512, not 256.** Matches the deployed production models, the analyzer config
   (`config.img_*` = 512), and the frozen ONNX contract — the port cannot use 256 without
   breaking the bridge.
5. **`forward()` returns logits; sigmoid is appended only at export.** Keeps training
   numerically stable with `BCEWithLogitsLoss` while satisfying the "sigmoid baked into the
   exported graph" contract (see `winmol_unet/export.py`).
6. **Soft (differentiable) F1 in the loss.** R's F1 term is `k_round`ed
   (non-differentiable → the loss effectively trains on BCE alone). The port uses an
   un-thresholded soft F1 so the term genuinely contributes gradient — a documented
   improvement (design spec §8). Reported metrics still use the hard-rounded 0.5-threshold
   F1, matching R.

## Unchanged from the original (all three implementations)

- The 5-level contracting/expansive U-shape and 4 down / 4 up resolution steps.
- The `64 → 128 → 256 → 512 → 1024` filter doubling per level.
- Skip connections from each encoder level to the matching decoder level.
- 3×3 convs, 2×2 max-pooling, ReLU.

## The Keras mirror

`winmol_unet/keras_model.py` reproduces **this PyTorch architecture** (not the R one)
layer-for-layer, so PyTorch ↔ HDF5/ONNX exports are numerically equivalent (parity ≤ 1e-4).
It is therefore *not* interchangeable with the R `.hdf5` models, which carry the R topology
(22 BN, Conv→ReLU→BN, 265-filter E7).

## Empirical parity — is this port as accurate as the R model?

The architectural table above says *what* differs; this section answers whether those
differences cost or gain accuracy. Both the R U-Net and the PyTorch U-Net were run through the
**identical two-stage pipeline** — GenDS10 (beech) stage-1 pre-train → SpecDS stage-2 fine-tune →
held-out **TestDS** evaluation — the canonical `main_training.R` protocol (80/20 split per stage,
early-stop on val_loss, Adam 1e-3, loss `BCE+(1−F1)`, `ReduceLROnPlateau`, batch 4, 100
epochs/stage cap). The R U-Net trains at 256×256 (its native size); the PyTorch models at 512×512
(the ONNX contract). Both train **from scratch** (no ImageNet). PyTorch numbers are ONNX-served
(CPU EP, exact fp32); R uses Keras `evaluate` on the same TestDS. Source:
`results/r_vs_pytorch_lrfix/`.

### Held-out TestDS — the primary comparison

| Model | Params | Precision | Recall | **F1 (Dice)** | IoU† | Train (min) |
|-------|-------:|----------:|-------:|--------------:|-----:|------------:|
| **R U-Net (Keras, 256)** | 31.0M | 0.7619 | 0.7259 | **0.7388** | 0.586 | — |
| U-Net (PyTorch, 512) | 31.0M | 0.7587 | 0.7621 | **0.7604** | 0.613 | 151 |
| DeepLabV3+ (PyTorch, 512) | 22.4M | 0.7593 | 0.7263 | **0.7424** | 0.590 | 82 |
| HRNet (PyTorch, 512) | 16.1M | 0.7606 | 0.7678 | **0.7642** | 0.618 | 106 |

†IoU here is the **stem-class** IoU. It is not logged during training, but for binary masks at a
fixed 0.5 threshold it is an exact algebraic function of F1/Dice: `IoU = F1 / (2 − F1)` (both come
from the same TP/FP/FN). This was **verified by direct measurement** — re-running the 256×256
PyTorch U-Net over TestDS gives measured stem-IoU 0.5956, matching `F1/(2−F1)` to four decimals (and
reproducing the reported P/R/F1 exactly). So the column is measured-equivalent, not a guess; it adds
no ranking information beyond F1. The *multi-class* mIoU (mean of stem + background IoU) is **0.786**
for the PyTorch U-Net at 256 — but background IoU (0.977) dominates it and it barely separates
models, which is why stem-IoU/F1 is the figure everyone reports for this task.

The table above reflects each model **as deployed** — R ships at 256×256, this port at 512×512 (the
ONNX contract) — so it mixes an architecture difference with a resolution difference. For a verdict
on the *port itself* the next table controls for resolution.

### The strictly fair comparison — same architecture family, same 256×256

The *same* PyTorch U-Net was also trained end-to-end at **256×256** through the identical two-stage
pipeline (172 min full run), so R-256 vs PyTorch-256 isolates framework + the architectural/loss
deltas with resolution held fixed. Source: `results/r_vs_pytorch_lrfix/ablation_results.json`.

| U-Net variant | Resolution | Precision | Recall | **F1** | IoU† |
|---------------|-----------:|----------:|-------:|-------:|-----:|
| **R U-Net (Keras)** | 256×256 | 0.7619 | 0.7259 | **0.7388** | 0.586 |
| **PyTorch U-Net** | 256×256 | 0.7265 | 0.7677 | **0.7465** | 0.596 |
| PyTorch U-Net (deployed) | 512×512 | 0.7587 | 0.7621 | **0.7604** | 0.613 |

**Verdict: at matched 256×256 the two are effectively equivalent — the port is a faithful
re-implementation, not a regression, and marginally ahead (+0.008 F1).** The P/R balance shifts
rather than the overall quality: R-256 is higher-precision / lower-recall (0.762 / 0.726), PyTorch-256
is lower-precision / higher-recall (0.727 / 0.768) — a threshold/operating-point difference, not a
capability gap. Going to the deployed **512×512 adds ~0.014 F1** (0.7465→0.7604), so most of the
headline 0.022 gap over R is the resolution the port actually runs at, and the remaining ~0.008 is
the framework/architecture delta. DeepLabV3+ (0.7424) and HRNet (0.7642 at half the params) bracket R
at 512. Net: the modernization deltas (Conv→BN→ReLU, 18-vs-22 BN, soft-F1 loss) are accuracy-neutral
at matched resolution, and the port is modestly better where it actually runs.

### Caveats

- **TestDS is the beech/Zenodo held-out set**, so this measures generalization to that
  distribution, not in-distribution spruce. It is the same set for every model, so the comparison
  is fair.
- An **earlier run** (`results/r_vs_pytorch/`) put R far lower (F1 0.6463). That run under-trained
  the R model — the stage-1→stage-2 learning rate was not carried across the two-stage handoff, so R
  never converged in stage 2. The LR-match fix (`results/r_vs_pytorch_lrfix/`, the table above)
  brings R to full strength and is the fair baseline; the older number is a pipeline artifact, not
  the R model's true ceiling.
- GenDS10→SpecDS leakage was not exhaustively excluded (the R repo's `runs/*/leakage_*` checks apply
  equally to both sides), so treat absolute F1 as approximate; the *relative* R-vs-PyTorch
  comparison is unaffected.

## Inference speed (baseline, before the CPU-speedup work)

This section reports the **unoptimized** deployable model's inference cost — the plain fp32 U-Net
served through ONNX Runtime, *without* the width-scaling / int8-quantization work that lives on the
`feat/cpu-inference-speedup` branch. It is the "what you get out of the box" figure. Setup: median
of 30 timed `session.run` (5 warmup), batch 1 (the CPU-bound single-tile case), the deployable
two-stage lrfix U-Net (F1 0.760). Source: `results/cpu_speedup/`.

| Hardware | Runtime | Precision | Latency / 512² tile | Throughput |
|----------|---------|-----------|--------------------:|-----------:|
| CPU — i7-1255U, 4 threads | ONNX Runtime CPU EP | fp32 | **~2521 ms** | ~0.4 tile/s |
| CPU — i7-1255U, 1 thread | ONNX Runtime CPU EP | fp32 | ~3929 ms | ~0.25 tile/s |
| GPU — RTX 4080 SUPER | PyTorch eager CUDA | fp32 | **~13.8 ms** | ~73 tile/s |
| GPU — RTX 4080 SUPER | ONNX Runtime CUDA EP | fp32 | ~14.9 ms | ~62 tile/s (b1) |

The headline is the **CPU cost: ~2.5 s per 512×512 tile** (single-threaded, ~3.9 s). Because the
analyzer sweeps a full orthomosaic as a grid of overlapping tiles, this dominates wall-clock on
CPU-only deployments — a few thousand tiles is tens of minutes. On GPU the same model is ~180×
faster (single-digit-to-teens ms), so inference is not the bottleneck there. fp32 CPU latency
barely scales with threads (1.9× from 1→8 threads — it is memory-bandwidth bound), so throwing cores
at it does not rescue the CPU case.

> **Not covered here:** the `feat/cpu-inference-speedup` branch removes this CPU bottleneck — static
> int8 quantization (≈3×, lossless), width scaling (≈4× at full accuracy), and the two combined
> (**width-0.5 + int8 = ~10× faster / 249 ms / F1 0.760 / 8 MB**, lossless; width-0.25 + int8 = ~29×
> for −0.006 F1). Those numbers and the GPU fp16 analogue live in
> [`docs/2026-07-21-cpu-inference-speedup-results.md`](2026-07-21-cpu-inference-speedup-results.md)
> on that branch (PR #12) and are intentionally out of scope for this baseline chapter.

> **On sources.** The `results/…` paths cited in this document are local raw-run
> artifacts: `results/` is gitignored, so these numbers cannot be re-derived from the
> repository alone. They are recorded here as the durable form of those runs.

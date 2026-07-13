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

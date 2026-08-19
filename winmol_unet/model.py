"""PyTorch U-Net adapted from the R model_UNet.R (512x512, BN+dropout, skip concats).

forward() returns LOGITS. Sigmoid is appended only at export time
(see winmol_unet.export / export_keras), satisfying the "sigmoid in graph"
contract while keeping training numerically stable with BCEWithLogitsLoss.

Deviations from the original R model_UNet.R (same skeleton: 19 Conv, 4 ConvTranspose,
64->128->256->512->1024 ladder, dropout 0.1, ~31M params). This is a modernized
adaptation, NOT a layer-exact port:

  1. Normalization order: this port uses Conv -> BN -> ReLU (per conv). The R model
     uses Conv -> ReLU -> BN (ReLU fused into layer_conv_2d, then batch_normalization).
     Both are common; Conv->BN->ReLU is the modern default.
  2. No BatchNorm after ConvTranspose. The R model inserts a BN after each of the 4
     up-convolutions (before the skip concat); this port omits them (18 BN vs R's 22).
  3. Decoder block E7 uses 256 filters; the R code has filters=265, a probable typo.
  4. Input is 512x512 (R script used 256) to match the deployed models + analyzer config.

The Keras mirror in winmol_unet.keras_model reproduces THIS architecture (not the R
one) exactly, so the PyTorch<->HDF5/ONNX exports are numerically equivalent.
"""
import torch
import torch.nn as nn

from .contract import IN_CHANNELS, OUT_CHANNELS


def _conv_block(in_ch, out_ch, order="bn_relu"):
    """One conv unit in either normalisation order.

    ``bn_relu``  Conv -> BN -> ReLU, the modern default and this port's original.
    ``relu_bn``  Conv -> ReLU -> BN, what R does: ``layer_conv_2d(activation='relu')``
                 followed by ``layer_batch_normalization()``.

    Both keep ``bias=False`` — R sets ``use_bias = FALSE`` in either case, and BN's beta
    subsumes the bias regardless of which side of the activation it sits on.
    """
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
    if order == "relu_bn":
        return nn.Sequential(conv, nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch))
    if order == "bn_relu":
        return nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    raise ValueError(f"block_order must be 'bn_relu' or 'relu_bn', got {order!r}")


# kept for callers that imported it before block_order existed
def _conv_bn_relu(in_ch, out_ch):
    return _conv_block(in_ch, out_ch, "bn_relu")


class _DoubleConv(nn.Module):
    """conv-norm-act -> dropout -> conv-norm-act (matches R block layout)."""

    def __init__(self, in_ch, out_ch, dropout, order="bn_relu"):
        super().__init__()
        self.c1 = _conv_block(in_ch, out_ch, order)
        self.drop = nn.Dropout2d(p=dropout)
        self.c2 = _conv_block(out_ch, out_ch, order)

    def forward(self, x):
        return self.c2(self.drop(self.c1(x)))


def _scale_width(base, width_mult):
    # Scale a base channel count, snapping to a multiple of 8 (hardware-friendly,
    # keeps int8/AVX kernels happy). width_mult=1.0 reproduces the base ladder exactly.
    return max(8, int(round(base * width_mult / 8.0)) * 8)


class UNet(nn.Module):
    """U-Net with an optional ``width_mult`` that scales every channel width.

    ``width_mult=1.0`` (default) is byte-identical to the original 64->1024 ladder.
    Smaller multipliers (e.g. 0.5) cut channel widths for a ~quadratic drop in conv
    FLOPs/params -- the primary lever for CPU-inference speedup -- while leaving the
    forward topology and the ONNX contract (512x512x3 -> 512x512x1) unchanged.
    """

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, dropout=0.1,
                 width_mult=1.0, block_order="bn_relu"):
        super().__init__()
        w1, w2, w3, w4, w5 = (_scale_width(b, width_mult)
                              for b in (64, 128, 256, 512, 1024))
        self.enc1 = _DoubleConv(in_channels, w1, dropout, block_order)
        self.enc2 = _DoubleConv(w1, w2, dropout, block_order)
        self.enc3 = _DoubleConv(w2, w3, dropout, block_order)
        self.enc4 = _DoubleConv(w3, w4, dropout, block_order)
        self.bottleneck = _DoubleConv(w4, w5, dropout, block_order)
        self.pool = nn.MaxPool2d(2)

        # Decoder input channels come from the actual forward-time concat
        # (skip + upsample output, each w_k -> 2*w_k), NOT the next ladder width:
        # with independent rounding in _scale_width, w5 == 2*w4 etc. only holds
        # for multipliers like 1.0/0.5/0.25. At those points 2*w_k equals the
        # historical ladder value, so width_mult=1.0 stays byte-identical.
        self.up4 = nn.ConvTranspose2d(w5, w4, kernel_size=2, stride=2, bias=False)
        self.dec4 = _DoubleConv(2 * w4, w4, dropout)
        self.up3 = nn.ConvTranspose2d(w4, w3, kernel_size=2, stride=2, bias=False)
        self.dec3 = _DoubleConv(2 * w3, w3, dropout)
        self.up2 = nn.ConvTranspose2d(w3, w2, kernel_size=2, stride=2, bias=False)
        self.dec2 = _DoubleConv(2 * w2, w2, dropout)
        self.up1 = nn.ConvTranspose2d(w2, w1, kernel_size=2, stride=2, bias=False)
        self.dec1 = _DoubleConv(2 * w1, w1, dropout)

        self.head = nn.Conv2d(w1, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))

        d4 = self.dec4(torch.cat([c4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([c3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([c2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([c1, self.up1(d2)], dim=1))
        return self.head(d1)  # logits

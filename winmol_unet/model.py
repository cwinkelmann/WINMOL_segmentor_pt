"""PyTorch U-Net. Mirrors the R model_UNet.R (512x512, BN+dropout, skip concats).

forward() returns LOGITS. Sigmoid is appended only at ONNX export time
(see winmol_unet.export), satisfying the "sigmoid in graph" contract while
keeping training numerically stable with BCEWithLogitsLoss.
"""
import torch
import torch.nn as nn

from .contract import IN_CHANNELS, OUT_CHANNELS


def _conv_bn_relu(in_ch, out_ch):
    # use_bias=False because BatchNorm follows (matches R: use_bias=FALSE)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    """conv-bn-relu -> dropout -> conv-bn-relu (matches R block layout)."""

    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.c1 = _conv_bn_relu(in_ch, out_ch)
        self.drop = nn.Dropout2d(p=dropout)
        self.c2 = _conv_bn_relu(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.drop(self.c1(x)))


class UNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, dropout=0.1):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, 64, dropout)
        self.enc2 = _DoubleConv(64, 128, dropout)
        self.enc3 = _DoubleConv(128, 256, dropout)
        self.enc4 = _DoubleConv(256, 512, dropout)
        self.bottleneck = _DoubleConv(512, 1024, dropout)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.dec4 = _DoubleConv(1024, 512, dropout)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.dec3 = _DoubleConv(512, 256, dropout)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.dec2 = _DoubleConv(256, 128, dropout)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dec1 = _DoubleConv(128, 64, dropout)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

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

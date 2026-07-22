"""FieldNet — compact multi-head UNet that predicts the Approach-A supervision fields.

Input:  the UNet stem **mask** (1 ch), optionally concatenated with RGB (4 ch).
Output: the three fields the decoder consumes (see fields.py / decode.py):
  heat   (N,1,H,W)  centerline ridge, as **logits** (use BCE/focal-with-logits; sigmoid to view)
  orient (N,2,H,W)  tangent (sin 2t, cos 2t), L2-normalised so it is a valid unit vector
  diam   (N,1,H,W)  stem diameter, softplus -> non-negative

Deliberately small: this task (mask -> centerline graph) is far easier than segmentation, and the
study's ceiling is the heuristic labels, not capacity. Self-contained (no winmol_unet import) so
the mini-project stays independent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class FieldNet(nn.Module):
    def __init__(self, in_channels=1, base=32, depth=3):
        super().__init__()
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth + 1)]      # e.g. 32,64,128,256

        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chans[:-1]:
            self.enc.append(_block(cin, c))
            cin = c
        self.bottleneck = _block(cin, chans[-1])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(chans[i], chans[i - 1], 2, stride=2))
            self.dec.append(_block(chans[i - 1] * 2, chans[i - 1]))

        self.pool = nn.MaxPool2d(2)
        self.head_heat = nn.Conv2d(base, 1, 1)
        self.head_orient = nn.Conv2d(base, 2, 1)
        self.head_diam = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))

        orient = self.head_orient(x)
        # normalise to a unit (sin2t, cos2t); eps keeps the gradient finite at the origin
        orient = orient / orient.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "heat": self.head_heat(x),                       # logits
            "orient": orient,
            "diam": F.softplus(self.head_diam(x)),           # >= 0
        }

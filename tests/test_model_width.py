import torch
from winmol_unet.model import UNet


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_width_mult_default_is_unchanged():
    # width_mult=1.0 must be byte-identical to the historical UNet: same keys+shapes.
    a = UNet()
    b = UNet(width_mult=1.0)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa.keys()) == list(sb.keys())
    assert all(sa[k].shape == sb[k].shape for k in sa)
    assert _nparams(b) == _nparams(a)


def test_width_half_builds_and_keeps_contract_shape():
    m = UNet(width_mult=0.5).eval()
    x = torch.zeros(2, 3, 512, 512)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1, 512, 512)


def test_width_half_has_roughly_quarter_params():
    full = _nparams(UNet(width_mult=1.0))
    half = _nparams(UNet(width_mult=0.5))
    # halving every channel width scales conv params ~ 1/4 (quadratic in width).
    assert 0.2 * full < half < 0.32 * full


def test_width_channels_are_multiples_of_eight():
    # first encoder conv out-channels must stay hardware-friendly (multiple of 8)
    m = UNet(width_mult=0.5)
    first_conv = m.enc1.c1[0]  # Conv2d
    assert first_conv.out_channels % 8 == 0
    assert first_conv.out_channels == 32  # round(0.5*64/8)*8

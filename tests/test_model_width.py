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


def test_arbitrary_width_mult_builds_and_forwards():
    # Non-power-of-two multipliers: independently rounded widths break the
    # coincidental w5 == 2*w4 (etc.) identities, so the decoder must declare
    # its input channels from the actual skip+upsample concat, not the ladder.
    for m in (0.3, 0.6, 0.75):
        net = UNet(width_mult=m).eval()
        with torch.no_grad():
            y = net(torch.randn(1, 3, 64, 64))
        assert y.shape == (1, 1, 64, 64), f"width_mult={m}"


def test_block_order_switches_conv_norm_activation():
    """`relu_bn` reproduces R's Conv -> ReLU -> BN; `bn_relu` is the modern default.

    Same parameter count either way, so a comparison isolates the ordering rather than
    capacity. R fuses ReLU into layer_conv_2d then applies batch_normalization.
    """
    import torch

    from winmol_unet.model import UNet

    a = UNet(block_order="bn_relu")
    b = UNet(block_order="relu_bn")
    assert [type(l).__name__ for l in a.enc1.c1] == ["Conv2d", "BatchNorm2d", "ReLU"]
    assert [type(l).__name__ for l in b.enc1.c1] == ["Conv2d", "ReLU", "BatchNorm2d"]
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    with torch.no_grad():
        assert b.eval()(torch.zeros(1, 3, 512, 512)).shape == (1, 1, 512, 512)


def test_unknown_block_order_is_rejected():
    import pytest

    from winmol_unet.model import UNet

    with pytest.raises(ValueError, match="block_order"):
        UNet(block_order="relu_only")

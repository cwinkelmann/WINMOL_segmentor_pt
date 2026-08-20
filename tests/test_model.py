"""The UNet definition and the architecture factory: output shape, channel width, block
order, weight init, and contract-shaped logits from every supported arch.

Merges tests/test_model_width.py, tests/test_init_weights.py, tests/test_model_factory.py,
and the build_model width tests from tests/test_width_wiring.py (whose config_from_args
CLI tests stay behind for Task 8).
"""
import pytest
import torch

from winmol_unet.model import UNet
from winmol_unet.training.model_factory import build_model
from winmol_unet.training.run_train import config_from_args, load_init_weights


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


# --- UNet: output shape & logits -------------------------------------------------


def test_unet_output_shape():
    model = UNet().eval()
    x = torch.zeros(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 512, 512)


def test_unet_returns_logits_not_probs():
    # logits are unbounded; a constant-0 input should not be forced into [0,1]
    model = UNet().eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert not any(m.__class__.__name__ == "Sigmoid" for m in model.modules())


# --- UNet: channel width -----------------------------------------------------------


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


# --- UNet: block order (Conv-BN-ReLU vs R's Conv-ReLU-BN) --------------------------


def test_block_order_switches_conv_norm_activation():
    """`relu_bn` reproduces R's Conv -> ReLU -> BN; `bn_relu` is the modern default.

    Same parameter count either way, so a comparison isolates the ordering rather than
    capacity. R fuses ReLU into layer_conv_2d then applies batch_normalization.
    """
    a = UNet(block_order="bn_relu")
    b = UNet(block_order="relu_bn")
    assert [type(l).__name__ for l in a.enc1.c1] == ["Conv2d", "BatchNorm2d", "ReLU"]
    assert [type(l).__name__ for l in b.enc1.c1] == ["Conv2d", "ReLU", "BatchNorm2d"]
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    with torch.no_grad():
        assert b.eval()(torch.zeros(1, 3, 512, 512)).shape == (1, 1, 512, 512)


def test_unknown_block_order_is_rejected():
    with pytest.raises(ValueError, match="block_order"):
        UNet(block_order="relu_only")


# --- Fine-tuning init weights: load or fail loudly ----------------------------------


def test_loads_matching_weights(tmp_path):
    src, dst = UNet(), UNet()
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.25)
    p = tmp_path / "w.pt"
    torch.save(src.state_dict(), p)
    load_init_weights(dst, str(p))
    assert all(float(q.detach().mean()) == pytest.approx(0.25) for q in dst.parameters())


def test_mismatched_architecture_fails_loudly(tmp_path):
    p = tmp_path / "w.pt"
    torch.save(UNet(width_mult=0.5).state_dict(), p)
    with pytest.raises(SystemExit, match="does not match this architecture"):
        load_init_weights(UNet(), str(p))


def test_flag_reaches_the_config(tmp_path):
    cfg = config_from_args(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path),
                            "--init-weights", "/x/model.pt"])
    assert cfg.init_weights == "/x/model.pt"
    assert config_from_args(["--data-dir", str(tmp_path),
                             "--out-dir", str(tmp_path)]).init_weights is None


# --- Architecture factory: every supported arch returns contract-shaped logits -----


def test_unet_returns_contract_shaped_logits():
    y = build_model("unet").eval()(torch.zeros(1, 3, 512, 512))
    assert y.shape == (1, 1, 512, 512)


@pytest.mark.parametrize("arch", ["deeplabv3plus", "hrnet", "segformer", "dpt"])
def test_smp_arch_returns_contract_shaped_logits(arch):
    # encoder_weights=None -> no ImageNet download (hermetic)
    y = build_model(arch, encoder_weights=None).eval()(torch.zeros(1, 3, 512, 512))
    assert y.shape == (1, 1, 512, 512)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model("frobnicate")


# --- Architecture factory: width wiring ---------------------------------------------


def test_build_model_unet_accepts_width_mult():
    full = build_model("unet")
    half = build_model("unet", width_mult=0.5)
    assert _nparams(half) < 0.32 * _nparams(full)


def test_build_model_default_width_is_full():
    assert _nparams(build_model("unet", width_mult=1.0)) == _nparams(build_model("unet"))

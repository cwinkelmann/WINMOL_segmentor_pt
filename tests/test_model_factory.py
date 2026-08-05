import pytest
import torch

from training.model_factory import build_model


def test_unet_returns_contract_shaped_logits():
    y = build_model("unet").eval()(torch.zeros(1, 3, 512, 512))
    assert y.shape == (1, 1, 512, 512)


@pytest.mark.parametrize("arch", ["deeplabv3plus", "hrnet"])
def test_smp_arch_returns_contract_shaped_logits(arch):
    # encoder_weights=None -> no ImageNet download (hermetic)
    y = build_model(arch, encoder_weights=None).eval()(torch.zeros(1, 3, 512, 512))
    assert y.shape == (1, 1, 512, 512)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model("frobnicate")


@pytest.mark.parametrize("arch", ["unet", "deeplabv3plus", "hrnet"])
def test_build_model_rgbd_forward(arch):
    model = build_model(arch=arch, encoder_weights=None, in_channels=4).eval()
    x = torch.randn(1, 4, 64, 64)   # 64: divisible by 32 (smp) and by 16 (UNet pools)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)

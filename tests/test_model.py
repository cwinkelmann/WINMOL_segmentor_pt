import torch
from winmol_unet.model import UNet


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

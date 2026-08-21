

def test_loss_components_sum_to_the_composite():
    """The parts must equal the whole, or the component charts lie."""
    import torch
    from winmol_unet.training.losses import LOSSES, LOSS_COMPONENTS

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    for name, comp_fn in LOSS_COMPONENTS.items():
        total = LOSSES[name](logits, target)
        parts = comp_fn(logits, target)
        assert len(parts) >= 2, name
        s = sum(v for v in parts.values())
        assert torch.allclose(total, s, atol=1e-6), (name, float(total), float(s))


def test_plain_losses_have_no_components():
    from winmol_unet.training.losses import LOSS_COMPONENTS
    assert "bce" not in LOSS_COMPONENTS
    assert "focal" not in LOSS_COMPONENTS



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


def test_run_logger_log_figure_tensorboard_only(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from winmol_unet.training.run_logger import RunLogger
    lg = RunLogger(str(tmp_path), use_wandb=False)
    fig = plt.figure()
    lg.log_figure("val/examples", fig, 0)
    lg.close()
    assert any(f.startswith("events") for f in __import__("os").listdir(tmp_path))


def test_evaluate_returns_val_loss_components():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from winmol_unet.training.evaluate import evaluate

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))   # evaluate() reads a param's device

        def forward(self, x):
            return x[:, :1] * 0.0 + self.bias

    ds = TensorDataset(torch.rand(4, 3, 16, 16), (torch.rand(4, 1, 16, 16) > 0.5).float())
    out = evaluate(Tiny(), DataLoader(ds, batch_size=2))
    assert "loss_bce" in out and "loss_soft_f1_term" in out
    assert abs(out["loss"] - (out["loss_bce"] + out["loss_soft_f1_term"])) < 1e-5

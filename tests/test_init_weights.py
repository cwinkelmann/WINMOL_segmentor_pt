"""Fine-tuning must start from the weights it claims to, or fail loudly.

A partial load (wrong architecture, wrong width) would train something that is neither the
pretrained model nor a clean baseline, and would still report a plausible F1.
"""
import pytest
import torch

from winmol_unet.training.run_train import config_from_args, load_init_weights
from winmol_unet.model import UNet


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

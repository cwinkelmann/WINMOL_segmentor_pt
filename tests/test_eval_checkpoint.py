"""eval_checkpoint must reproduce run_train's own test number for the same model+set.

The whole point of the script is that a 2x2's off-diagonal is comparable to the diagonal
that training printed. If it scored differently -- a different resize, a different
threshold, augmentation left on -- the off-diagonal would be measuring the harness rather
than the filter mismatch.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from winmol_unet.training.score_checkpoint import score
from winmol_unet.training.dataset import StemDataset
from winmol_unet.training.evaluate import evaluate
from winmol_unet.training.model_factory import build_model


def _ds(d, n=6, seed=0):
    rng = np.random.default_rng(seed)
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "mask").mkdir(parents=True, exist_ok=True)
    for k in range(1, n + 1):
        rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(rgb, "RGB").save(d / "train" / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8)
        m[:, : 8 + k] = 255
        Image.fromarray(m, "L").save(d / "mask" / f"mask{k}.gif")


def test_matches_the_training_loops_own_evaluate(tmp_path):
    data = tmp_path / "test_ds"
    _ds(data)

    model = build_model("deeplabv3plus", encoder_weights=None).eval()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    # what run_train._run_test does: no augmentation, plain resize dataset, same evaluate()
    ds = StemDataset(str(data / "train"), str(data / "mask"), 32, transform=None,
                     cache=False)
    expected = evaluate(model, DataLoader(ds, batch_size=2))

    got, n = score(str(ckpt), "deeplabv3plus", str(data), batch_size=2, device="cpu",
                   img_size=32, num_workers=0)

    assert n == 6
    for k in ("f1", "precision", "recall", "loss"):
        assert abs(got[k] - expected[k]) < 1e-6, f"{k}: {got[k]} != {expected[k]}"

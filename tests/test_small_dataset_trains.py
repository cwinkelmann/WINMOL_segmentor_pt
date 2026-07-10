"""Regression: drop_last must not zero out training when the train split is smaller
than batch_size (else run_training silently exports a random-init model)."""
import glob
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.config import TrainConfig
from training.run_train import run_training


def test_training_runs_when_train_split_smaller_than_batch(tmp_path):
    img_dir, mask_dir = tmp_path / "train", tmp_path / "mask"
    img_dir.mkdir(); mask_dir.mkdir()
    for k in range(1, 5):                       # 4 pairs -> ~3 train, batch_size 4
        rgb = np.zeros((32, 32, 3), np.uint8); rgb[:, :16, :] = 255
        Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
        m = np.zeros((32, 32), np.uint8); m[:, :16] = 255
        Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
    out = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ck"),
        log_dir=str(tmp_path / "log"), hdf5_out=str(out / "m.hdf5"),
        onnx_out=str(out / "m.onnx"), epochs=1, batch_size=4, patience=999, device="cpu",
    )
    run_training(cfg)
    ea = EventAccumulator(sorted(glob.glob(str(tmp_path / "log" / "events*")))[-1]); ea.Reload()
    train_loss = ea.Scalars("train/loss")[0].value
    assert train_loss > 0.0     # >0 means the train loop actually ran (not zeroed by drop_last)

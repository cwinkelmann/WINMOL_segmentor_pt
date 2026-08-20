"""Shared fixtures for the WINMOL test suite.

Replaces the synthetic-dataset builder that was duplicated across 22 test modules
and the TrainConfig block duplicated across 15. Both are factories rather than
plain fixtures because call sites need several datasets (train + val + test) or
several configs (stage 1 + stage 2) inside one test.
"""
import os
import pathlib

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def stem_dataset():
    """Build a StemDataset-shaped folder: train/train{k}.jpeg + mask/mask{k}.gif.

    The image is a hard vertical split -- white where the mask is white -- so a
    model can actually fit it, which is what the overfit test needs.

    ``ids`` overrides the default contiguous ``1..n`` numbering when a caller
    needs specific (e.g. non-contiguous) integer ids to prove pairing logic;
    it defaults to None so every existing call site is unaffected.
    """
    def _make(root, n=6, size=32, stem_frac=0.5, ids=None):
        root = pathlib.Path(root)
        img_dir, mask_dir = root / "train", root / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        split = max(1, int(size * stem_frac))
        for k in (ids if ids is not None else range(1, n + 1)):
            rgb = np.zeros((size, size, 3), np.uint8)
            rgb[:, :split, :] = 255
            Image.fromarray(rgb, "RGB").save(img_dir / f"train{k}.jpeg")
            m = np.zeros((size, size), np.uint8)
            m[:, :split] = 255
            Image.fromarray(m, "L").save(mask_dir / f"mask{k}.gif")
        return root
    return _make


@pytest.fixture
def train_config(tmp_path):
    """A hermetic TrainConfig: one epoch, CPU, no encoder download."""
    def _make(data_dir=None, **overrides):
        from winmol_unet.training.config import TrainConfig   # lazy: pulls torch
        out = tmp_path / "out"
        base = dict(
            data_dir=str(data_dir if data_dir is not None else tmp_path),
            checkpoint_dir=str(tmp_path / "ck"),
            log_dir=str(tmp_path / "log"),
            hdf5_out=str(out / "m.hdf5"),
            onnx_out=str(out / "m.onnx"),
            pt_out=str(out / "m.pt"),
            keras_out=str(out / "m.keras"),
            epochs=1,
            batch_size=2,
            patience=999,
            device="cpu",
            encoder_weights=None,
        )
        base.update(overrides)
        return TrainConfig(**base)
    return _make


@pytest.fixture
def force_cpu_onnx(monkeypatch):
    """Pin the CPU EP so ONNX parity assertions are bit-exact.

    CoreML and CUDA compute in fp16 and drift past a tight epsilon.
    """
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    yield
    monkeypatch.delenv("WINMOL_ONNX_FORCE_CPU", raising=False)

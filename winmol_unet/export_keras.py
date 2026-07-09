"""Copy trained PyTorch UNet weights into the Keras mirror and save HDF5.

The exported file loads via keras.models.load_model(path, compile=False) — the
analyzer's existing load path — with no analyzer code change.

Conventions:
- Conv2d kernel: torch [out,in,kh,kw] -> keras [kh,kw,in,out]  (transpose 2,3,1,0)
- ConvTranspose2d kernel: torch [in,out,kh,kw] -> keras [kh,kw,out,in] (transpose 2,3,1,0)
- BatchNorm: torch (weight,bias,running_mean,running_var)
             -> keras [gamma,beta,moving_mean,moving_variance]
Weighted layers are collected in construction order from both models and zipped;
counts and per-layer shapes are asserted so a mismatch raises instead of
silently misaligning.
"""
import numpy as np
import torch.nn as nn
from tensorflow.keras import layers as klayers

from .keras_model import build_keras_unet


def _torch_weighted(model):
    return [m for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))]


def _keras_weighted(model):
    return [l for l in model.layers
            if isinstance(l, (klayers.Conv2D, klayers.Conv2DTranspose,
                              klayers.BatchNormalization))]


def _transfer(torch_model, keras_model):
    t_layers = _torch_weighted(torch_model)
    k_layers = _keras_weighted(keras_model)
    if len(t_layers) != len(k_layers):
        raise ValueError(
            f"weighted-layer count mismatch: torch {len(t_layers)} vs "
            f"keras {len(k_layers)} — architectures differ")

    for t, k in zip(t_layers, k_layers):
        if isinstance(t, nn.Conv2d):
            if not isinstance(k, klayers.Conv2D):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            w = t.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
            weights = [w]
            if t.bias is not None:
                weights.append(t.bias.detach().cpu().numpy())
        elif isinstance(t, nn.ConvTranspose2d):
            if not isinstance(k, klayers.Conv2DTranspose):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            w = t.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
            weights = [w]
            if t.bias is not None:
                weights.append(t.bias.detach().cpu().numpy())
        else:  # nn.BatchNorm2d
            if not isinstance(k, klayers.BatchNormalization):
                raise ValueError(f"type mismatch: {type(t)} vs {type(k)}")
            weights = [
                t.weight.detach().cpu().numpy(),
                t.bias.detach().cpu().numpy(),
                t.running_mean.detach().cpu().numpy(),
                t.running_var.detach().cpu().numpy(),
            ]

        expected = [wgt.shape for wgt in k.get_weights()]
        got = [wgt.shape for wgt in weights]
        if expected != got:
            raise ValueError(
                f"weight-shape mismatch for {k.name}: keras {expected} vs "
                f"transferred {got}")
        k.set_weights(weights)


def _build_and_transfer(torch_model, dropout=0.1):
    torch_model = torch_model.eval()
    keras_model = build_keras_unet(dropout=dropout)
    _transfer(torch_model, keras_model)
    return keras_model


def export_to_keras_hdf5(torch_model, path, dropout=0.1):
    """Legacy HDF5 (analyzer's current load path)."""
    _build_and_transfer(torch_model, dropout).save(path, save_format="h5")
    return path


def export_to_keras(torch_model, path, dropout=0.1):
    """Native Keras 3 format (path should end in .keras)."""
    _build_and_transfer(torch_model, dropout).save(path)
    return path

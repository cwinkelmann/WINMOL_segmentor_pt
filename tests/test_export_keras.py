# tests/test_export_keras.py
import numpy as np
import torch

from winmol_unet.model import UNet
from winmol_unet.export_keras import export_to_keras_hdf5

def _torch_probs(model, x_nhwc):
    with torch.no_grad():
        nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))
        return torch.sigmoid(model(nchw)).numpy().transpose(0, 2, 3, 1)

def test_export_parity_torch_vs_keras(tmp_path):
    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)

    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # unmodified-analyzer load path
    keras_out = loaded.predict(x, verbose=0)

    assert keras_out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), keras_out, rtol=0.0, atol=1e-4)

def test_transfer_fails_loudly_on_mismatch(tmp_path):
    # A model with a different architecture must not silently mis-map.
    import torch.nn as nn
    class Wrong(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 1, 1)
        def forward(self, x):
            return self.c(x)
    import pytest
    with pytest.raises((ValueError, AssertionError)):
        export_to_keras_hdf5(Wrong().eval(), str(tmp_path / "bad.hdf5"))

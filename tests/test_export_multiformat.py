import numpy as np
import torch

from winmol_unet.model import UNet
from winmol_unet.export import export_to_pt
from winmol_unet.export_keras import export_to_keras, export_to_keras_hdf5


def _torch_probs(model, x_nhwc):
    with torch.no_grad():
        nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))
        return torch.sigmoid(model(nchw)).numpy().transpose(0, 2, 3, 1)


def test_export_pt_roundtrip(tmp_path):
    m = UNet().eval()
    path = export_to_pt(m, str(tmp_path / "m.pt"))
    m2 = UNet()
    m2.load_state_dict(torch.load(path))
    m2.eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)


def test_export_keras_native_parity(tmp_path):
    model = UNet().eval()
    path = export_to_keras(model, str(tmp_path / "m.keras"))
    assert path.endswith(".keras")
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)   # native-format load
    out = loaded.predict(x, verbose=0)
    assert out.shape == (2, 512, 512, 1)
    assert np.allclose(_torch_probs(model, x), out, rtol=0.0, atol=1e-4)


def test_export_hdf5_still_works(tmp_path):
    model = UNet().eval()
    path = export_to_keras_hdf5(model, str(tmp_path / "m.hdf5"))
    from tensorflow import keras
    loaded = keras.models.load_model(path, compile=False)
    x = np.random.rand(1, 512, 512, 3).astype(np.float32)
    assert np.allclose(_torch_probs(model, x), loaded.predict(x, verbose=0), atol=1e-4)

import numpy as np
import pytest

# the Keras mirror is opt-in (--export-keras) and TensorFlow is not installed in
# CI; skip rather than fail collection where it is absent
pytest.importorskip("tensorflow")

from winmol_unet.keras_model import build_keras_unet


def test_keras_unet_output_shape_and_range():
    model = build_keras_unet()
    x = np.random.rand(2, 512, 512, 3).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 512, 512, 1)
    assert (y >= 0).all() and (y <= 1).all()   # sigmoid head


def test_keras_unet_uses_torch_bn_epsilon():
    from tensorflow.keras.layers import BatchNormalization
    model = build_keras_unet()
    bns = [l for l in model.layers if isinstance(l, BatchNormalization)]
    assert bns and all(abs(l.epsilon - 1e-5) < 1e-12 for l in bns)


def test_keras_unet_roundtrips_via_load_model(tmp_path):
    from tensorflow import keras
    path = str(tmp_path / "m.hdf5")
    build_keras_unet().save(path, save_format="h5")
    loaded = keras.models.load_model(path, compile=False)   # analyzer's load path
    assert loaded.predict(np.zeros((1, 512, 512, 3), np.float32), verbose=0).shape == (1, 512, 512, 1)

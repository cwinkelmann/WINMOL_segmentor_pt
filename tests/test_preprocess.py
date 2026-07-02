import numpy as np
from winmol_unet import preprocess


def test_to_float01_uint8():
    arr = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = preprocess.to_float01(arr)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_resize_batch_shape():
    batch = np.zeros((4, 100, 120, 3), dtype=np.float32)
    out = preprocess.resize_batch(batch, size=512)
    assert out.shape == (4, 512, 512, 3)
    assert out.dtype == np.float32

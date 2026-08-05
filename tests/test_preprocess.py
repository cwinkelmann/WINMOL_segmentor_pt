import numpy as np
from winmol_unet import preprocess
from winmol_unet.preprocess import normalize_depth


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


def test_normalize_depth_minmax_uint16():
    arr = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    out = normalize_depth(arr)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    np.testing.assert_allclose(out, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32))


def test_normalize_depth_constant_maps_to_zeros():
    out = normalize_depth(np.full((4, 4), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))


def test_normalize_depth_fixed_range_clips():
    arr = np.array([-5.0, 0.0, 5.0, 15.0], dtype=np.float32)
    out = normalize_depth(arr, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32))

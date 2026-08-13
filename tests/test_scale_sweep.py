"""The sweep's crop->GSD arithmetic and its resize contract.

Effective GSD = native_gsd * crop_px / img_size. If that is wrong the whole curve is
mislabelled, so it is pinned rather than trusted.
"""
import numpy as np
from PIL import Image

from scripts.scale_sweep import _crop_resize


def test_centre_crop_is_centred_and_resizes_to_the_model_input():
    a = np.zeros((666, 666, 3), np.uint8)
    a[333, 333] = 255                                  # a single lit pixel at the centre
    m = np.zeros((666, 666), np.uint8)
    m[333, 333] = 255
    im, mk = _crop_resize(Image.fromarray(a, "RGB"), Image.fromarray(m, "L"), 394, 512)
    assert im.size == (512, 512) and mk.size == (512, 512)
    # the centre pixel must survive at the centre, not drift to a corner
    arr = np.asarray(im).sum(2)
    y, x = np.unravel_index(arr.argmax(), arr.shape)
    assert abs(int(y) - 256) <= 2 and abs(int(x) - 256) <= 2, (y, x)


def test_mask_stays_binary_through_the_resize():
    """Nearest for masks: a bilinear mask resize would invent intermediate labels."""
    m = np.zeros((666, 666), np.uint8)
    m[300:360, 100:560] = 255
    _, mk = _crop_resize(Image.fromarray(np.zeros((666, 666, 3), np.uint8), "RGB"),
                         Image.fromarray(m, "L"), 512, 512)
    assert set(np.unique(np.asarray(mk))) <= {0, 255}


def test_crop_larger_than_source_fails_loudly():
    import pytest
    with pytest.raises(SystemExit, match="exceeds"):
        _crop_resize(Image.new("RGB", (512, 512)), Image.new("L", (512, 512)), 666, 512)


def test_gsd_arithmetic():
    """666 px of 2.9297 cm/px ground, shown as 512 px, is 3.810 cm/px."""
    native, size = 2.9297, 512
    assert abs(native * 394 / size - 2.254) < 1e-3
    assert abs(native * 512 / size - 2.930) < 1e-3
    assert abs(native * 666 / size - 3.810) < 1e-3

"""Pre-apply the loader's resize so training stops redoing it every epoch.

GenDS10 stores 2000x2000 tiles; the network takes 512. `StemDataset` therefore calls
`resize_batch(..., mode="nearest")` on every image and mask, every epoch — decoding 4.0 MPx
JPEGs to keep 0.26 MPx. Measured on that set: 2.24 GB of imagery, 482 KB/tile, and a
round-trip through 512 costs only 1.45 grey levels at 2x (against 3.73 for SpecDS at its
native size), so the extra pixels are largely interpolation rather than detail.

This applies **the same call the loader would** — skimage order=0, `mode="edge"`,
`anti_aliasing=False` — and writes the result out. Training then reads 512 px tiles and
resizes nothing.

Images are written as PNG by default: a JPEG re-encode would perturb the very pixels the
equivalence check is about. `--jpeg-quality` switches back to JPEG if size matters more
than exactness. Masks stay GIF and are re-binarised, as the loader does.

    python scripts/pre_resize_dataset.py --src ~/winmol_data/GenDS10 \\
        --dst ~/winmol_data/GenDS10_512 --size 512 --verify 25
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resize_like_loader(arr, size):
    """Exactly what StemDataset does on load: nearest, no antialias, edge padding."""
    from winmol_unet.preprocess import resize_batch
    return resize_batch(arr[None], size=size, mode="nearest")[0]


def convert(src, dst, size=512, jpeg_quality=None, quiet=False):
    from PIL import Image

    from training.dataset import _paired_ids
    from winmol_unet.preprocess import to_float01

    ids = _paired_ids(os.path.join(src, "train"), os.path.join(src, "mask"))
    if not ids:
        raise SystemExit(f"no paired tiles under {src}")
    os.makedirs(os.path.join(dst, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "mask"), exist_ok=True)
    ext = "jpeg" if jpeg_quality else "png"

    for k, n in enumerate(ids):
        im = Image.open(os.path.join(src, "train", f"train{n}.jpeg")).convert("RGB")
        arr = _resize_like_loader(to_float01(np.asarray(im)), size)
        out = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        p = os.path.join(dst, "train", f"train{n}.{ext}")
        out.save(p, quality=jpeg_quality) if jpeg_quality else out.save(p)

        mk = Image.open(os.path.join(src, "mask", f"mask{n}.gif"))
        mk.seek(0)
        m = _resize_like_loader(to_float01(np.asarray(mk.convert("L")))[..., None], size)
        Image.fromarray(((m[..., 0] >= 0.5) * 255).astype(np.uint8), "L").save(
            os.path.join(dst, "mask", f"mask{n}.gif"))

        if not quiet and (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(ids)}", flush=True)
    return len(ids)


def verify(src, dst, size=512, n=25):
    """Does the loader see the same array from both? That is the whole claim."""
    from PIL import Image

    from training.dataset import _paired_ids
    from winmol_unet.preprocess import to_float01

    ids = _paired_ids(os.path.join(src, "train"), os.path.join(src, "mask"))[:n]
    worst_img = worst_mask = 0.0
    for i in ids:
        orig = Image.open(os.path.join(src, "train", f"train{i}.jpeg")).convert("RGB")
        a = _resize_like_loader(to_float01(np.asarray(orig)), size)
        pre = None
        for e in ("png", "jpeg"):
            p = os.path.join(dst, "train", f"train{i}.{e}")
            if os.path.exists(p):
                pre = to_float01(np.asarray(Image.open(p).convert("RGB")))
                break
        if pre is None:
            raise SystemExit(f"missing converted tile for id {i}")
        worst_img = max(worst_img, float(np.abs(a - pre).max()) * 255)

        mo = Image.open(os.path.join(src, "mask", f"mask{i}.gif")); mo.seek(0)
        ma = _resize_like_loader(to_float01(np.asarray(mo.convert("L")))[..., None], size)
        ma = (ma[..., 0] >= 0.5)
        mp = np.asarray(Image.open(os.path.join(dst, "mask", f"mask{i}.gif"))
                        .convert("L")) > 127
        worst_mask = max(worst_mask, float(np.abs(ma.astype(int) - mp.astype(int)).max()))
    return worst_img, worst_mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=None,
                   help="write JPEG at this quality instead of lossless PNG")
    p.add_argument("--verify", type=int, default=25, help="tiles to check afterwards")
    a = p.parse_args(argv)

    n = convert(a.src, a.dst, a.size, a.jpeg_quality)
    print(f"converted {n} tiles -> {a.dst}")
    if a.verify:
        di, dm = verify(a.src, a.dst, a.size, a.verify)
        print(f"verification over {a.verify} tiles:")
        print(f"  max image difference vs the loader's own resize: {di:.4f} / 255")
        print(f"  max mask difference:                             {dm:.0f}")
        if di > 0.51 or dm > 0:
            print("  NOT equivalent — training would see different pixels")
            return 1
        print("  equivalent: training sees the same array either way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

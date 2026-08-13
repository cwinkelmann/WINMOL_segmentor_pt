"""Score a model across effective ground resolutions, from one oversized test set.

The Analyzer's effective GSD is `tile_size / 512`, a user-set knob, and a fixed-scale model
measured here lost 5.2 F1 across +/-30% zoom. This measures that curve directly.

**Why centre-crop one source rather than re-sample the orthomosaic per scale.** Sampling
five test sets at five extents changes the inward buffer (`extent * sqrt(2)/2`) at each
scale, so each set would sit on different ground and the sweep would confound scale with
location. Cropping one 666 px source keeps tile centres, ground and stem population fixed;
only resolution moves:

    effective GSD = native_gsd * crop_px / img_size

Resizing matches training: albumentations `RandomSizedCrop` uses bilinear for the image and
`mask_interpolation=0` (nearest) for the mask, so the same pair is used here. That keeps
test preprocessing identical to what the crops did in training — the point of comparison is
between arms at each scale, not against the Analyzer's own resampler.

    python scripts/scale_sweep.py --model runs/x/model.onnx \\
        --data-dir DS/test --crops 394 453 512 589 666 --native-gsd-cm 2.9297
"""
import argparse
import json
import os
import re

import numpy as np


def _crop_resize(im, mk, crop_px, size):
    """Centre-crop `crop_px` then resize to `size` — image bilinear, mask nearest."""
    from PIL import Image

    w, h = im.size
    if crop_px > min(w, h):
        raise SystemExit(f"crop {crop_px} exceeds the {w}x{h} source tile")
    off = ((w - crop_px) // 2, (h - crop_px) // 2)
    box = (off[0], off[1], off[0] + crop_px, off[1] + crop_px)
    return (im.crop(box).resize((size, size), Image.BILINEAR),
            mk.crop(box).resize((size, size), Image.NEAREST))


def sweep(model, data_dir, crops, size=512, native_gsd_cm=None, threshold=0.5,
          batch=8, limit=None):
    import onnxruntime as ort
    from PIL import Image

    img_dir, msk_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "mask")
    pat = re.compile(r"^train(\d+)\.jpe?g$")
    ids = sorted(int(m.group(1)) for m in
                 (pat.match(f) for f in os.listdir(img_dir)) if m)
    if limit:
        ids = ids[:limit]
    if not ids:
        raise SystemExit(f"no train<N>.jpeg under {img_dir}")

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    nhwc = len(inp.shape) == 4 and inp.shape[-1] in (3, "3")

    out = []
    for c in crops:
        tp = fp = fn = 0
        for i in range(0, len(ids), batch):
            xs, gs = [], []
            for n in ids[i:i + batch]:
                im = Image.open(os.path.join(img_dir, f"train{n}.jpeg")).convert("RGB")
                mk = Image.open(os.path.join(msk_dir, f"mask{n}.gif")).convert("L")
                im, mk = _crop_resize(im, mk, c, size)
                xs.append(np.asarray(im, np.float32) / 255.0)
                gs.append(np.asarray(mk) > 127)
            x = np.stack(xs)
            if not nhwc:
                x = np.transpose(x, (0, 3, 1, 2))
            y = np.squeeze(sess.run(None, {inp.name: x.astype(np.float32)})[0])
            if y.ndim == 2:
                y = y[None]
            pred, gt = y > threshold, np.stack(gs)
            tp += int((pred & gt).sum())
            fp += int((pred & ~gt).sum())
            fn += int((~pred & gt).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        row = {"crop_px": c, "f1": 2 * p * r / max(p + r, 1e-9),
               "precision": p, "recall": r, "tiles": len(ids)}
        if native_gsd_cm:
            row["gsd_cm"] = native_gsd_cm * c / size
            row["ratio"] = c / size
        out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True, help="test set of oversized tiles")
    p.add_argument("--crops", nargs="+", type=int, default=[394, 453, 512, 589, 666])
    p.add_argument("--size", type=int, default=512, help="model input")
    p.add_argument("--native-gsd-cm", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    rows = sweep(a.model, a.data_dir, a.crops, a.size, a.native_gsd_cm,
                 a.threshold, limit=a.limit)
    label = a.label or os.path.basename(os.path.dirname(a.model))
    hdr = f"{'crop':>6} {'ratio':>6} {'GSD cm':>7} {'F1':>7} {'P':>7} {'R':>7}"
    print(f"{label}   ({rows[0]['tiles']} tiles)")
    print(hdr)
    peak = max(r["f1"] for r in rows)
    for r in rows:
        print(f"{r['crop_px']:6d} {r.get('ratio', 0):6.2f} {r.get('gsd_cm', 0):7.3f} "
              f"{r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f}"
              + ("   <- peak" if r["f1"] == peak else f"   {r['f1']-peak:+.4f}"))
    worst = min(r["f1"] for r in rows)
    print(f"  peak {peak:.4f}   worst {worst:.4f}   spread {peak-worst:.4f}")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"label": label, "model": a.model, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

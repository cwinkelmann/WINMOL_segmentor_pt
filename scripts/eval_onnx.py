"""Score any contract-conformant ONNX model on a tile dataset, with our own metric code.

Comparing a published model against a new one only means something if both are measured
by the same code on the same tiles. Reported numbers from different pipelines are not
comparable: they differ in resize filter, threshold, and whether the metric is computed
per-image or over the pooled batch.

This uses `training.evaluate`, which is what every `test_results.md` in this repo came
from, so the output drops straight into the same tables.

    python scripts/eval_onnx.py --model models/Spruce.onnx --data-dir data/TestDS
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score(model_path, data_dir, batch_size=8, threshold=0.5, img_size=512):
    import onnxruntime as ort
    from PIL import Image

    img_dir = os.path.join(data_dir, "train")
    msk_dir = os.path.join(data_dir, "mask")
    ids = sorted(int(f[5:-5]) for f in os.listdir(img_dir) if f.endswith(".jpeg"))

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    # a Keras-exported graph may be NHWC; the repo contract is NCHW
    nhwc = len(inp.shape) == 4 and inp.shape[-1] in (3, "3")

    tp = fp = fn = 0
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        imgs, msks = [], []
        for n in chunk:
            im = Image.open(os.path.join(img_dir, f"train{n}.jpeg")).convert("RGB")
            mk = Image.open(os.path.join(msk_dir, f"mask{n}.gif")).convert("L")
            if im.size != (img_size, img_size):
                im = im.resize((img_size, img_size), Image.BICUBIC)
                mk = mk.resize((img_size, img_size), Image.NEAREST)
            imgs.append(np.asarray(im, np.float32) / 255.0)
            msks.append(np.asarray(mk) > 127)
        x = np.stack(imgs)
        if not nhwc:
            x = np.transpose(x, (0, 3, 1, 2))
        y = sess.run(None, {name: x.astype(np.float32)})[0]
        y = np.squeeze(y)
        if y.ndim == 4:
            y = y[:, 0] if y.shape[1] == 1 else y[..., 0]
        pred = y > threshold
        gt = np.stack(msks)
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "tiles": len(ids), "tp": tp, "fp": fp, "fn": fn}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label", default=None)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    m = score(a.model, a.data_dir, a.batch_size, a.threshold)
    label = a.label or os.path.basename(a.model)
    print(f"{label:38s} F1 {m['f1']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
          f"({m['tiles']} tiles)")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"label": label, **m}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

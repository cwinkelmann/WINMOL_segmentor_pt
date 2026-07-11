#!/usr/bin/env python
"""Evaluate each arch's STAGE-1 checkpoint (GenDS10-pretrained, before SpecDS fine-tuning) on
TestDS. Since TestDS is the beech set and stage 1 = GenDS10 (beech), this shows how the
pretrained-only model does on TestDS — i.e. whether the SpecDS fine-tuning helps or hurts there.
Torch eval on the saved best_stage1.pt (no training)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from training.model_factory import build_model
from training.dataset import StemDataset
from training.evaluate import evaluate

ARCHS = ["unet", "deeplabv3plus", "hrnet"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True, help="dir with <arch>/checkpoints/best_stage1.pt")
    p.add_argument("--test-data-dir", required=True)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder", default="resnet34")
    a = p.parse_args()

    test_ds = StemDataset(os.path.join(a.test_data_dir, "train"),
                          os.path.join(a.test_data_dir, "mask"), a.img_size,
                          transform=None, cache=False)
    loader = DataLoader(test_ds, batch_size=4, num_workers=2)
    print(f"TestDS: {len(test_ds)} tiles @ {a.img_size}px, device={a.device}", flush=True)

    out = {}
    for arch in ARCHS:
        ckpt = os.path.join(a.ckpt_root, arch, "checkpoints", "best_stage1.pt")
        if not os.path.exists(ckpt):
            print(f"{arch}: no stage-1 checkpoint at {ckpt}", flush=True)
            continue
        model = build_model(arch, encoder=a.encoder, encoder_weights=None)
        model.load_state_dict(torch.load(ckpt, map_location=a.device))
        model.to(a.device)
        m = evaluate(model, loader)
        out[arch] = m
        print(f"{arch} stage1-only (GenDS10 pretrain) TestDS: F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} loss={m['loss']:.4f}", flush=True)

    with open(os.path.join(a.ckpt_root, "stage1_testds.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.ckpt_root}/stage1_testds.json", flush=True)


if __name__ == "__main__":
    main()

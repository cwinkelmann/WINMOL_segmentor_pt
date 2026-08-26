"""Score an already-trained checkpoint on an arbitrary test set.

`run_train` evaluates on one `--test-data-dir` as part of training. A 2x2 needs each model
scored on *several* test sets, which otherwise means retraining. This loads the exported
.pt and runs the same `evaluate()` the training loop uses, so the numbers are produced by
identical code and are directly comparable to any `test_results.md`.

    python evaluate.py --model runs/aa-on/model.pt --arch hrnet \\
        --test-data-dir datasets/BeechAA_off/test --label "aa-on -> off"
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from winmol_unet.training.config import TrainConfig
from winmol_unet.training.dataset import StemDataset
from winmol_unet.training.evaluate import evaluate
from winmol_unet.training.model_factory import build_model


def score(model_path, arch, test_data_dir, encoder=None, batch_size=16, device="cuda",
          img_size=512, num_workers=4, width_mult=1.0):
    # width_mult must match what the checkpoint was trained at: a .pt is a bare state_dict,
    # so the architecture is rebuilt from these arguments and a mismatch fails on the first
    # load_state_dict with a wall of size-mismatch errors. The release ships width-0.5
    # models, so this is not a hypothetical.
    model = build_model(arch, encoder=encoder, encoder_weights=None, width_mult=width_mult)
    # weights_only=True: these are our own exports, and a plain state_dict needs no pickle
    # of arbitrary objects. torch flips this default in a later release anyway.
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" in state:
        state = state.get("state_dict", state)
    model.load_state_dict(state)

    from winmol_unet.training.device import resolve_device
    model.to(resolve_device(device)).eval()

    ds = StemDataset(os.path.join(test_data_dir, "train"),
                     os.path.join(test_data_dir, "mask"),
                     img_size, transform=None, cache=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return evaluate(model, loader), len(ds)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to model.pt")
    p.add_argument("--arch", required=True)
    p.add_argument("--encoder", default=None)
    p.add_argument("--test-data-dir", required=True)
    p.add_argument("--label", default=None, help="row label for the printed line")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    m, n = score(a.model, a.arch, a.test_data_dir, encoder=a.encoder,
                 batch_size=a.batch_size, device=a.device, num_workers=a.num_workers)
    label = a.label or f"{os.path.basename(os.path.dirname(a.model))} -> {a.test_data_dir}"
    print(f"{label}: F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
          f"loss={m['loss']:.4f}  ({n} tiles)")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"label": label, "model": a.model, "test_data_dir": a.test_data_dir,
                       "tiles": n, **m}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert a trained UNet `.pt` into a Keras `.hdf5` for the legacy WINMOL Analyzer.

    python scripts/pt_to_keras_hdf5.py model.pt model.hdf5 [--keras model.keras]

## Run it in its own environment

    python -m venv .venv-convert && . .venv-convert/bin/activate
    pip install -e "." tensorflow
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel

That keeps TensorFlow out of the segmentor environment: nothing here is imported
by training or by ONNX serving. The CPU torch wheel is used because the
conversion only reads tensors — no GPU is involved.

## Why from `.pt` rather than from the ONNX

The `.pt` state_dict carries the full topology — 114 tensors including every
BatchNorm's weight, bias, running mean and variance. The exported ONNX carries 29,
because constant folding **fuses BatchNorm into the preceding convolution**; those
parameters no longer exist separately and can only be reconstructed arithmetically.
Converting from `.pt` needs no reconstruction: the Keras mirror has exactly the same
layers, so weights are copied across one for one by the existing, tested transfer in
`winmol_unet.export_keras`.

## Scope

UNet only, as the Keras mirror is a hand-built copy of `winmol_unet/model.py`. A
state_dict from another architecture fails at load with a clear message rather than
producing a wrong model. `--width-mult` must match whatever the model was trained
with; a mismatch is caught by the state_dict shapes.
"""
import argparse
import sys


def convert(pt_path, hdf5_path, keras_path=None, width_mult=1.0, in_channels=None,
            check=True):
    import numpy as np
    import torch

    # _build_and_transfer returns the Keras model itself; export_to_keras_hdf5
    # returns the path, and we need the model for the numerical check
    from winmol_unet.export_keras import _build_and_transfer
    from winmol_unet.model import UNet

    state = torch.load(pt_path, map_location="cpu")
    if hasattr(state, "state_dict"):                 # a pickled module, not a state_dict
        state = state.state_dict()

    # infer the input channel count from the first conv rather than making the
    # caller remember whether this was an --rgbd run
    first = state.get("enc1.c1.0.weight")
    if first is None:
        raise SystemExit(
            "this state_dict has no 'enc1.c1.0.weight' — it is not a winmol_unet UNet. "
            "The Keras mirror is UNet-only; other architectures export to ONNX instead.")
    ch = int(first.shape[1]) if in_channels is None else in_channels

    kwargs = {"in_channels": ch}
    try:
        model = UNet(width_mult=width_mult, **kwargs)
    except TypeError:                                 # branch without width_mult
        if width_mult != 1.0:
            raise SystemExit("this winmol_unet build has no width_mult support")
        model = UNet(**kwargs)
    model.load_state_dict(state)
    model.eval()

    if ch != 3:
        print(f"warning: this is a {ch}-channel (RGBD) model. The legacy analyzer feeds "
              "3-channel tiles, so it will not be able to drive it.", file=sys.stderr)

    keras_model = _build_and_transfer(model)
    if check:
        _check(model, keras_model, ch)      # verify BEFORE writing anything
    keras_model.save(hdf5_path)
    if keras_path:
        keras_model.save(keras_path)
    return keras_model


def _check(torch_model, keras_model, channels, tol=2e-4, seed=0):
    """The transfer is only useful if both models agree on the same input."""
    import numpy as np
    import torch

    # the Keras mirror is built at the contract's fixed 512x512 input, so the
    # comparison has to use that size
    rng = np.random.default_rng(seed)
    x = rng.random((1, 512, 512, channels)).astype(np.float32)        # NHWC
    with torch.no_grad():
        t = torch.sigmoid(torch_model(torch.from_numpy(
            np.transpose(x, (0, 3, 1, 2)).copy()))).numpy()
    t = np.transpose(t, (0, 2, 3, 1))
    k = keras_model.predict(x, verbose=0)
    diff = float(np.abs(t - k).max())
    print(f"numerical check: max |torch - keras| = {diff:.2e} (tol {tol:g})")
    if diff > tol:
        raise SystemExit(
            f"converted model differs from the source by {diff:.3e}; refusing to treat "
            "this as a drop-in replacement")
    return diff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pt", help="trained UNet state_dict (.pt)")
    p.add_argument("hdf5", help="output .hdf5 for the legacy analyzer")
    p.add_argument("--keras", default=None, help="also write the native .keras format")
    p.add_argument("--width-mult", type=float, default=1.0,
                   help="must match the value the model was trained with")
    p.add_argument("--in-channels", type=int, default=None,
                   help="override the channel count inferred from the first conv")
    p.add_argument("--no-check", action="store_true",
                   help="skip the torch-vs-keras numerical comparison")
    a = p.parse_args(argv)
    convert(a.pt, a.hdf5, a.keras, a.width_mult, a.in_channels, check=not a.no_check)
    print(f"wrote {a.hdf5}" + (f" and {a.keras}" if a.keras else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

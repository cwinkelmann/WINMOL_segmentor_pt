"""Convert a WINMOL Keras HDF5 UNet to a contract-conformant ONNX file.

The upstream analyzer models are Keras (NHWC: [N,512,512,3] -> [N,512,512,1], sigmoid baked
in). The winmol_unet ONNX contract is NCHW ([N,3,512,512] -> [N,1,512,512]) served by
OnnxSegmenter, so this wraps the Keras graph with input/output Permute layers to expose NCHW,
converts with tf2onnx (opset 17), renames the I/O to "input"/"output", and validates the
contract. No retraining -- a pure format conversion; the weights are untouched.

Run in the winmol-convert image (TF 2.15 + tf2onnx).
"""
import argparse
import os

import onnx


def _wrap_nchw(base):
    """Wrap an NHWC Keras model so it takes/returns NCHW (matches the ONNX contract)."""
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Permute
    from tensorflow.keras.models import Model

    inp = Input(shape=(3, 512, 512), name="input")     # NCHW
    x = Permute((2, 3, 1))(inp)                          # -> NHWC (512,512,3)
    y = base(x)                                          # -> NHWC (512,512,1)
    out = Permute((3, 1, 2))(y)                          # -> NCHW (1,512,512)
    return Model(inp, out, name="winmol_unet_nchw")


def _rename_io(model, in_name="input", out_name="output"):
    g = model.graph
    old_in, old_out = g.input[0].name, g.output[0].name
    for node in g.node:
        node.input[:] = [in_name if t == old_in else t for t in node.input]
        node.output[:] = [out_name if t == old_out else t for t in node.output]
    g.input[0].name, g.output[0].name = in_name, out_name
    return model


def convert(hdf5_path, onnx_path, opset=17):
    import tensorflow as tf
    import tf2onnx
    from winmol_unet.contract import validate_onnx_model

    base = tf.keras.models.load_model(hdf5_path, compile=False)
    assert base.input_shape[1:] == (512, 512, 3), f"unexpected input {base.input_shape}"
    wrapped = _wrap_nchw(base)
    spec = (tf.TensorSpec((None, 3, 512, 512), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(wrapped, input_signature=spec, opset=opset)
    model_proto = _rename_io(model_proto)
    validate_onnx_model(model_proto)                     # enforce the frozen contract
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    onnx.save(model_proto, onnx_path)
    return onnx_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5")
    ap.add_argument("onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    convert(args.hdf5, args.onnx, args.opset)
    print(f"wrote {args.onnx} ({os.path.getsize(args.onnx) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Frozen ONNX I/O contract shared by training export and analyzer inference."""

IMG_SIZE = 512
IN_CHANNELS = 3
OUT_CHANNELS = 1
OPSET = 17

INPUT_NAME = "input"
OUTPUT_NAME = "output"
DYNAMIC_AXES = {"input": {0: "batch"}, "output": {0: "batch"}}

INPUT_SHAPE = ("batch", IN_CHANNELS, IMG_SIZE, IMG_SIZE)
OUTPUT_SHAPE = ("batch", OUT_CHANNELS, IMG_SIZE, IMG_SIZE)


def _dim_values(tensor_type):
    dims = []
    for d in tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def validate_onnx_model(onnx_model):
    """Raise ValueError if the model graph violates the contract."""
    graph = onnx_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Contract requires exactly one input and one output")

    got_in = _dim_values(graph.input[0].type.tensor_type)
    got_out = _dim_values(graph.output[0].type.tensor_type)

    # spatial + channel dims must match; batch dim must be symbolic (dynamic)
    if got_in[1:] != [IN_CHANNELS, IMG_SIZE, IMG_SIZE]:
        raise ValueError(f"Input shape {got_in} violates contract {INPUT_SHAPE}")
    if got_out[1:] != [OUT_CHANNELS, IMG_SIZE, IMG_SIZE]:
        raise ValueError(f"Output shape {got_out} violates contract {OUTPUT_SHAPE}")
    if isinstance(got_in[0], int) or isinstance(got_out[0], int):
        raise ValueError("Batch axis must be dynamic (symbolic), not fixed")

"""Frozen ONNX I/O contract shared by training export and analyzer inference."""

IMG_SIZE = 512
IN_CHANNELS = 3
RGBD_IN_CHANNELS = 4
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


def _spatial_ok(dim):
    # A spatial dim conforms if it is fixed 512 OR dynamic (symbolic string).
    # Some architectures (e.g. smp HRNet, whose decoder uses a Resize op) export
    # with symbolic spatial dims; at runtime a 512x512 input still yields 512x512
    # output, which is all the analyzer ever feeds. A WRONG fixed size (e.g. 256)
    # is still rejected.
    return dim == IMG_SIZE or isinstance(dim, str)


def _check_shape(got, channels, label):
    if len(got) != 4:
        raise ValueError(f"{label} must be 4D NCHW, got {got}")
    if isinstance(got[0], int):
        raise ValueError(f"{label} batch axis must be dynamic (symbolic), not fixed: {got}")
    if got[1] != channels:
        raise ValueError(f"{label} channel dim must be {channels}, got {got}")
    if not (_spatial_ok(got[2]) and _spatial_ok(got[3])):
        raise ValueError(f"{label} spatial dims must be {IMG_SIZE} or dynamic, got {got}")


def validate_onnx_model(onnx_model, in_channels=IN_CHANNELS):
    """Raise ValueError if the model graph violates the contract.

    Batch axis must be dynamic; channels fixed (`in_channels` in — default 3,
    pass RGBD_IN_CHANNELS for RGBD models — 1 out); spatial dims either fixed
    512 or dynamic (see _spatial_ok).
    """
    graph = onnx_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Contract requires exactly one input and one output")

    _check_shape(_dim_values(graph.input[0].type.tensor_type), in_channels, "Input")
    _check_shape(_dim_values(graph.output[0].type.tensor_type), OUT_CHANNELS, "Output")

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


def _sigmoid_on_path(graph):
    """True if a Sigmoid feeds the graph output, directly or through later ops.

    Deliberately NOT "the last node is a Sigmoid". Real conformant models append
    ops after it: fp16 conversion adds Cast, static int8 adds
    QuantizeLinear/DequantizeLinear, and the Keras converter adds Reshape. A
    terminal-node test rejects 11 of the 15 models this project has shipped or
    consumes, every one of which emits probabilities.

    Walking back from the output (rather than scanning the whole graph) means a
    Sigmoid on a dead branch does not satisfy the contract.
    """
    producer = {}
    for node in graph.node:
        for name in node.output:
            producer[name] = node

    seen = set()
    stack = [graph.output[0].name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        node = producer.get(name)
        if node is None:                      # graph input or initializer
            continue
        if node.op_type == "Sigmoid":
            return True
        stack.extend(node.input)
    return False


def _opset_version(onnx_model):
    """Version of the default (ai.onnx) domain, or None if it is absent."""
    for entry in onnx_model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return None


def validate_onnx_model(onnx_model):
    """Raise ValueError if the model graph violates the contract.

    Batch axis must be dynamic; channels fixed (3 in / 1 out); spatial dims either
    fixed 512 or dynamic (see _spatial_ok); opset at least OPSET; and a sigmoid on
    the path to the output, so the graph emits probabilities rather than logits.
    """
    graph = onnx_model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Contract requires exactly one input and one output")

    _check_shape(_dim_values(graph.input[0].type.tensor_type), IN_CHANNELS, "Input")
    _check_shape(_dim_values(graph.output[0].type.tensor_type), OUT_CHANNELS, "Output")

    version = _opset_version(onnx_model)
    if version is None:
        raise ValueError("Contract requires an ai.onnx opset import; none found")
    if version < OPSET:
        # `>=`, not `==`: a later re-export is not a breach, a stale one is.
        raise ValueError(f"Contract requires opset >= {OPSET}, got {version}")

    if not _sigmoid_on_path(graph):
        raise ValueError(
            "Contract requires a Sigmoid on the path to the output so the model "
            "emits probabilities; found none. The analyzer thresholds at 0.5 and "
            "would read raw logits as probabilities.")

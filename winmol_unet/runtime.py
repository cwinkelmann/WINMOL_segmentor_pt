"""Runtime adapter that makes an ONNX model duck-type the Keras model object."""
import numpy as np
import onnxruntime as ort

from .contract import IMG_SIZE, INPUT_NAME, OUTPUT_NAME


class OnnxOutOfMemoryError(RuntimeError):
    """Raised on ONNX runtime OOM; caught by the analyzer's batch-backoff loop."""


def _default_providers():
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxSegmenter:
    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or _default_providers()
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        # Channel count comes from the loaded graph (3 = RGB, 4 = RGBD), so one
        # adapter serves both model families without a mode switch.
        self.in_channels = int(self.session.get_inputs()[0].shape[1])

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "numpy"):        # TF tensor / torch tensor
            x = x.numpy()
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    def predict_on_batch(self, x):
        x = self._as_numpy(x)          # NHWC [N,512,512,3]
        if x.shape[3] != self.in_channels:
            raise ValueError(
                f"model expects NHWC with C={self.in_channels}, got {x.shape}")
        nchw = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
        try:
            out = self.session.run([OUTPUT_NAME], {INPUT_NAME: nchw})[0]
        except Exception as exc:       # normalize OOM for the retry loop
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg or "cudaerror" in msg:
                raise OnnxOutOfMemoryError(str(exc)) from exc
            raise
        return np.transpose(out, (0, 2, 3, 1)).astype(np.float32)  # NHWC [N,512,512,1]

    def summary(self):
        print(
            f"OnnxSegmenter(providers={self.providers}) "
            f"input=[N,{IMG_SIZE},{IMG_SIZE},{self.in_channels}] -> output=[N,{IMG_SIZE},{IMG_SIZE},1] (NHWC)"
        )

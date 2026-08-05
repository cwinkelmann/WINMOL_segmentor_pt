"""Runtime adapter that makes an ONNX model duck-type the Keras model object."""
import os

import numpy as np
import onnxruntime as ort

from .contract import IMG_SIZE, INPUT_NAME, OUTPUT_NAME


class OnnxOutOfMemoryError(RuntimeError):
    """Raised on ONNX runtime OOM; caught by the analyzer's batch-backoff loop."""


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _default_providers():
    """Select execution providers, preferring an available accelerator.

    Precedence:
      1. ``WINMOL_ONNX_PROVIDERS`` - explicit comma-separated provider list
         (highest priority; lets a caller pin an exact configuration).
      2. ``WINMOL_ONNX_FORCE_CPU`` - force CPU only, for exact fp32 parity with
         the PyTorch/Keras reference (CoreML/CUDA may compute in fp16).
      3. ``CUDAExecutionProvider`` when available (NVIDIA GPUs).
      4. ``CoreMLExecutionProvider`` when available (Apple GPU + Neural Engine
         on macOS). Unsupported subgraphs fall back to CPU automatically.
      5. ``CPUExecutionProvider``.

    CPU is always appended as a fallback so any op an accelerator cannot run
    still executes.
    """
    override = os.environ.get("WINMOL_ONNX_PROVIDERS")
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    if _truthy(os.environ.get("WINMOL_ONNX_FORCE_CPU", "")):
        return ["CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in avail:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxSegmenter:
    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or _default_providers()
        self.session = ort.InferenceSession(model_path, providers=self.providers)

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "numpy"):        # TF tensor / torch tensor
            x = x.numpy()
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    def predict_on_batch(self, x):
        x = self._as_numpy(x)          # NHWC [N,512,512,3]
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
            f"input=[N,{IMG_SIZE},{IMG_SIZE},3] -> output=[N,{IMG_SIZE},{IMG_SIZE},1] (NHWC)"
        )

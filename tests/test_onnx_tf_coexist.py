"""onnx and TensorFlow must survive in one process — that is the --export-keras path.

`training/run_train.py` imports `winmol_unet.export` (and so `onnx`) at module level, then
imports TensorFlow lazily when `--export-keras` is set. Both packages ship their own C++
protobuf runtime, and before onnx 1.18 those were built against incompatible protobuf
majors: whichever loaded second either aborted the interpreter ("mutex lock failed:
Invalid argument") or deadlocked ("RAW: Lock blocking"), hanging a finished training run
at the export step.

pip cannot catch this. Every declared range is satisfied by the broken pair — the clash is
at the native ABI level, below anything a resolver inspects. So it needs a test.

**Nothing here imports TensorFlow or onnx into the test session.** Detection uses
`find_spec`, and every actual import happens in a subprocess. That is deliberate: on a
broken pair, importing them here would abort the interpreter and take the whole suite down
with no report — which is exactly how this bug stayed invisible.

The subprocess tests skip when TensorFlow is absent (CI, and any install without the
`[keras]` extra). The version assertion always runs, so a too-old onnx is still caught.
"""
import importlib.util
import subprocess
import sys

import pytest

HAS_TF = importlib.util.find_spec("tensorflow") is not None


@pytest.mark.skipif(not HAS_TF, reason="only reproducible with the [keras] extra installed")
# a fresh interpreter per order — once a process has loaded either library the outcome is
# already decided, so both orders must start from scratch
@pytest.mark.parametrize("first,second", [("onnx", "tensorflow"), ("tensorflow", "onnx")])
def test_imports_coexist_in_one_process(first, second):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {first}, {second}; print('ok')"],
        # generous for a cold TensorFlow import, short enough that the deadlocking
        # pair reports in ~2 min instead of stalling the suite
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "ok" in proc.stdout, (
        f"importing {first} then {second} failed (exit {proc.returncode}). This is the "
        f"onnx/TensorFlow protobuf clash — onnx must be >=1.18.\n{proc.stderr[-1500:]}")


def test_onnx_is_new_enough_for_protobuf_6():
    """Runs everywhere: reads the version from metadata, without importing onnx."""
    from importlib.metadata import version

    v = version("onnx")
    major, minor = (int(p) for p in v.split(".")[:2])
    assert (major, minor) >= (1, 18), (
        f"onnx {v} predates protobuf 6 support; with TensorFlow installed it deadlocks or "
        f"aborts on import, breaking --export-keras. pyproject pins onnx>=1.18.")

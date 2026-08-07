"""Importing the training package must not pull in TensorFlow.

TF is opt-in (`--export-keras`, the `[keras]` extra). When it *is* installed alongside
torch, importing it after torch's native libraries are loaded deadlocks inside TF's
abseil mutex — `import training.run_train` hangs forever, taking the whole test suite
with it. CI never caught this because CI does not install TF.

A plain `pytest tests/test_no_eager_tensorflow.py` cannot catch it either: by then the
test session has already imported half the tree. So this runs a fresh subprocess and
asserts on its `sys.modules`.
"""
import subprocess
import sys

import pytest

# each import is run in a fresh interpreter, so the check is real rather than
# dependent on what the enclosing test session happened to import first
MODULES = ["training.run_train", "training.train", "training.run_logger",
           "winmol_unet.runtime", "winmol_unet.contract"]

PROBE = """
import sys
import {mod}
heavy = sorted(m for m in ("tensorflow", "keras") if m in sys.modules)
print(",".join(heavy))
"""


@pytest.mark.parametrize("mod", MODULES)
def test_import_does_not_load_tensorflow(mod):
    proc = subprocess.run([sys.executable, "-c", PROBE.format(mod=mod)],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"importing {mod} failed:\n{proc.stderr[-2000:]}"
    loaded = proc.stdout.strip()
    assert not loaded, (
        f"importing {mod} eagerly loaded {loaded}. TensorFlow must stay lazy: with torch "
        f"already loaded, importing it deadlocks on an abseil mutex and hangs the suite.")

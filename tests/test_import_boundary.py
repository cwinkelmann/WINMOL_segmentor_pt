"""The analyzer installs `winmol_unet` with no torch and no TensorFlow.

This is the constraint the package layout exists to protect, and it is the one that
breaks silently: adding `from .training import ...` to an `__init__.py` costs nothing
locally — where torch is installed anyway — and turns `pip install winmol_unet` into a
2 GB download plus a hard failure on any machine without it.

Folding the old top-level `training/` package into `winmol_unet.training` put a
torch-importing subpackage inside the shipped one for the first time, so the invariant
needs a test rather than a convention.

Run in a subprocess: once pytest has imported the training tests, torch is in this
process's sys.modules and an in-process check would pass for the wrong reason.
"""
import subprocess
import sys
import textwrap

import pytest

HEAVY = ("torch", "tensorflow", "keras", "albumentations", "segmentation_models_pytorch",
         "torchvision", "wandb", "rasterio", "fiona", "shapely")


def _import_in_subprocess(statement):
    """Import `statement` in a clean interpreter; return the heavy modules it pulled in."""
    code = textwrap.dedent(f"""
        import sys
        {statement}
        heavy = [m for m in {HEAVY!r} if m in sys.modules]
        print(",".join(heavy))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"import failed:\n{out.stderr}"
    return [m for m in out.stdout.strip().split(",") if m]


def test_importing_the_package_pulls_in_no_training_dependency():
    assert _import_in_subprocess("import winmol_unet") == []


@pytest.mark.parametrize("module", ["winmol_unet.contract", "winmol_unet.runtime",
                                    "winmol_unet.preprocess"])
def test_analyzer_facing_modules_stay_light(module):
    """These three are what the analyzer actually imports to serve ONNX."""
    assert _import_in_subprocess(f"import {module}") == []


def test_the_subpackages_are_reachable_when_asked_for():
    """The boundary is about *eager* imports, not about hiding the training code."""
    out = subprocess.run(
        [sys.executable, "-c", "import winmol_unet.training.config as c; print(c.TrainConfig)"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "TrainConfig" in out.stdout


# Relocated from tests/test_no_eager_tensorflow.py (formerly its only test), rather than
# folded into tests/test_keras_bridge.py with the rest of the Keras bridge suite: that file
# is guarded per-test by `pytest.importorskip("tensorflow")` wherever TensorFlow is actually
# used, but this test's whole point is to catch code that eagerly imports TensorFlow, so it
# must keep running -- and be able to fail -- in a process that never imported TensorFlow at
# all. TF is opt-in (`--export-keras`, the `[keras]` extra). When it *is* installed alongside
# torch, importing it after torch's native libraries are loaded deadlocks inside TF's abseil
# mutex -- `import winmol_unet.training.run_train` hangs forever, taking the whole test suite
# with it. CI never catches this directly because CI does not install TF; this still runs
# there and passes vacuously (nothing loads TF because TF isn't present to load).
@pytest.mark.parametrize("module", ["winmol_unet.training.run_train", "winmol_unet.training.train",
                                    "winmol_unet.training.run_logger", "winmol_unet.runtime",
                                    "winmol_unet.contract"])
def test_training_modules_do_not_eagerly_import_tensorflow(module):
    loaded = [m for m in _import_in_subprocess(f"import {module}") if m in ("tensorflow", "keras")]
    assert not loaded, (
        f"importing {module} eagerly loaded {loaded}. TensorFlow must stay lazy: with torch "
        f"already loaded, importing it deadlocks on an abseil mutex and hangs the suite.")

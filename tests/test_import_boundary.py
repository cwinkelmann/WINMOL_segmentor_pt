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

HEAVY = ("torch", "tensorflow", "albumentations", "segmentation_models_pytorch",
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

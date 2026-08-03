"""Label-safe background inpainting: add vegetation without invalidating masks.

A plain img2img pass over a synthetic tile silently breaks the dataset — the
generator nudges stems by a few pixels and the mask no longer describes the
image, so training proceeds on quietly wrong labels. This module constrains
generation to the background and then *hard-composites* the result, so every
labelled pixel is bit-identical to what the renderer produced. The mask stays
exactly valid by construction rather than by hoping the model behaved.

Diffusion itself lives in scripts/inpaint_vegetation.py; everything here is
numpy, so the guarantee is testable without a GPU.
"""
import numpy as np
from scipy import ndimage as ndi


def repaint_region(stem_mask, margin_px=3):
    """Boolean map of pixels the generator may repaint.

    Everything that is not a stem, minus a `margin_px` collar around the stems:
    inpainting blends across the mask boundary, so without a collar vegetation
    creeps over stem edges and erodes the very labels we are protecting.
    """
    stem = np.asarray(stem_mask, bool)
    protected = ndi.binary_dilation(stem, iterations=margin_px) if margin_px > 0 else stem
    return ~protected


def composite(original, generated, region):
    """Take `generated` inside `region`, `original` everywhere else.

    Diffusion pipelines round-trip the whole image through a VAE, so even
    "untouched" pixels come back subtly changed. Compositing here — rather than
    trusting the pipeline's own blending — is what makes the guarantee exact.
    """
    original = np.asarray(original)
    generated = np.asarray(generated)
    region = np.asarray(region, bool)
    if original.shape != generated.shape or region.shape != original.shape[:2]:
        raise ValueError(
            f"shape mismatch: original {original.shape}, generated {generated.shape}, "
            f"region {region.shape}")
    out = original.copy()
    out[region] = generated[region]
    return out


def stems_are_intact(original, result, stem_mask):
    """True when every labelled pixel survived unchanged — the invariant that
    keeps an inpainted tile usable as training data."""
    stem = np.asarray(stem_mask, bool)
    return bool(np.array_equal(np.asarray(original)[stem], np.asarray(result)[stem]))


def vegetation_coverage(original, result, region):
    """Share of the repaintable area the generator actually changed.

    Near 0 means the prompt did nothing (a wasted pass); near 1 with a bland
    prompt usually means it repainted the whole floor into something uniform.
    Reported per tile so a bad prompt is visible instead of silent.
    """
    region = np.asarray(region, bool)
    if not region.any():
        return 0.0
    diff = np.abs(np.asarray(original, np.int16) - np.asarray(result, np.int16))
    changed = diff.max(axis=2) > 8 if diff.ndim == 3 else diff > 8
    return float(changed[region].mean())

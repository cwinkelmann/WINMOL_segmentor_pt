# Depth Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/simulate_depth.py` generates synthetic terrain depth maps (`depth/depth{N}.png`, uint16, native mask resolution) from existing `mask/mask{N}.gif` files, per the approved spec `docs/superpowers/specs/2026-07-29-depth-simulator-design.md`.

**Architecture:** A single script with pure numpy/scipy generation functions (`fractal_terrain`, `bulge_from`, `simulate_depth_map`) and a thin CLI (`main(argv)`) doing discovery/IO — same shape as `scripts/build_dataset.py`, tested the same way (`sys.path.insert` + direct import). The generation model was validated visually via a prototype (see the PDF from the design session); this plan productionizes that exact model.

**Tech Stack:** numpy, PIL, `scipy.ndimage` (`zoom`, `gaussian_filter`, `label`, `distance_transform_edt`) — scipy is already a transitive dependency of scikit-image; **no new install requirements**. pytest for tests.

## Global Constraints

- Output: `<DS>/depth/depth{N}.png`, **uint16** (PIL mode `I;16`), **native mask resolution**, arbitrary relative-height units (downstream `normalize_depth` min-max normalizes per image).
- Integer-N pairing, same tolerance as the loader: only `mask{N}.gif` with digit-only N is processed; other files ignored.
- Deterministic: per-image seed derived from `(--seed, N)` via `np.random.SeedSequence` — same (seed, N) → byte-identical output regardless of processing order.
- Never write into an existing `<DS>/depth/` without `--overwrite`; never silently mix old and new files.
- Anti-leakage defaults: `--stem-drop-p 0.2`; distractor count sampled in `[d-1, d+2]` for `--distractors d` (default 3 → 2–5).
- Documented behavior (from prototyping): **crossing stems merge into one connected component** — the junction gets a larger inradius and bulges higher, resembling piled logs. Intentional; do not "fix".
- Tests hermetic: synthetic masks in `tmp_path`, no network, no real data.
- Heavy deps stay out of `winmol_unet/` — this is a `scripts/` tool; it must not be imported by anything in `winmol_unet/`.

---

### Task 1: Generation model (pure functions)

**Files:**
- Create: `scripts/simulate_depth.py`
- Test: `tests/test_simulate_depth.py` (create)

**Interfaces:**
- Consumes: nothing in-repo (numpy/scipy/PIL only).
- Produces (used by Task 2's CLI):
  - `fractal_terrain(shape, rng, amplitude) -> float32 HW` — fractal heightfield, std ≈ `amplitude`.
  - `bulge_from(region_bool_hw) -> (float32 HW, float r)` — rounded cylindrical bulge and the region's max inradius.
  - `simulate_depth_map(mask_bool_hw, rng, stem_drop_p=0.2, distractors=3) -> float32 HW` — full model: terrain + per-component bulges (with dropout) + distractor bulges + noise + blur.

- [ ] **Step 1: Create the test file with failing unit tests**

```python
"""Tests for scripts/simulate_depth.py (synthetic terrain depth from stem masks)."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from simulate_depth import bulge_from, fractal_terrain, simulate_depth_map


def _stem_mask(size=128, y0=40, y1=56, x0=10, x1=118):
    """Horizontal bar 'stem' mask (16px wide -> inradius ~8)."""
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_fractal_terrain_shape_and_amplitude():
    rng = np.random.default_rng(0)
    t = fractal_terrain((128, 128), rng, amplitude=4.0)
    assert t.shape == (128, 128) and t.dtype == np.float32
    assert 3.0 < t.std() < 5.0                      # std normalized to ~amplitude


def test_bulge_is_rounded_and_scaled_by_inradius():
    m = _stem_mask()
    h, r = bulge_from(m)
    assert h.shape == m.shape
    assert h[~m].max() == 0.0                       # bulge only inside the region
    assert abs(h.max() - r) / r < 0.15              # peak ~ inradius (half-buried cylinder)
    # rounded: interior height exceeds edge height (no flat plateau / step)
    edge_h = h[48, 10]                              # on the stem edge column
    center_h = h[48, 64]                            # stem center
    assert center_h > edge_h


def test_simulate_deterministic_per_rng_seed():
    m = _stem_mask()
    a = simulate_depth_map(m, np.random.default_rng(42))
    b = simulate_depth_map(m, np.random.default_rng(42))
    c = simulate_depth_map(m, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stems_elevated_above_nearby_ground():
    from scipy.ndimage import binary_dilation
    m = _stem_mask()
    d = simulate_depth_map(m, np.random.default_rng(0), stem_drop_p=0.0, distractors=0)
    ring = binary_dilation(m, iterations=6) & ~m    # nearby ground
    assert d[m].mean() > d[ring].mean() + 1.0       # clearly above local terrain


def test_drop_p_one_removes_mask_correlation():
    from scipy.ndimage import binary_dilation
    m = _stem_mask()
    d = simulate_depth_map(m, np.random.default_rng(0), stem_drop_p=1.0, distractors=0)
    ring = binary_dilation(m, iterations=6) & ~m
    assert abs(d[m].mean() - d[ring].mean()) < 1.0  # no bulge left on the stem


def test_distractors_add_offmask_bulges():
    m = _stem_mask()
    flat = simulate_depth_map(m, np.random.default_rng(7), stem_drop_p=1.0, distractors=0)
    with_d = simulate_depth_map(m, np.random.default_rng(7), stem_drop_p=1.0, distractors=3)
    # same rng seed consumes identical draws up to the distractor stage, so any
    # difference off-mask comes from distractor bulges
    assert (with_d - flat).max() > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulate_depth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simulate_depth'`.

- [ ] **Step 3: Implement the generation functions in `scripts/simulate_depth.py`**

```python
"""Generate synthetic terrain depth maps (depth{N}.png) from stem masks (mask{N}.gif).

Design: docs/superpowers/specs/2026-07-29-depth-simulator-design.md. Synthetic
pretraining/plumbing data for --rgbd, NOT evaluation-grade depth: fractal terrain
+ per-stem cylindrical bulges, degraded by bulge dropout, off-mask distractor
bulges, and sensor noise so depth is a helpful-but-unreliable cue (anti-leakage).

Crossing stems merge into one connected component; the junction gets a larger
inradius and bulges higher, resembling piled logs. Intentional.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def fractal_terrain(shape, rng, amplitude):
    """Fractal heightfield: octaves at 64/16/4 px feature scales, amplitude
    halving per octave, std normalized to `amplitude`."""
    out = np.zeros(shape, np.float32)
    for i, scale in enumerate((64, 16, 4)):
        coarse = rng.standard_normal((shape[0] // scale + 2, shape[1] // scale + 2))
        up = ndi.zoom(coarse, scale, order=3)[: shape[0], : shape[1]]
        out += ndi.gaussian_filter(up, scale / 4) * (0.5 ** i)
    return ((out / out.std()) * amplitude).astype(np.float32)


def bulge_from(region):
    """Half-buried-cylinder bulge for a boolean region: inward distance d gives
    centerline offset x = r - d, so h = r*sqrt(1-(x/r)^2) = r*sqrt(2u-u^2), u=d/r.
    Returns (height HW float32, max inradius r)."""
    d = ndi.distance_transform_edt(region)
    r = float(d.max())
    if r == 0:
        return np.zeros(region.shape, np.float32), 0.0
    u = d / r
    return (r * np.sqrt(np.clip(2 * u - u * u, 0, 1))).astype(np.float32), r


def _distractor_region(shape, rng):
    """Random rotated ellipse (rock / root plate / debris)."""
    cy, cx = rng.integers(15, shape[0] - 15), rng.integers(15, shape[1] - 15)
    ry, rx = rng.uniform(4, 11, 2)
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0: shape[0], 0: shape[1]]
    ys, xs = yy - cy, xx - cx
    yr = ys * np.cos(ang) - xs * np.sin(ang)
    xr = ys * np.sin(ang) + xs * np.cos(ang)
    return (yr / ry) ** 2 + (xr / rx) ** 2 <= 1


def simulate_depth_map(mask, rng, stem_drop_p=0.2, distractors=3):
    """Full model for one mask -> float32 heightfield (arbitrary relative units)."""
    mask = np.asarray(mask, bool)
    labels, n = ndi.label(mask)
    radii = [ndi.distance_transform_edt(labels == k).max() for k in range(1, n + 1)]
    # terrain amplitude ~ same order as stem height so depth alone can't
    # trivially separate ground from stem
    amp = 0.5 * (float(np.median(radii)) if radii else 6.0)
    depth = fractal_terrain(mask.shape, rng, amp)

    for k in range(1, n + 1):                       # per-stem bulge with dropout
        if rng.random() < stem_drop_p:
            continue
        h, _ = bulge_from(labels == k)
        depth += h

    if distractors > 0:                             # sampled in [d-1, d+2]
        for _ in range(int(rng.integers(distractors - 1, distractors + 3))):
            h, _ = bulge_from(_distractor_region(mask.shape, rng))
            depth += h

    depth += rng.standard_normal(mask.shape).astype(np.float32) * 0.05 * amp
    return ndi.gaussian_filter(depth, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulate_depth.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/simulate_depth.py tests/test_simulate_depth.py
git commit -m "feat(sim): terrain + cylindrical stem-bulge depth generation model"
```

---

### Task 2: CLI, file IO, mismatch control

**Files:**
- Modify: `scripts/simulate_depth.py` (append CLI section)
- Test: `tests/test_simulate_depth.py` (append)

**Interfaces:**
- Consumes: `simulate_depth_map(mask, rng, stem_drop_p, distractors)` from Task 1.
- Produces: `main(argv=None) -> int` (0 on success, 2 on usage error) and the CLI:
  `python scripts/simulate_depth.py --dataset <DS> [--seed 1] [--stem-drop-p 0.2] [--distractors 3] [--overwrite] [--mismatch]`.
  `--mismatch` is the leakage-control mode: `depth{N}` is generated from a *different* image's mask (ids rotated by one), for the RGB-vs-RGBD experiment's third arm.

- [ ] **Step 1: Append failing CLI tests**

```python
from simulate_depth import main


def _write_mask(mask_dir, n, size=96):
    os.makedirs(mask_dir, exist_ok=True)
    arr = np.zeros((size, size), np.uint8)
    arr[20 + n: 36 + n, 8: size - 8] = 255          # position varies with n
    Image.fromarray(arr).convert("P").save(os.path.join(mask_dir, f"mask{n}.gif"))


def _read_depth(ds, n):
    return np.asarray(Image.open(os.path.join(ds, "depth", f"depth{n}.png")))


def test_cli_writes_one_depth_per_mask(tmp_path):
    ds = str(tmp_path)
    for n in (1, 2, 7):
        _write_mask(os.path.join(ds, "mask"), n)
    assert main(["--dataset", ds, "--seed", "1"]) == 0
    for n in (1, 2, 7):
        arr = _read_depth(ds, n)
        assert arr.shape == (96, 96)
        assert arr.dtype in (np.uint16, np.int32)   # PIL I;16 may decode as int32
        assert arr.max() > arr.min()                # min-max scaled to uint16 range


def test_cli_deterministic_and_order_independent(tmp_path):
    ds1, ds2 = str(tmp_path / "a"), str(tmp_path / "b")
    for ds, ns in ((ds1, (1, 2)), (ds2, (2,))):     # ds2 lacks mask1
        for n in ns:
            _write_mask(os.path.join(ds, "mask"), n)
    main(["--dataset", ds1, "--seed", "5"])
    main(["--dataset", ds2, "--seed", "5"])
    # per-image seed depends on (seed, N) only, not on which other ids exist
    np.testing.assert_array_equal(_read_depth(ds1, 2), _read_depth(ds2, 2))


def test_cli_overwrite_guard(tmp_path):
    ds = str(tmp_path)
    _write_mask(os.path.join(ds, "mask"), 1)
    assert main(["--dataset", ds]) == 0
    assert main(["--dataset", ds]) == 2             # refuses without --overwrite
    assert main(["--dataset", ds, "--overwrite"]) == 0


def test_cli_missing_mask_dir_errors(tmp_path):
    assert main(["--dataset", str(tmp_path / "nope")]) == 2


def test_cli_mismatch_pairs_depth_with_other_mask(tmp_path):
    ds_m, ds_f = str(tmp_path / "mm"), str(tmp_path / "faith")
    for ds in (ds_m, ds_f):
        for n in (1, 2):
            _write_mask(os.path.join(ds, "mask"), n)
    main(["--dataset", ds_f, "--seed", "3"])
    main(["--dataset", ds_m, "--seed", "3", "--mismatch"])
    # mismatch: depth1 is generated from mask2 (ids rotated), under depth1's seed
    assert not np.array_equal(_read_depth(ds_m, 1), _read_depth(ds_f, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulate_depth.py -v -k cli`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Append the CLI implementation to `scripts/simulate_depth.py`**

```python
def _mask_ids(mask_dir):
    out = {}
    for name in os.listdir(mask_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".gif" and stem.startswith("mask") and stem[len("mask"):].isdigit():
            out[int(stem[len("mask"):])] = name
    return out


def _load_mask(path):
    im = Image.open(path)
    im.seek(0)
    return np.asarray(im.convert("L")) >= 128       # same binarization as the loader


def _save_uint16(depth, path):
    lo, hi = float(depth.min()), float(depth.max())
    scaled = np.zeros_like(depth) if hi <= lo else (depth - lo) / (hi - lo)
    Image.fromarray((scaled * 65535).astype(np.uint16), mode="I;16").save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset dir containing mask/mask{N}.gif")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stem-drop-p", type=float, default=0.2)
    p.add_argument("--distractors", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mismatch", action="store_true",
                   help="leakage control: generate depth{N} from a DIFFERENT image's "
                        "mask (ids rotated by one)")
    a = p.parse_args(argv)

    mask_dir = os.path.join(a.dataset, "mask")
    depth_dir = os.path.join(a.dataset, "depth")
    if not os.path.isdir(mask_dir):
        print(f"error: {mask_dir} not found (expected <DS>/mask/mask{{N}}.gif)",
              file=sys.stderr)
        return 2
    if os.path.isdir(depth_dir) and os.listdir(depth_dir) and not a.overwrite:
        print(f"error: {depth_dir} exists; pass --overwrite to regenerate",
              file=sys.stderr)
        return 2
    ids = _mask_ids(mask_dir)
    if not ids:
        print(f"error: no mask{{N}}.gif files in {mask_dir}", file=sys.stderr)
        return 2

    os.makedirs(depth_dir, exist_ok=True)
    ordered = sorted(ids)
    # mismatch: depth{N} uses the NEXT id's mask (rotation = a derangement for >1 id)
    source = {n: ordered[(i + 1) % len(ordered)] if a.mismatch else n
              for i, n in enumerate(ordered)}
    for n in ordered:
        mask = _load_mask(os.path.join(mask_dir, ids[source[n]]))
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, n]))
        depth = simulate_depth_map(mask, rng, a.stem_drop_p, a.distractors)
        _save_uint16(depth, os.path.join(depth_dir, f"depth{n}.png"))
    print(f"wrote {len(ordered)} depth maps to {depth_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulate_depth.py -v`
Expected: PASS (all 11).

- [ ] **Step 5: Commit**

```bash
git add scripts/simulate_depth.py tests/test_simulate_depth.py
git commit -m "feat(sim): simulate_depth CLI (depth{N}.png writer, overwrite guard, --mismatch control)"
```

---

### Task 3: Experiment runbook + docs

**Files:**
- Create: `docs/rgbd-experiment.md`
- Modify: `CLAUDE.md` (tooling entry points), `docs/FEATURES.md`

**Interfaces:**
- Consumes: the Task 2 CLI; the `--rgbd` flag from the RGBD plan (`docs/superpowers/plans/2026-07-29-rgbd-input.md`) — the runbook is a document, so it may reference that flag before the RGBD plan is executed.
- Produces: the written experiment protocol; no code.

- [ ] **Step 1: Write `docs/rgbd-experiment.md`**

```markdown
# RGB vs RGBD experiment (synthetic depth)

Prereqs: `scripts/simulate_depth.py` (this plan) and `--rgbd`
(docs/superpowers/plans/2026-07-29-rgbd-input.md) are implemented.

## Data

    python scripts/simulate_depth.py --dataset <DS> --seed 1
    # leakage-control copy: same images, depth from the WRONG mask
    cp -R <DS> <DS>-mismatch && rm -rf <DS>-mismatch/depth
    python scripts/simulate_depth.py --dataset <DS>-mismatch --seed 1 --mismatch

## Runs (identical seed => identical train/val split; only the input differs)

    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgb \
        --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS> --out-dir output/exp-rgbd --rgbd \
        --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps
    python -m training.run_train --data-dir <DS>-mismatch --out-dir output/exp-rgbd-mismatch \
        --rgbd --arch deeplabv3plus --encoder resnet34 --seed 1 --epochs 50 --device mps

## Reading the result

Compare val F1/IoU across the three runs:

| Outcome | Interpretation |
|---|---|
| rgbd > rgb, mismatch ~ rgb | depth is genuinely fused (what we hope for) |
| rgbd > rgb, mismatch > rgb | gain is label leakage through synthetic depth |
| rgbd ~ rgb | net ignores depth; check channel-4 first-conv weights |

Caveat: synthetic depth is derived from the masks, so even with dropout and
distractors the rgbd score is optimistically biased. This experiment validates
plumbing and fusion behavior, NOT real-sensor gains.
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the "Training / tooling entry points" code block, add one line after the `build_dataset.py` line:

```bash
python scripts/simulate_depth.py --dataset <DS>    # synth depth{N}.png from masks (RGBD)
```

- [ ] **Step 3: Update `docs/FEATURES.md`**

Annotate the "Training data simulator" backlog entry: depth simulation is implemented (`scripts/simulate_depth.py`, spec `docs/superpowers/specs/2026-07-29-depth-simulator-design.md`); RGB tile simulation remains open.

- [ ] **Step 4: Run the full suite to confirm nothing regressed**

Run: `pytest tests/test_simulate_depth.py tests/test_build_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/rgbd-experiment.md CLAUDE.md docs/FEATURES.md
git commit -m "docs: RGB-vs-RGBD experiment runbook + simulate_depth tooling entry"
```

# Optional wandb Logging — Design Spec

**Date:** 2026-07-09
**Status:** Approved (brainstorming complete)
**Repo:** `WINMOL_segmentor_pt` (`training/` package)

## 1. Goal

Add optional Weights & Biases scalar logging to the single-stage trainer, alongside the
existing TensorBoard logging. Opt-in per run; metrics only (no model/artifact uploads).

## 2. Scope (locked with user)

- **Metrics only** — log scalars (`train/loss`, `val/loss`, `val/precision`, `val/recall`,
  `val/f1`) per epoch. **No** artifact/checkpoint/model uploads.
- **Off unless `--wandb`** is passed. Default training behavior is unchanged.
- **Optional dependency** — `wandb` + `python-dotenv` live in a new `[wandb]` extra;
  training runs without them installed. TensorBoard logging is always on (unchanged).
- API key loads from a `.env` file (`WANDB_API_KEY`) via `python-dotenv` at run start.

## 3. Components

- **`pyproject.toml`** — new extra `wandb = ["wandb>=0.16", "python-dotenv>=1.0"]`.
- **`TrainConfig`** (`training/config.py`) — add `wandb: bool = False`,
  `wandb_project: Optional[str] = None`, `wandb_run_name: Optional[str] = None`.
- **`training/run_logger.py`** — `RunLogger`:
  - `__init__(log_dir, use_wandb=False, project=None, run_name=None)`: always creates a
    TensorBoard `SummaryWriter(log_dir)`. If `use_wandb`: import `wandb` + `dotenv` (raise a
    clear `RuntimeError("pip install '.[wandb]'")` if missing), `load_dotenv()`, then
    `wandb.init(project=project, name=run_name)`.
  - `log_scalars(scalars: dict, step: int)`: write each to TensorBoard; if wandb active,
    `wandb.log(scalars, step=step)`.
  - `close()`: `writer.close()`; if wandb active, `wandb.finish()`.
- **`training/train.py`** — replace the raw `SummaryWriter` with `RunLogger` built from
  `cfg`; accumulate **mean train loss per epoch** and log `train/loss`, `val/loss`,
  `val/precision`, `val/recall`, `val/f1` via `logger.log_scalars`.
- **`training/run_train.py`** — CLI flags `--wandb` (store_true), `--wandb-project`,
  `--wandb-run-name`, mapped into `TrainConfig`.

## 4. Data flow

`run_train.main()` parses `--wandb*` → `TrainConfig` → `train_one_run` builds
`RunLogger(cfg.log_dir, cfg.wandb, cfg.wandb_project, cfg.wandb_run_name)` → per epoch
`logger.log_scalars({...}, epoch)` → TensorBoard (+ wandb if on) → `logger.close()`.

## 5. Error handling

- `use_wandb=True` and `wandb` or `python-dotenv` not importable → raise
  `RuntimeError` naming the fix (`pip install '.[wandb]'`). Explicit opt-in means fail
  loud, not silent-skip.
- `use_wandb=False` → never import wandb/dotenv; pure TensorBoard, current behavior.

## 6. Testing (hermetic — no network, no real wandb account)

1. `RunLogger(use_wandb=False)` logs to TensorBoard only; a subsequent read of the event
   files shows the scalars; `wandb` is never imported.
2. `RunLogger(use_wandb=True)` with a **monkeypatched fake `wandb` module** (via
   `sys.modules`/`monkeypatch`): asserts `init(project, name)`, `log(scalars, step)`, and
   `finish()` are called with expected arguments. No real network.
3. `use_wandb=True` with wandb absent (monkeypatch import to fail) → raises `RuntimeError`
   mentioning the install extra.
4. CLI: `--wandb --wandb-project P --wandb-run-name R` → `cfg.wandb is True`,
   `cfg.wandb_project == "P"`, `cfg.wandb_run_name == "R"`; without `--wandb` → `False`.

## 7. Out of scope (YAGNI)

- Artifact/model/checkpoint uploads to wandb.
- wandb sweeps / hyperparameter search.
- Auto-enabling from env vars (must pass `--wandb`).
- Config-file parsing beyond the existing `TrainConfig` + CLI (FEATURES mentions "config
  file"; the dataclass + CLI already serve that; a file loader is a separate feature).

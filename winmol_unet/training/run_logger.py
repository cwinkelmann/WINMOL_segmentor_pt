"""Scalar logging to TensorBoard, plus optional Weights & Biases (opt-in, metrics only).

wandb is imported lazily and only when use_wandb=True, so it stays an optional dependency.
SummaryWriter is imported lazily for a different reason: see RunLogger.__init__.
"""


class RunLogger:
    def __init__(self, log_dir, use_wandb=False, project=None, run_name=None):
        # Imported here, not at module level. torch.utils.tensorboard pulls in the
        # `tensorboard` package, which loads TensorFlow when TF is installed in the same
        # environment. Doing that at import time, after torch's native libraries are
        # already loaded, deadlocks inside TF's abseil mutex ("RAW: Lock blocking") — so
        # merely importing winmol_unet.training.run_train hangs forever on a machine that has both.
        # That made the whole test suite unrunnable there while CI (no TF) stayed green.
        # The same reason run_train.py imports winmol_unet.export_keras lazily.
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)
        self._log_dir = log_dir
        # Every scalar dict that comes through log_scalars, kept so a finished run
        # leaves its numbers on disk as data -- re-plottable, diffable between runs,
        # and readable without a TensorBoard install.
        self._history = []
        self._wandb = None
        if use_wandb:
            try:
                try:
                    import wandb
                    from dotenv import load_dotenv
                except ImportError as e:
                    raise RuntimeError(
                        "wandb logging requires the optional extra: "
                        "pip install '.[wandb]'") from e
                load_dotenv()                  # picks up WANDB_API_KEY from .env
                wandb.init(project=project, name=run_name)
                self._wandb = wandb
            except Exception:
                self.writer.close()            # don't leak the TB writer if wandb setup fails
                raise

    def log_scalars(self, scalars, step):
        self._history.append({"step": step, **{k: float(v) for k, v in scalars.items()}})
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        if self._wandb is not None:
            self._wandb.log(dict(scalars), step=step)

    def log_figure(self, tag, fig, step):
        """A matplotlib figure to TB and (if enabled) wandb, then closed."""
        self.writer.add_figure(tag, fig, step)
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(fig)}, step=step)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def close(self):
        self._write_history()
        self.writer.close()
        if self._wandb is not None:
            self._wandb.finish()

    def _write_history(self):
        """Write the accumulated per-epoch scalars beside the TensorBoard events.

        Never fatal: a run that trained successfully must not fail at teardown
        because its metrics file could not be written.
        """
        if not self._history:
            return
        import json
        import os
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(os.path.join(self._log_dir, "metrics_history.json"), "w") as fh:
                json.dump(self._history, fh, indent=2)
        except OSError:
            pass

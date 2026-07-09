"""Scalar logging to TensorBoard, plus optional Weights & Biases (opt-in, metrics only).

wandb is imported lazily and only when use_wandb=True, so it stays an optional dependency.
"""
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, log_dir, use_wandb=False, project=None, run_name=None):
        self.writer = SummaryWriter(log_dir)
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
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        if self._wandb is not None:
            self._wandb.log(dict(scalars), step=step)

    def close(self):
        self.writer.close()
        if self._wandb is not None:
            self._wandb.finish()

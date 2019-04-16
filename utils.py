"""
From https://github.com/BloodAxe/pytorch-toolbelt/blob/feature/examples/pytorch_toolbelt/utils/catalyst_utils.py
"""
import torch
from catalyst.dl.callbacks import Callback, RunnerState, TensorboardLogger
from tensorboardX import SummaryWriter


def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
    for logger in state.loggers:
        if isinstance(logger, TensorboardLogger):
            return logger.loggers[state.loader_name]
    raise RuntimeError(f"Cannot find Tensorboard logger for loader {state.loader_name}")


class EpochJaccardMetric(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "jaccard"):
        """

        Args:
            input_key: input key to use for precision calculation; specifies our `y_true`.
            output_key: output key to use for precision calculation; specifies our `y_pred`.
            prefix:
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.intersection = 0
        self.union = 0

    def on_loader_start(self, state):
        self.intersection = 0
        self.union = 0

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = (outputs > 0).float()

        intersection = float(torch.sum(targets * outputs))
        union = float(torch.sum(targets) + torch.sum(outputs))
        self.intersection += intersection
        self.union += union

    def on_loader_end(self, state):
        metric_name = self.prefix
        eps = 1e-7
        metric = (self.intersection + eps) / (self.union - self.intersection + eps)
        state.metrics.epoch_values[state.loader_name][metric_name] = metric

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f"{self.prefix}/epoch", metric, global_step=state.epoch)

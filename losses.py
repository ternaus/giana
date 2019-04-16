from torch import nn
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class LossBinary(_Loss):
    """
    Loss defined as (1 - \alpha) BCE + \alpha \alpha (1 - SoftJaccard)
    """

    def __init__(self, jaccard_weight=0):
        super(LossBinary, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def forward(self, outputs: Tensor, targets:Tensor) -> Tensor:
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss += self.jaccard_weight * (1 - (intersection + eps) / (union - intersection + eps))
        return loss

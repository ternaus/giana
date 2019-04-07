from torch import nn
from torch.nn import functional as F


class LossBinary:
    """
    Loss defined as (1 - \alpha) BCE + \alpha \alpha (1 - SoftJaccard)
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss += self.jaccard_weight * (1 - (intersection + eps) / (union - intersection + eps))
        return loss

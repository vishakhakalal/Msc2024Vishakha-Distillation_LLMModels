import torch
from torch import Tensor
import torch.nn.functional as F
import BaseLoss

residual = lambda x: x[:, 0].unsqueeze(1) - x[:, 1:]


class KL_DivergenceLoss(BaseLoss):
    """KL Divergence loss"""

    def __init__(self, reduction='batchmean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction=self.reduction)

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        return self.kl_div(F.log_softmax(pred / self.temperature, dim=1), F.softmax(labels / self.temperature, dim=1))


class ContrastiveLoss(BaseLoss):
    """Contrastive loss with log_softmax and negative log likelihood."""

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor = None) -> Tensor:
        softmax_scores = F.log_softmax(pred / self.temperature, dim=1)
        labels = labels.argmax(dim=1) if labels is not None else torch.zeros(pred.size(0), dtype=torch.long,
                                                                             device=pred.device)
        return F.nll_loss(softmax_scores, labels, reduction=self.reduction)


LOSSES = {
    'kl_div': KL_DivergenceLoss,
    'contrastive': ContrastiveLoss,
}

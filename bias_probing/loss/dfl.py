import torch
from torch import nn
from torch.nn.functional import one_hot

from .core import EnsembleLoss


# Focal loss's implementation is adapted from
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
class DebiasedFocalLoss(EnsembleLoss):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"

    def forward(self, inputs, expert_inputs, target):
        """Compute the Debiased Focal Loss, given the main model and expert inputs, and target labels.

        :param inputs: Main model raw logits, of size (N, C)
        :param expert_inputs: Expert model raw logits, of size (N, C)
        :param target: Target labels, of size (N,), where each target is in the range [0, ..., C-1]
        :return: If reduction is 'mean' or 'sum', return a scalar. Otherwise, return a per-sample loss tensor
        of size (N,).
        """
        if self.training:
            prob: torch.Tensor = self.softmax(inputs) + self.eps  # (N, C)
            expert_prob: torch.Tensor = self.softmax(expert_inputs) + self.eps  # (N, C)

            # (N, C)
            one_hot_targets = torch.eye(inputs.size(1), device=inputs.device)[target]

            # (N, C)
            batch_loss = -self.alpha * (torch.pow(-expert_prob + 1., self.gamma) * torch.log(prob))
            # (N, C) => (N,)
            batch_loss = (one_hot_targets * batch_loss).sum(dim=1)
            if self.reduction == 'mean':
                loss = batch_loss.mean()
            elif self.reduction == 'sum':
                loss = batch_loss.sum()
            elif self.reduction == 'none':
                loss = batch_loss
            else:
                raise NotImplementedError(f'Invalid reduction mode: {self.reduction}')
            return loss, inputs

        return self.ce_loss(inputs, target), inputs

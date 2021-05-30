from abc import ABC, abstractmethod
from typing import Any
from torch import nn


class EnsembleLoss(ABC, nn.Module):
    @abstractmethod
    def forward(self, inputs, expert_input, target):
        raise NotImplementedError()

    def _forward_unimplemented(self, *inp: Any) -> None:
        pass


class DistillLoss(ABC, nn.Module):
    """Torch classification debiasing loss function"""

    @abstractmethod
    def forward(self, hidden, logits, bias, teach_probs, labels):
        """
        :param hidden: (N, H) hidden features from the model
        :param logits: (N, C) logit score for each class
        :param bias: (N, C) log-probabilities from the bias model for each class
        :param teach_probs: (N, C) log-probabilities from the teacher model, for each class
        :param labels: (N,) integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

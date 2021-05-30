from torch import nn
from .core import EnsembleLoss


class ProductOfExpertsLoss(EnsembleLoss):
    """Implements the Product of Experts loss (PoE) for a single expert (teacher) and main model"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def forward(self, inputs, expert_input, target):
        """Computes the PoE cross-entropy loss for a batch size N and number of classes C.
        During training, we use the expert_input for the PoE loss. During evaluation, the expert logits are
        ignored and standard cross-entropy loss is used

        :param inputs: The main model raw outputs (before applying Softmax), of shape (N, C)
        :param expert_input: The expert model raw outputs (before applying Softmax), of shape (N, C)
        :param target: Target labels, of shape (N,)
        :return:
            loss: The calculated loss tensor
            inputs: The normalized logits. If this is called during evaluation, the original input logits are returned
        """
        orig_loss = self.ce_loss(inputs, target)
        if self.training:
            assert expert_input is not None, "During training, expert_input must not be None"
            poe_logits = self.log_softmax(inputs) + self.log_softmax(expert_input)
            return self.ce_loss(
                poe_logits,
                target
            ) + self.alpha * orig_loss, poe_logits
        else:
            # During evaluation, we use standard cross-entropy for the main model only
            return orig_loss, inputs

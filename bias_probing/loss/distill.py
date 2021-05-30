import torch
from torch.nn.functional import cross_entropy
from .core import DistillLoss


class SmoothedDistillLoss(DistillLoss):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _beta_weights(self, weak_logits, labels, softmax_bias):
        # (N, C)
        bias = self.softmax(weak_logits) if softmax_bias else weak_logits
        # (C,C)[N] => (N, C)
        one_hot_labels = torch.eye(weak_logits.size(1)).to(weak_logits.device)[labels]
        # (N,)
        betas = (one_hot_labels * bias).sum(1)
        return (1 - betas).unsqueeze(1).expand_as(weak_logits)

    def forward(self, hidden, logits, weak_logits, teacher_logits, labels, softmax_teacher=True, softmax_bias=True):
        if not self.training:
            loss = cross_entropy(logits, labels)
            return loss
        # assert logits.size() == weak_logits.size() == teacher_logits.size()

        # (N, C)
        teacher_probs = self.softmax(teacher_logits) if softmax_teacher else teacher_logits
        # (N, C)
        weights = self._beta_weights(weak_logits, labels, softmax_bias)
        exp_teacher_probs = torch.pow(teacher_probs, weights)
        # (N, C) => (N,) => (N, 1) => (N, C)
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        loss = -(norm_teacher_probs * self.log_softmax(logits)).sum(1).mean()
        return loss


class SmoothedReweightLoss(DistillLoss):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, hidden, logits, weak_logits, teacher_logits, labels):
        if not self.training:
            loss = cross_entropy(logits, labels)
            return loss

        teacher_probs = self.softmax(teacher_logits)
        one_hot_labels = torch.eye(weak_logits.size(1)).to(weak_logits.device)[labels]
        bias = self.softmax(weak_logits)
        weights = (1 - (one_hot_labels * bias).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)
        scaled_weights = (one_hot_labels * norm_teacher_probs).sum(1)

        loss = cross_entropy(logits, labels, reduction='none')

        return (scaled_weights * loss).sum() / scaled_weights.sum()

# class DebiasedFocalDistillLoss(DistillLoss):
#     def __init__(self, gamma=2.0):
#         super().__init__()
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.log_softmax = torch.nn.LogSoftmax(dim=-1)
#         self.gamma = gamma
#
#     def __repr__(self):
#         return f'{self.__class__.__name__}()'
#
#     def forward(self, hidden, logits, weak_logits, teacher_logits, labels, output_teacher_probs=False):
#         if not self.training:
#             loss = cross_entropy(logits, labels)
#             if output_teacher_probs:
#                 return loss, self.softmax(teacher_logits)
#             return loss
#         assert logits.size() == weak_logits.size() == teacher_logits.size()
#         return None


class BiasProductBaseline(DistillLoss):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, hidden, logits, weak_logits, teacher_logits, labels):
        if not self.training:
            # weak_logits == teacher_logits == None
            loss = cross_entropy(logits, labels)
            return loss
        logits = logits.float()  # In case we were in fp16 mode
        return self.ce_loss(self.log_softmax(logits) + self.log_softmax(weak_logits), labels)

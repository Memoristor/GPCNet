# coding=utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    "CrossEntropy",
    "CrossEntropySoftTarget",
]


class CrossEntropy(nn.CrossEntropyLoss):
    """
    Cross Entropy loss for efficient callbacks.
    """

    def forward(self, preds, target):
        assert len(preds.shape) == 2 and len(target.shape) == 1
        return super().forward(input=preds, target=target.long())


class CrossEntropySoftTarget(nn.Module):
    """
    Cross entropy that accepts soft targets

    Params:
         size_average: if false, sum is returned instead of mean

    Forward:
         preds: predictions for neural network
         target: target, which can be soft

    Examples:
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = CrossEntropySoftTarget()(input, target)
        loss.backward()
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, preds, target):
        if self.size_average:
            return torch.mean(torch.sum(-target * self.logsoftmax(preds), dim=1))
        else:
            return torch.sum(torch.sum(-target * self.logsoftmax(preds), dim=1))


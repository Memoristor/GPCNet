# coding=utf-8

import torch
from torch import distributed as dist

from .pruner import Pruner

__all__ = [
    "RandomUniform",
    "RandomGaussian",
]


class RandomUniform(Pruner):
    """
    Randomized uniform distribution with values ranging from 0 to 1
    """

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        for s in self.scores.values():
            s.copy_(torch.rand_like(s))
            if dist.get_world_size() > 1:
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                s.div_(float(dist.get_world_size()))


class RandomGaussian(Pruner):
    """
    Randomized normal distribution with mean 0 and variance 1
    """

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        for s in self.scores.values():
            s.copy_(torch.randn_like(s))
            if dist.get_world_size() > 1:
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                s.div_(float(dist.get_world_size()))

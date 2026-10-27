# coding=utf-8

import torch
from torch import distributed as dist
from tqdm import tqdm

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "SNIP",
]


class SNIP(Pruner):
    """
    SNIP: Single-shot Network Pruning based on Connection Sensitivity

    SNIP introduce a saliency criterion based on connection sensitivity that identifies structurally important
    connections in the network for the given task.

    Reference:
    https://arxiv.org/abs/1810.02340
    """

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        self._model.zero_grad()

        # Synchronizes all processes
        dist.barrier()

        # Inference once
        bar = tqdm(dataloader)
        for iter, truth in enumerate(bar):
            for k, v in truth.items():
                truth[k] = v.to(torch.device(local_rank()))

            pred = predict_func(self._model, truth)
            losses = loss_func(pred, truth)
            L = torch.sum(torch.stack(list(losses.values())))

            bar.set_description(f"[Prune] Rank: {local_rank()}, Loss: {L:5.5f}")
            L.backward()  # all reduce

        # Calculate saliency score
        params = named_params(self._model)
        for name, score in self.scores.items():
            param = params[name]
            score.copy_(torch.clone(param.grad * param).detach().abs_())

        # Normalization
        sum_score = torch.cat([torch.flatten(v) for v in self.scores.values()]).sum()
        for score in self.scores.values():
            score.div_(sum_score)

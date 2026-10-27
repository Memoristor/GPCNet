# coding=utf-8

import torch
from torch import distributed as dist
from tqdm import tqdm

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "GraSP",
]


class GraSP(Pruner):
    """
    Gradient Signal Preservation (GraSP).
    Picking Winning Tickets Before Training by Preserving Gradient Flow

    GraSP aims to prune networks at initialization, thereby saving resources at training time
    as well. Specifically, authors argue that efficient training requires preserving the gradient
    flow through the network. This leads to a simple but effective pruning criterion we term
    Gradient Signal Preservation.

    Reference:
    https://openreview.net/forum?id=SkgsACVKPH
    """

    def __init__(self, param_names: list, verbose=False, logit_tensor_key="logit"):
        super().__init__(param_names, verbose)

        self.logit_tensor_key = logit_tensor_key
        self.temp = 200
        self.eps = 1e-10

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        # Get named parameters
        params = named_params(self._model)

        # Computing stopped gradient without computational graph
        stopped_grads = 0
        bar = tqdm(dataloader)
        for iter, truth in enumerate(bar):
            for k, v in truth.items():
                truth[k] = v.to(torch.device(local_rank()))

            pred = predict_func(self._model, truth)
            pred[self.logit_tensor_key] /= self.temp
            losses = loss_func(pred, truth)
            L = torch.sum(torch.stack(list(losses.values())))

            bar.set_description(f"[Prune] Computing stopped gradient, Rank: {local_rank}, Loss: {L:5.5f}")

            grads = torch.autograd.grad(L, list(params.values()), create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # Reduces the stopped grads across all machines
        dist.all_reduce(stopped_grads, op=dist.ReduceOp.SUM)

        # Computing hessian vector product with computational graph
        hvps = dict([(name, torch.zeros_like(param).detach()) for name, param in params.items()])
        bar = tqdm(dataloader)
        for iter, truth in enumerate(bar):
            for k, v in truth.items():
                truth[k] = v.to(torch.device(local_rank()))

            pred = predict_func(self._model, truth)
            pred[self.logit_tensor_key] /= self.temp
            losses = loss_func(pred, truth)
            L = torch.sum(torch.stack(list(losses.values())))

            bar.set_description(f"[Prune] Computing hessian vector product, Rank: {local_rank}, Loss: {L:5.5f}")

            grads = torch.autograd.grad(L, list(params.values()), create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm_grads = torch.autograd.grad(gnorm, list(params.values()), create_graph=False)
            for hvp, g in zip(hvps.values(), gnorm_grads):
                hvp.add_(g)

        # Reduces the stopped grads across all machines
        for hvp in hvps.values():
            dist.all_reduce(hvp, op=dist.ReduceOp.SUM)

        # Calculate score Hg * theta (negate to remove top percent)
        for name, score in self.scores.items():
            hvp = hvps[name]
            param = params[name]
            score.copy_(torch.clone(hvp * param.data).detach())

        # Normalization
        sum_score = torch.cat([torch.flatten(v) for v in self.scores.values()]).sum() + self.eps
        for score in self.scores.values():
            score.div_(sum_score)

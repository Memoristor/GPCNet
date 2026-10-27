# coding=utf-8

import torch

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "SynFlowL2",
]


class SynFlowL2(Pruner):
    """
    SynFlow-L2 described in `A Unified Paths Perspective for Pruning at Initialization`

    Reference:
    http://arxiv.org/abs/2101.10552
    """

    def __init__(self, param_names: list, verbose=False, input_tensor_key="image"):
        super().__init__(param_names, verbose)

        self.input_tensor_key = input_tensor_key

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        self._model.eval()  # disable runing parameters
        self._model.zero_grad()

        @torch.no_grad()
        def power2(model):
            # model.double()
            signs = {}
            for name, param in named_params(model).items():
                signs[name] = torch.sign(param)
                param.mul_(param)
            return signs

        @torch.no_grad()
        def depower2(model, signs):
            # model.float()
            for name, param in named_params(model).items():
                param.sqrt_().mul_(signs[name])

        signs = power2(self._model)

        # Model pruning
        truth = next(iter(dataloader))
        input_dim = list(truth[self.input_tensor_key][0, :].shape)
        input = torch.ones([1] + input_dim).to(torch.device(local_rank()))  # , dtype=torch.float64).to(device)
        output = self._model(input)
        torch.sum(output).backward()

        # Calculate saliency score
        params = named_params(self._model)
        for name, score in self.scores.items():
            param = params[name]
            score.copy_(torch.clone(param.grad * param.sqrt()).detach().abs_())

        depower2(self._model, signs)

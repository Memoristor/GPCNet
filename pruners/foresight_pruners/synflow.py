# coding=utf-8

import torch

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "SynFlow",
]


class SynFlow(Pruner):
    """
    Iterative Synaptic Flow Pruning (SynFlow).
    Pruning neural networks without any data by iteratively conserving synaptic flow

    SynFlow can identify highly sparse trainable subnetworks at initialization, without
    ever training, or indeed without ever looking at the data

    Reference:
    https://proceedings.neurips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html
    """

    def __init__(self, param_names: list, verbose=False, input_tensor_key="image"):
        super().__init__(param_names, verbose)

        self.input_tensor_key = input_tensor_key

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        self._model.eval()  # disable runing parameters
        self._model.zero_grad()

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in named_params(model).items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in named_params(model).items():
                param.mul_(signs[name])

        signs = linearize(self._model)

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
            score.copy_(torch.clone(param.grad * param).detach().abs_())

        nonlinearize(self._model, signs)

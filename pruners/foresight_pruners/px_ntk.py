# coding=utf-8

import copy

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = ["PX"]


class PX(Pruner):
    """Path eXclusion (PX), a foresight pruning method designed to preserve
    the parameters that mostly influence the NTK's trace. PX is able to find
    lottery tickets (i.e. good paths) even at high sparsity levels and largely
    reduces the need for additional training.

    Reference:
    [1] Finding Lottery Tickets in Vision Models via Data-driven Spectral
        Foresight Pruning. CVPR 2024.

    Note: Original code is modified to apply our pruning and training pipeline.
    [2] https://github.com/iurada/px-ntk-pruning
    """

    def __init__(
        self,
        param_names: list,
        verbose=False,
        input_tensor_key="image",
        logit_tensor_key="logit",
    ):
        super(PX, self).__init__(param_names, verbose)

        self.input_tensor_key = input_tensor_key
        self.logit_tensor_key = logit_tensor_key

        self.orig_relu = F.relu
        self.orig_leakyrelu = F.leaky_relu

    def copy_ddp_model(self, model: DistributedDataParallel):
        """Copy model in distributed data parallel"""
        return DistributedDataParallel(
            copy.deepcopy(model.module),
            device_ids=[local_rank()],
            output_device=local_rank(),
            # find_unused_parameters=True
        )

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        self._model.zero_grad()

        def hook_activation(input, inplace=False):  # overrides F.relu
            self.activation_maps.append(input.clone().detach())
            return self.orig_relu(input, inplace)

        def apply_activation(input, inplace=False):  # overrides F.relu
            map = torch.where(self.activation_maps.pop(0) > 0, 1.0, 0.0)
            return input * map

        # Path Kernel Estimator
        pk_model = self.copy_ddp_model(self._model)
        pk_model.train()
        pk_model.zero_grad()

        with torch.no_grad():
            for name, param in named_params(pk_model).items():
                if name in self.param_names:
                    param.pow_(2)

        # Path Activation Matrix Estimator
        jvf_model = self.copy_ddp_model(self._model)
        jvf_model.train()
        jvf_model.zero_grad()

        with torch.no_grad():
            for name, param in named_params(jvf_model).items():
                if name in self.param_names:
                    param.fill_(1.0)

        # Auxiliary Activation Maps Extractor
        activation_model = self.copy_ddp_model(self._model)
        activation_model.eval()
        activation_model.zero_grad()

        bar = tqdm(dataloader)
        for iter, truth in enumerate(bar):
            for k, v in truth.items():
                truth[k] = v.to(torch.device(local_rank()))

            with torch.no_grad():
                F.relu = hook_activation
                F.leaky_relu = hook_activation
                self.activation_maps = []
                predict_func(activation_model, truth)

                F.relu = apply_activation
                input_tensor = truth[self.input_tensor_key]
                input_tensor = input_tensor.pow(2)
                truth[self.input_tensor_key] = input_tensor
                pred = predict_func(jvf_model, truth)
                z1 = pred[self.logit_tensor_key]

            bar.set_description(f"[{self.__class__.__name__}] Rank: {local_rank()}")

            F.relu = lambda input, *args, **kwargs: input
            F.leaky_relu = lambda input, *args, **kwargs: input
            truth[self.input_tensor_key] = torch.ones_like(input_tensor)
            pred = predict_func(pk_model, truth)
            z2 = pred[self.logit_tensor_key]

            (z1 * z2).sum().backward()

            F.relu = self.orig_relu
            F.leaky_relu = self.orig_leakyrelu

        with torch.no_grad():
            org_params = named_params(self._model)
            pk_params = named_params(pk_model)
            jvf_params = named_params(jvf_model)
            for name, score in self.scores.items():
                p = org_params[name]
                p1 = pk_params[name]
                p2 = jvf_params[name]
                if p1.grad is None:
                    score.copy_(torch.zeros_like(p))
                else:
                    s = p1.grad.abs() * p.pow(2)
                    if dist.get_world_size() > 1:
                        dist.all_reduce(s, op=dist.ReduceOp.SUM)
                        s /= dist.get_world_size()
                    score.copy_(torch.clone(s))

        # delete auxiliary models
        del pk_model
        del jvf_model
        del activation_model

# coding=utf-8

import torch
from torch import distributed as dist

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "RandERK",
]


class RandERK(Pruner):
    """
    Random ERK

    ERK enables the number of connections in a sparse layer to scale with the sum of
    the number of output and input channels.

    Reference:
    [1] Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    [2] Rigging the Lottery: Making All Tickets Winners
        https://proceedings.mlr.press/v119/evci20a.html
    """

    def __init__(self, param_names: list, verbose=False, power_scale=1.0):
        super().__init__(param_names, verbose)
        self.power_scale = power_scale

    def calc_score(self, dataloader, predict_func, loss_func):
        pass

    def layerwise_density(self, density):
        """Calculate layerwise density base on ERK."""
        params_dict = named_params(self._model)
        global_params = torch.cat([torch.flatten(v) for v in params_dict.values()])
        target_ones = int(density * global_params.numel())
        mask_probs = {n: (sum(m.size()) / m.numel()) ** self.power_scale for n, m in self.masks.items()}
        params_density = dict()
        # Find parameters with full density
        params_with_full_density = -1
        while params_with_full_density != len(params_density.keys()):
            params_with_full_density = len(params_density.keys())
            current_ones = 0
            for n, p in params_dict.items():
                if n not in self.masks.keys():  # Pruning is disabled
                    current_ones += p.numel()
                elif n in params_density.keys():  # Pruning is enabled
                    d = params_density[n]
                    current_ones += int(d * p.numel())
            prob_ones = sum(
                [int(mask_probs[n] * m.numel()) for n, m in self.masks.items() if n not in params_density.keys()]
            )
            ratio = (target_ones - current_ones) / prob_ones
            for n, m in self.masks.items():
                if n not in params_density.keys():
                    expected_ones = int(ratio * mask_probs[n] * m.numel())
                    if expected_ones >= m.numel():
                        params_density[n] = 1.0
                        if self.verbose:
                            print(f"The density of {n} has been set to 1.")
        # Alloc other parameters
        current_ones = 0
        for n, p in params_dict.items():
            if n not in self.masks.keys():  # Pruning is disabled
                current_ones += p.numel()
            elif n in params_density.keys():  # Pruning is enabled
                d = params_density[n]
                current_ones += int(d * p.numel())
        prob_ones = sum(
            [int(mask_probs[n] * m.numel()) for n, m in self.masks.items() if n not in params_density.keys()]
        )
        ratio = (target_ones - current_ones) / prob_ones
        for n, m in self.masks.items():
            if n not in params_density.keys():
                expected_ones = int(ratio * mask_probs[n] * m.numel())
                if expected_ones <= m.numel():
                    params_density[n] = expected_ones / m.numel()
                else:
                    raise AttributeError(f"The density of {n} can not be set to 1.")
        # Check parameters density
        for n in self.masks.keys():
            if n not in params_density.keys():
                raise AttributeError(f"The density of {n} has not been assigned.")
        return params_density

    def local_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level parameter-wise."""
        raise NotImplementedError

    def global_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level globally."""
        raise NotImplementedError

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the model in the graph-base manner."""
        # Check parameters density globally
        params_dict = named_params(self._model)
        global_params = torch.cat([torch.flatten(v) for v in params_dict.values()])
        target_ones = int(density * global_params.numel())
        params_density = self.layerwise_density(density=density)
        current_ones = 0
        for n, p in params_dict.items():
            if n not in self.masks.keys():  # Pruning is disabled
                current_ones += p.numel()
            elif n in params_density.keys():  # Pruning is enabled
                d = params_density[n]
                current_ones += int(d * p.numel())
        if abs(target_ones - current_ones) / global_params.numel() > tolerance:
            raise AttributeError(
                f"Out of {global_params.numel()} parameters, {global_params.numel() - target_ones} parameters should be pruned, "
                f"but {global_params.numel() - current_ones} parameters has been pruned"
            )
        # Get masks
        for name, mask in self.masks.items():
            score = torch.rand_like(mask)
            if dist.get_world_size() > 1:
                dist.all_reduce(score, op=dist.ReduceOp.SUM)
                score.div_(dist.get_world_size())

            d = params_density[name]
            if d < 1.0:
                k = int((1.0 - d) * mask.numel())
                sorted, _ = torch.sort(score.flatten(), descending=False)
                threshold = sorted[k]
                zero = torch.tensor([0.0]).to(torch.device(local_rank()))
                one = torch.tensor([1.0]).to(torch.device(local_rank()))
                mask.copy_(torch.where(score <= threshold, zero, one))
            else:
                mask.copy_(torch.ones_like(mask).detach())

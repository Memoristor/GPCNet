# coding=utf-8

import torch

from tools import local_rank, named_params

from .rand_erk import RandERK

__all__ = [
    "MagnitudeERK",
]


class MagnitudeERK(RandERK):
    """
    Magnitude ERK

    ERK enables the number of connections in a sparse layer to scale with the sum of
    the number of output and input channels.

    Reference:
    [1] Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    [2] Rigging the Lottery: Making All Tickets Winners
        https://proceedings.mlr.press/v119/evci20a.html
    """

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
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
            score = torch.clone(params_dict[name]).detach()
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

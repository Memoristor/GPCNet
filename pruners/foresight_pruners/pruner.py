# coding=utf-8

import torch

from tools import local_rank, named_params

__all__ = [
    "Pruner",
]


class Pruner:
    """
    Basic module for model pruning

    Params:
        param_names: list. Prunable parameters names
        verbose: bool. Display detailed information during runtime or not
    """

    def __init__(self, param_names: list, verbose=False):
        self.param_names = param_names
        self.verbose = verbose
        self.scores = dict()
        self.masks = dict()
        self.is_initialized = False
        self._model = None

    def initialize(self, model):
        """
        Initialize `self.scores` and `self.masks` based on prunable parameters names

        Params:
            model: nn.Module. prunable model
        """
        self._model = model
        if not self.is_initialized:
            params = named_params(model)
            for k in self.param_names:
                self.scores[k] = torch.zeros_like(params[k], requires_grad=False)
                self.masks[k] = torch.ones_like(params[k], requires_grad=False)
            self.is_initialized = True

    def reset_scores(self):
        """Reset scores to zero."""
        assert self.is_initialized, "initialize the pruner first"
        for k in self.param_names:
            self.scores[k].zero_()

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        raise NotImplementedError

    def mask_density(self):
        """Calculate current density of the masks."""
        assert self.is_initialized, "initialize the pruner first"
        total_params = 0
        remaining_params = 0
        for name, mask in self.masks.items():
            total_params += mask.numel()
            remaining_params += mask.sum().item()
        return remaining_params / total_params

    def model_density(self, model=None):
        """Calculate current density of the model parameters."""
        assert self.is_initialized, "initialize the pruner first"
        params = named_params(self._model if model is None else model)
        total_params = 0
        remaining_params = 0
        for name in self.param_names:
            total_params += params[name].numel()
            remaining_params += (params[name] != 0).int().sum().item()
        return remaining_params / total_params

    def local_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level parameter-wise."""
        assert self.is_initialized, "initialize the pruner first"
        for name, score in self.scores.items():
            k = int((1.0 - density) * score.numel())
            if k > 0:
                local_scores = torch.flatten(score)
                num_nan = torch.isnan(local_scores).int().sum()
                assert num_nan == 0, f"There are {num_nan} nan value in local scores of {name}, pruning stopped."
                # threshold, _ = torch.kthvalue(local_scores, k)
                sorted, _ = torch.sort(local_scores, descending=False)
                threshold = sorted[k]
                num_pruned = (score <= threshold).int().sum()
                if (k - num_pruned).abs() / score.numel() > tolerance:
                    raise AttributeError(
                        f"Out of {score.numel()} parameters of {name}, {k} parameters should be pruned, "
                        f"but {num_pruned} parameters will be pruned"
                    )
                zero = torch.tensor([0.0]).to(torch.device(local_rank()))
                one = torch.tensor([1.0]).to(torch.device(local_rank()))
                self.masks[name].copy_(torch.where(score <= threshold, zero, one))
            else:
                self.masks[name].copy_(torch.ones_like(score).detach())

    def global_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level globally."""
        assert self.is_initialized, "initialize the pruner first"
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        num_nan = torch.isnan(global_scores).int().sum()
        assert num_nan == 0, f"There are {num_nan} nan value in global scores, pruning stopped."
        # global_params = torch.cat([torch.flatten(v) for v in named_params(self._model).values()])
        # k = int((1.0 - density) * global_params.numel())
        k = int((1.0 - density) * global_scores.numel())
        if k > 0:
            # threshold, _ = torch.kthvalue(global_scores, k)
            sorted, _ = torch.sort(global_scores, descending=False)
            threshold = sorted[k]
            num_pruned = (global_scores <= threshold).int().sum()
            if (k - num_pruned).abs() / global_scores.numel() > tolerance:
                raise AttributeError(
                    f"Out of {global_scores.numel()} parameters, {k} parameters should be pruned, "
                    f"but {num_pruned} parameters will be pruned"
                )
            for name, score in self.scores.items():
                zero = torch.tensor([0.0]).to(torch.device(local_rank()))
                one = torch.tensor([1.0]).to(torch.device(local_rank()))
                self.masks[name].copy_(torch.where(score <= threshold, zero, one))
        else:
            for name, score in self.scores.items():
                self.masks[name].copy_(torch.ones_like(score).detach())

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
        raise NotImplementedError

    @torch.no_grad()
    def apply_mask_on_params(self, model=None):
        """Apply sparse mask on parameters."""
        assert self.is_initialized, "initialize the pruner first"
        params = named_params(self._model if model is None else model)
        for k, v in self.masks.items():
            params[k].mul_(v)

    @torch.no_grad()
    def apply_mask_on_grads(self, model=None):
        """Apply sparse mask on gradients of parameters."""
        assert self.is_initialized, "initialize the pruner first"
        params = named_params(self._model if model is None else model)
        for k, v in self.masks.items():
            params[k].grad.mul_(v)

    def state_dict(self):
        """Returns the state of the sparse as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the model.
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def load_state_dict(self, state_dict):
        """Loads the sparses state.
        Args:
            state_dict (dict): sparse state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

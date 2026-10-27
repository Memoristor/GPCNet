# coding=utf-8

import copy

import torch
from torch import distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tools import hard_bn_momentum, local_rank, named_params, reset_bn_momentum, reset_bn_status, weight_init

from .pruner import Pruner

__all__ = [
    "NTKSAP",
]


class NTKSAP(Pruner):
    """
    NTK-SAP: Improving neural network pruning by aligning training dynamics

    Reference:
    https://openreview.net/forum?id=-5EWhW_4qWP
    https://github.com/YiteWang/NTK-SAP
    """

    def __init__(
        self,
        param_names: list,
        verbose=False,
        scoring_round=1,
        perturb_epsilon=1e-4,
        noise_input=False,
        reinit_weights=False,
        reinit_weights_method="kaiming_normal_init",
        input_tensor_key="image",
        logit_tensor_key="logit",
    ):
        super(NTKSAP, self).__init__(param_names, verbose)

        self.scoring_round = scoring_round
        self.perturb_epsilon = perturb_epsilon
        self.noise_input = noise_input
        self.reinit_weights = reinit_weights
        self.reinit_weights_method = reinit_weights_method
        self.input_tensor_key = input_tensor_key
        self.logit_tensor_key = logit_tensor_key

    @torch.no_grad()
    def initialize_weights(self, model):
        """initialize weights"""
        model.apply(getattr(weight_init, self.reinit_weights_method))
        if dist.get_world_size() > 1:
            for p in named_params(model).values():
                dist.all_reduce(p, op=dist.ReduceOp.SUM)
                p.data.div_(float(dist.get_world_size()))

    @torch.no_grad()
    def perturb(self, model_orig, model_copy):
        """Perturb copied model"""
        params_orig = named_params(model_orig)
        params_copy = named_params(model_copy)
        for n in self.param_names:
            p_orig = params_orig[n]
            p_copy = params_copy[n]
            # generate same random values across ranks
            delta = torch.randn_like(p_orig.data)
            if dist.get_world_size() > 1:
                dist.all_reduce(delta, op=dist.ReduceOp.SUM)
                delta.div_(float(dist.get_world_size()))
            # parameter perturbation
            p_copy.data.copy_(p_orig.data + self.perturb_epsilon * delta)

        for module, module_cpy in zip(model_orig.modules(), model_copy.modules()):
            if isinstance(module, _BatchNorm):
                module_cpy.running_mean.data.copy_(module.running_mean.data)
                module_cpy.running_var.data.copy_(module.running_var.data)
                module_cpy.num_batches_tracked.data.copy_(module.num_batches_tracked.data)

    def copy_ddp_model(self, model: DistributedDataParallel):
        """Copy model in distributed data parallel"""
        return DistributedDataParallel(
            copy.deepcopy(model.module),
            device_ids=[local_rank()],
            output_device=local_rank(),
            # find_unused_parameters=True
        )

    def set_params_grads(self, models, enable=True, detach=True):
        """Set parameters gradients"""
        if not isinstance(models, list):
            models = [models]
        for model in models:
            params = named_params(model)
            for p in params.values():
                if detach:
                    p.detach_()
                p.requires_grad = enable

    def set_masks_grads(self, enable=True, zero_grad=True):
        """Set mask gradients"""
        for m in self.masks.values():
            m.requires_grad = enable
            if m.grad is not None and zero_grad:
                m.grad.data.zero_()

    def apply_masks_on_detached_params(self, model):
        """Apply masks on corresponding detached paramters"""
        params = named_params(model)
        for n, m in self.masks.items():
            p = params[n]
            p.detach_().mul_(m)

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        # Copy a same model
        model_orig = self._model
        model_copy = self.copy_ddp_model(model_orig)

        # Enable mask grads, disable parameter grads
        self.set_params_grads(models=[model_orig, model_copy], enable=False, detach=True)
        self.set_masks_grads(enable=True, zero_grad=True)

        # Set hard momentum of BatchNorm
        hard_bn_momentum(model_orig)
        hard_bn_momentum(model_copy)

        # Synchronizes all processes
        dist.barrier()

        for _ in range(self.scoring_round):
            bar = tqdm(dataloader)
            for iter, truth in enumerate(bar):
                for k, v in truth.items():
                    if self.noise_input:
                        if k == self.input_tensor_key:  # noise input
                            v = torch.randn_like(v)
                    truth[k] = v.to(torch.device(local_rank()))

                if self.reinit_weights:
                    self.initialize_weights(model_orig)

                reset_bn_status(model_orig)

                # Init running parameters of BatchNorm
                with torch.no_grad():
                    model_orig.train()
                    self.apply_masks_on_detached_params(model_orig)
                    pred_orig = predict_func(model_orig, truth)

                # Compute the true graph using eval mode
                model_orig.eval()
                self.apply_masks_on_detached_params(model_orig)
                pred_orig = predict_func(model_orig, truth)
                pred_orig = pred_orig[self.logit_tensor_key]

                # Perturb copied model parameters
                self.perturb(model_orig, model_copy)

                model_copy.eval()
                self.apply_masks_on_detached_params(model_copy)
                pred_copy = predict_func(model_copy, truth)
                pred_copy = pred_copy[self.logit_tensor_key]

                jac_approx = (torch.norm(pred_orig - pred_copy, dim=-1) ** 2).sum()
                jac_approx.backward()

                bar.set_description(
                    f"[{self.__class__.__name__}] Computing Jaccard approximation, "
                    f"Rank: {local_rank}, "
                    f"Jacc: {jac_approx:5.5f}"
                )

        for k, s in self.scores.items():
            m = self.masks[k]
            g = m.grad
            g = g.contiguous()
            if dist.get_world_size() > 1:
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
            s.copy_(torch.clone(g * (m != 0)).detach().abs_())
            g.zero_()

        # Enable parameter grads, disable mask grads
        self.set_params_grads(models=model_orig, enable=True, detach=True)
        self.set_masks_grads(enable=False, zero_grad=True)

        # Reset momentum of BatchNorm2d
        reset_bn_momentum(model_orig)
        del model_copy

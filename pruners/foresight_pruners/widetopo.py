# coding=utf-8

import copy

import torch
from torch import distributed as dist
from torch.autograd import grad
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .pruner import Pruner
from tools import hard_bn_momentum, named_params, reset_bn_momentum, reset_bn_status, weight_init

__all__ = [
    "WideTopo",
]


class WideTopo(Pruner):
    """
    WideTopo: Improving Foresight Neural Network Pruning through Training Dynamics Preservation and Wide Topologies Exploration

    https://www.sciencedirect.com/science/article/pii/S0893608025010160
    """

    def __init__(
        self,
        param_names: list,
        verbose=False,
        noise_input=False,
        reinit_weights=False,
        reinit_weights_method="kaiming_normal_init",
        params_perturb_epsilon=0.0005,
        inputs_perturb_epsilon=0.0005,
        ntk_hutch_iters=1,
        ntk_score_factor=0.01,
        nad_score_factor=1.0,
        enable_wide_topologies=True,
        compression_factor=1.0,
        input_tensor_key="image",
        logit_tensor_key="logit",
    ):
        super(WideTopo, self).__init__(param_names, verbose)

        self.noise_input = noise_input
        self.reinit_weights = reinit_weights
        self.reinit_weights_method = reinit_weights_method
        self.params_perturb_epsilon = params_perturb_epsilon
        self.inputs_perturb_epsilon = inputs_perturb_epsilon
        self.ntk_hutch_iters = ntk_hutch_iters
        self.ntk_score_factor = ntk_score_factor
        self.nad_score_factor = nad_score_factor
        self.enable_wide_topologies = enable_wide_topologies
        self.compression_factor = compression_factor
        self.input_tensor_key = input_tensor_key
        self.logit_tensor_key = logit_tensor_key

        self.scale_factor = None

    @torch.no_grad()
    def initialize_weights(self, model):
        """initialize weights"""
        model.apply(getattr(weight_init, self.reinit_weights_method))
        if dist.get_world_size() > 1:
            for p in named_params(model).values():
                dist.all_reduce(p, op=dist.ReduceOp.SUM)
                p.data.div_(float(dist.get_world_size()))

    @torch.no_grad()
    def copy_parameters(self, src_model, tgt_model):
        """Copy parameters from the source model to target model"""
        src_params = named_params(src_model)
        tgt_params = named_params(tgt_model)
        for n in self.param_names:
            p_src = src_params[n]
            p_tgt = tgt_params[n]
            p_tgt.data.copy_(p_src.data)

        for module_src, module_tgt in zip(src_model.modules(), tgt_model.modules()):
            if isinstance(module_src, _BatchNorm):
                module_tgt.running_mean.data.copy_(module_src.running_mean.data)
                module_tgt.running_var.data.copy_(module_src.running_var.data)
                module_tgt.num_batches_tracked.data.copy_(module_src.num_batches_tracked.data)

    @torch.no_grad()
    def perturb_parameters(self, model):
        params = named_params(model)
        for n in self.param_names:
            p = params[n]
            # generate same random values across ranks
            delta = torch.randn_like(p.data)
            if dist.get_world_size() > 1:
                dist.all_reduce(delta, op=dist.ReduceOp.SUM)
                delta.div_(float(dist.get_world_size()))
            # parameter perturbation
            p.data.add_(self.params_perturb_epsilon * delta)

    def copy_ddp_model(self, model):
        """Copy model in distributed data parallel"""
        if isinstance(model, DistributedDataParallel):
            return DistributedDataParallel(
                copy.deepcopy(model.module),
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                # find_unused_parameters=True
            )
        else:
            raise AttributeError(f"The type of input model {type(model)} is not `DistributedDataParallel`")

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

    def calc_ntk_score(self, model, truth, predict_func):
        """Calculate the score of each parameter of unbiased estimation of NTK"""
        scores = dict()
        for k, v in self.scores.items():
            scores[k] = torch.zeros_like(v)

        reset_bn_status(model)

        # Init running parameters of BatchNorm
        with torch.no_grad():
            model.train()
            predict_func(model, truth)

        # Compute the true graph using eval mode
        ntk_estimation = 0
        params = named_params(model)
        for _ in range(self.ntk_hutch_iters):
            model.eval()
            pred = predict_func(model, truth)
            logit = pred[self.logit_tensor_key]

            coff = torch.randn_like(logit)
            dots = (coff * logit).sum()

            grads = grad(dots, list(params.values()), create_graph=True)
            grads_flatten = torch.cat([g.flatten() for g in grads])
            dist.all_reduce(grads_flatten, op=dist.ReduceOp.SUM)

            masks_flatten = torch.cat(
                [
                    self.masks[k].flatten() if k in self.masks.keys() else torch.ones_like(p).flatten()
                    for k, p in params.items()
                ]
            )

            grads_flatten = grads_flatten * masks_flatten
            trace = (grads_flatten * grads_flatten).sum()
            ntk_estimation += trace / self.ntk_hutch_iters

            grads = grad(trace, list(params.values()), create_graph=False, allow_unused=True)
            for g, (n, p) in zip(grads, params.items()):
                if n in self.masks.keys():
                    s = scores[n]
                    m = self.masks[n]
                    if g is not None:
                        g = g.contiguous()
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                        g = g * m
                        s.add_(torch.clone(g * p).detach())
                    else:
                        if self.verbose:
                            print(f"[{self.__class__.__name__}] param {n} is unused in second order derivate")

        for s in scores.values():
            s.abs_()

        return scores, ntk_estimation

    def calc_nad_score(
        self,
        osrc_model,
        psrc_model,
        otgt_model,
        ptgt_model,
        truth,
        truth_pert,
        predict_func,
    ):
        """Calculate the score of each parameter of nad"""
        scores = dict()
        for k, v in self.scores.items():
            scores[k] = torch.zeros_like(v)

        for k, m in self.masks.items():
            if m.grad is not None:
                m.grad.zero_()

        reset_bn_status(osrc_model)

        # Init running parameters of BatchNorm
        with torch.no_grad():
            osrc_model.train()
            self.apply_masks_on_detached_params(osrc_model)
            predict_func(osrc_model, truth)

        # Compute the true graph using eval mode
        osrc_model.eval()
        self.apply_masks_on_detached_params(osrc_model)
        osrc_logit = predict_func(osrc_model, truth)
        osrc_logit = osrc_logit[self.logit_tensor_key]

        self.copy_parameters(osrc_model, psrc_model)
        psrc_model.eval()
        self.apply_masks_on_detached_params(psrc_model)
        psrc_logit = predict_func(psrc_model, truth_pert)
        psrc_logit = psrc_logit[self.logit_tensor_key]

        self.copy_parameters(osrc_model, otgt_model)
        self.perturb_parameters(otgt_model)
        otgt_model.eval()
        self.apply_masks_on_detached_params(otgt_model)
        otgt_logit = predict_func(otgt_model, truth)
        otgt_logit = otgt_logit[self.logit_tensor_key]

        self.copy_parameters(otgt_model, ptgt_model)
        ptgt_model.eval()
        self.apply_masks_on_detached_params(ptgt_model)
        ptgt_logit = predict_func(ptgt_model, truth_pert)
        ptgt_logit = ptgt_logit[self.logit_tensor_key]

        delta = (ptgt_logit - otgt_logit) - (psrc_logit - osrc_logit)
        nad_approx = (torch.norm(delta, dim=-1) ** 2).sum()

        # if self.verbose:
        #     print(f'NAD approximation without scale factor: {nad_approx:4.4f}')

        nad_approx = nad_approx / (self.inputs_perturb_epsilon * self.params_perturb_epsilon)
        nad_approx.backward()

        for k, m in self.masks.items():
            g = m.grad
            g = g.contiguous()
            if dist.get_world_size() > 1:
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
            s = scores[k]
            s.copy_(torch.clone(g * (m != 0)).detach().abs_())
            g.zero_()

        return scores, nad_approx

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        # Copy model
        osrc_model = self._model
        psrc_model = self.copy_ddp_model(osrc_model)
        otgt_model = self.copy_ddp_model(osrc_model)
        ptgt_model = self.copy_ddp_model(osrc_model)

        # Set hard momentum of BatchNorm
        hard_bn_momentum(osrc_model)
        hard_bn_momentum(psrc_model)
        hard_bn_momentum(otgt_model)
        hard_bn_momentum(ptgt_model)

        # Reset scores
        global_ntk_scores = dict()
        global_nad_scores = dict()
        for k, v in self.scores.items():
            v.zero_()
            global_ntk_scores[k] = torch.zeros_like(v)
            global_nad_scores[k] = torch.zeros_like(v)

        # Synchronizes all processes
        dist.barrier()

        bar = tqdm(dataloader)
        for iter, truth in enumerate(bar):
            for k, v in truth.items():
                if self.noise_input:
                    if k == self.input_tensor_key:  # noise input
                        v = torch.randn_like(v)
                truth[k] = v.to(torch.device(dist.get_rank()))

            truth_pert = copy.deepcopy(truth)
            for k, v in truth_pert.items():
                if k == self.input_tensor_key:
                    v.add_(self.inputs_perturb_epsilon * torch.randn_like(v))

            if self.reinit_weights:
                self.initialize_weights(osrc_model)

            # Enable mask grads, disable parameter grads
            self.set_params_grads(models=osrc_model, enable=True, detach=True)
            ntk_scores, ntk_estimation = self.calc_ntk_score(osrc_model, truth, predict_func)

            # Enable mask grads, disable parameter grads
            self.set_params_grads(
                models=[osrc_model, psrc_model, otgt_model, ptgt_model],
                enable=False,
                detach=True,
            )
            self.set_masks_grads(enable=True, zero_grad=True)
            nad_scores, nad_approx = self.calc_nad_score(
                osrc_model,
                psrc_model,
                otgt_model,
                ptgt_model,
                truth,
                truth_pert,
                predict_func,
            )

            for k, s in global_ntk_scores.items():
                s.add_(torch.clone(ntk_scores[k].detach()))
            for k, s in global_nad_scores.items():
                s.add_(torch.clone(nad_scores[k].detach()))

            bar.set_description(
                f"[{self.__class__.__name__}] Computing scores, "
                f"Rank: {dist.get_rank()}, "
                f"NTK: {ntk_estimation:5.5f}, "
                f"NAD: {nad_approx:5.5f}"
            )

        # Normalize score
        global_ntk_scores_norm = torch.norm(torch.cat([s.flatten() for s in global_ntk_scores.values()]))
        global_nad_scores_norm = torch.norm(torch.cat([s.flatten() for s in global_nad_scores.values()]))
        current_scale_factor = global_ntk_scores_norm / global_nad_scores_norm

        if self.verbose:
            print(
                f"[{self.__class__.__name__}] The norm of `global_ntk_scores` and `global_nad_scores` are "
                f"{global_ntk_scores_norm:4.4f} and {global_nad_scores_norm:4.4f}, respectively. "
                f"Ratio between these norm is {current_scale_factor:4.4f}"
            )

        for k, s in self.scores.items():
            ntk_score = global_ntk_scores[k] * self.ntk_score_factor
            nad_score = global_nad_scores[k] * self.nad_score_factor
            s.copy_(torch.clone(ntk_score + nad_score).detach())

        # Wide topology
        if self.enable_wide_topologies:
            masks_vec = torch.cat([m.flatten() for m in self.masks.values()])
            masks_density = masks_vec.sum() / masks_vec.numel()
            model_compression = -torch.log10(masks_density + 1e-10)

            for s, m in zip(self.scores.values(), self.masks.values()):
                if len(m.size()) == 4:  # convolution layers
                    out_deg = m.sum(dim=[1, 2, 3])
                    inp_deg = m.sum(dim=[0, 2, 3])
                    outer = torch.outer(out_deg, inp_deg)
                    outer = torch.unsqueeze(outer, 2)
                    outer = torch.unsqueeze(outer, 3)
                    sensitivity = (m / outer.clamp(min=1)) ** (model_compression / self.compression_factor)
                    s.mul_(torch.clone(sensitivity).detach())
                elif len(m.size()) == 2:  # linear layers
                    out_deg = m.sum(dim=1)
                    inp_deg = m.sum(dim=0)
                    outer = torch.outer(out_deg, inp_deg)
                    sensitivity = (m / outer.clamp(min=1)) ** (model_compression / self.compression_factor)
                    s.mul_(torch.clone(sensitivity).detach())

        # Enable parameter grads, disable mask grads
        self.set_params_grads(models=osrc_model, enable=True, detach=True)
        self.set_masks_grads(enable=False, zero_grad=True)

        # Reset momentum of BatchNorm2d
        reset_bn_momentum(osrc_model)
        del psrc_model
        del otgt_model
        del ptgt_model

        return global_ntk_scores_norm, global_nad_scores_norm

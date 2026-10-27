# coding=utf-8


import copy

import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm
from torch_pruning.pruner import function
from torch_pruning.pruner.importance import GroupMagnitudeImportance

__all__ = [
    "GradientImportance",
    "NTKImportance",
    "SNIPImportance",
    "SynflowImportance",
    "GraSPImportance",
    "NTKSAPImportance",
    "PXImportance",
]


class GroupGradientImportance(GroupMagnitudeImportance):
    """Importance score based on gradients."""

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = h.sum(1)

                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    h = layer.bias.grad.data[idxs]
                    local_imp = h
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        h = (layer.weight.grad).flatten(1)
                    else:
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = h.sum(1)

                    if (
                        prune_fn == function.prune_conv_in_channels
                        and layer.groups != layer.in_channels
                        and layer.groups != 1
                    ):
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        h = layer.weight.grad.data[idxs]
                        local_imp = h
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        h = layer.bias.grad.data[idxs]
                        local_imp = h
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        h = layer.weight.grad.data[idxs]
                        local_imp = h
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        h = layer.bias.grad.data[idxs]
                        local_imp = h
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class GradientImportance(GroupGradientImportance):
    """NTK Importance score proposed in "WideWideTopo: Improving foresight neural network pruning
    through training dynamics preservation and wide topologies exploration"

    https://www.sciencedirect.com/science/article/pii/S0893608025010160
    """

    pass


class GroupNTKImportance(GroupGradientImportance):
    """NTK Importance score proposed in "WideWideTopo: Improving foresight neural network pruning
    through training dynamics preservation and wide topologies exploration"

    https://www.sciencedirect.com/science/article/pii/S0893608025010160
    """

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        ntk_val_mode: str = "standard",
        accu_ntk_steps: int = 1,
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.ntk_val_mode = ntk_val_mode
        self.accu_ntk_steps = accu_ntk_steps
        self.target_types = target_types
        self.bias = bias
        self._accu_grad = {}
        self._accu_ntk = {}
        self._counter = {}

    def zero_grad(self):
        self._accu_grad = {}
        self._accu_ntk = {}
        self._counter = {}

    @property
    def is_prepared(self) -> bool:
        return len(self._accu_ntk) > 0

    def _calculate_ntk_value(self, outputs):
        if self.ntk_val_mode == "standard":
            if isinstance(outputs, torch.Tensor):
                dot = torch.randn_like(outputs).mul_(outputs).sum()
            elif isinstance(outputs, (list, tuple)):
                dot = sum(torch.randn_like(out).mul_(out).sum() for out in outputs)
            elif isinstance(outputs, dict):
                dot = sum(torch.randn_like(out).mul_(out).sum() for out in outputs.values())
            else:
                raise ValueError("Unsupported output type for NTK importance accumulation.")

        elif self.ntk_val_mode == "first":
            if isinstance(outputs, torch.Tensor):
                dot = torch.randn_like(outputs).mul_(outputs).sum()
            elif isinstance(outputs, (list, tuple)):
                dot = torch.randn_like(outputs[0]).mul_(outputs[0]).sum()
            elif isinstance(outputs, dict):
                outputs = list(outputs.values())
                dot = torch.randn_like(outputs[0]).mul_(outputs[0]).sum()
            else:
                raise ValueError("Unsupported output type for NTK importance accumulation.")

        elif self.ntk_val_mode == "last":
            if isinstance(outputs, torch.Tensor):
                dot = torch.randn_like(outputs).mul_(outputs).sum()
            elif isinstance(outputs, (list, tuple)):
                dot = torch.randn_like(outputs[-1]).mul_(outputs[-1]).sum()
            elif isinstance(outputs, dict):
                outputs = list(outputs.values())
                dot = torch.randn_like(outputs[-1]).mul_(outputs[-1]).sum()
            else:
                raise ValueError("Unsupported output type for NTK importance accumulation.")

        elif self.ntk_val_mode == "balance":
            if isinstance(outputs, torch.Tensor):
                dot = torch.randn_like(outputs).mul_(outputs).sum() / outputs.numel()
            elif isinstance(outputs, (list, tuple)):
                dot = sum(torch.randn_like(out).mul_(out).sum() / out.numel() for out in outputs)
            elif isinstance(outputs, dict):
                dot = sum(torch.randn_like(out).mul_(out).sum() / out.numel() for out in outputs.values())
            else:
                raise ValueError("Unsupported output type for NTK importance accumulation.")

        else:
            raise ValueError(f"Unsupported ntk_val_mode: {self.ntk_val_mode}")

        return dot

    def accumulate_score(self, model, outputs):
        dot = self._calculate_ntk_value(outputs)
        grads = grad(dot, model.parameters(), create_graph=True, allow_unused=True)
        grads_flatten = torch.cat([g.flatten() for g in grads if g is not None])

        trace = (grads_flatten * grads_flatten).sum()
        grads = grad(trace, model.parameters(), create_graph=False, allow_unused=True)
        for (n, p), g in zip(model.named_parameters(), grads):
            if g is None:
                continue
            g_detached = g.detach()
            # gradient accumulation
            if n not in self._accu_grad:
                self._accu_grad[n] = copy.deepcopy(g_detached)
                self._counter[n] = 1
            else:
                self._accu_grad[n].data.add_(g_detached)
                self._counter[n] += 1
            # get ntk score
            if n in self._counter and self._counter[n] % self.accu_ntk_steps == 0:
                current_score = (self._accu_grad[n] * p).abs_().detach()
                if n not in self._accu_ntk:
                    self._accu_ntk[n] = copy.deepcopy(current_score)
                else:
                    self._accu_ntk[n].data.add_(current_score)
                self._accu_grad[n].data.zero_()

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_ntk) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_ntk:
                p.grad = torch.clone(self._accu_ntk[n] / self._counter[n]).detach().to(p.device)


class NTKImportance(GroupNTKImportance):
    """NTK Importance score proposed in "WideWideTopo: Improving foresight neural network pruning
    through training dynamics preservation and wide topologies exploration"

    https://www.sciencedirect.com/science/article/pii/S0893608025010160
    """

    pass


class GroupSNIPImportance(GroupGradientImportance):
    """
    SNIP: Single-shot Network Pruning based on Connection Sensitivity

    SNIP introduce a saliency criterion based on connection sensitivity that identifies structurally important
    connections in the network for the given task.

    Reference:
    https://arxiv.org/abs/1810.02340
    """

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        accu_score_steps: int = 1,
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        super(GroupSNIPImportance, self).__init__(group_reduction, normalizer, bias, target_types)

        self.accu_score_steps = accu_score_steps

        self._accu_grad = {}
        self._accu_score = {}
        self._counter = {}

    def accumulate_score(self, model, pred, truth, loss_func, sum_losses=None):
        losses = loss_func(pred, truth)
        if sum_losses is not None:
            L = sum_losses(losses)
        else:
            L = torch.sum(torch.stack(list(losses.values())))

        grads = grad(L, model.parameters(), create_graph=False, allow_unused=True)
        for (n, p), g in zip(model.named_parameters(), grads):
            if g is None:
                continue
            g_detached = g.detach()
            # gradient accumulation
            if n not in self._accu_grad:
                self._accu_grad[n] = copy.deepcopy(g_detached)
                self._counter[n] = 1
            else:
                self._accu_grad[n].data.add_(g_detached)
                self._counter[n] += 1
            # get score score
            if n in self._counter and self._counter[n] % self.accu_score_steps == 0:
                current_score = (self._accu_grad[n] * p).abs_().detach()
                if n not in self._accu_score:
                    self._accu_score[n] = copy.deepcopy(current_score)
                else:
                    self._accu_score[n].data.add_(current_score)
                self._accu_grad[n].data.zero_()

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_score) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_score:
                p.grad = torch.clone(self._accu_score[n] / self._counter[n]).detach().to(p.device)


class SNIPImportance(GroupSNIPImportance):
    """
    SNIP: Single-shot Network Pruning based on Connection Sensitivity

    SNIP introduce a saliency criterion based on connection sensitivity that identifies structurally important
    connections in the network for the given task.

    Reference:
    https://arxiv.org/abs/1810.02340
    """

    pass


class GroupGarSPImportance(GroupGradientImportance):
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

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        super(GroupGarSPImportance, self).__init__(group_reduction, normalizer, bias, target_types)

        self._accu_score = {}

    def accumulate_score(self, model, pred, truth, loss_func, sum_losses=None):
        losses = loss_func(pred, truth)
        if sum_losses is not None:
            L = sum_losses(losses)
        else:
            L = torch.sum(torch.stack(list(losses.values())))

        grads = grad(L, model.parameters(), create_graph=True, allow_unused=True)
        grads_flatten = torch.cat([g.flatten() for g in grads if g is not None])
        dot = (grads_flatten * grads_flatten.detach()).sum()

        grads = grad(dot, model.parameters(), create_graph=False, allow_unused=True)

        for (n, p), g in zip(model.named_parameters(), grads):
            if g is not None:
                if n not in self._accu_score:
                    self._accu_score[n] = torch.zeros_like(p)
                self._accu_score[n].add_(torch.clone(g * p).detach())

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_score) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_score:
                p.grad = torch.clone(self._accu_score[n]).detach().to(p.device)


class GraSPImportance(GroupGarSPImportance):
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

    pass


class GroupSynflowImportance(GroupGradientImportance):
    """
    Iterative Synaptic Flow Pruning (SynFlow).
    Pruning neural networks without any data by iteratively conserving synaptic flow

    SynFlow can identify highly sparse trainable subnetworks at initialization, without
    ever training, or indeed without ever looking at the data

    Reference:
    https://proceedings.neurips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html
    """

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        super(GroupSynflowImportance, self).__init__(group_reduction, normalizer, bias, target_types)

        self._accu_score = {}

    def accumulate_score(self, model, image):
        @torch.no_grad()
        def named_params(model):
            return dict([(n, p) for n, p in model.named_parameters()])

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

        signs = linearize(model)

        input_dim = list(image[0, :].shape)
        input = torch.ones([1] + input_dim).to(image.device)
        output = model(input)
        output_sum = sum([out.sum() for out in output]) if isinstance(output, (list, tuple)) else output.sum()
        grads = grad(output_sum, model.parameters(), create_graph=False, allow_unused=True)

        for (n, p), g in zip(named_params(model).items(), grads):
            if g is not None:
                if n not in self._accu_score:
                    self._accu_score[n] = torch.zeros_like(p)
                self._accu_score[n].add_(torch.clone(g * p).detach().abs_())

        nonlinearize(model, signs)

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_score) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_score:
                p.grad = torch.clone(self._accu_score[n]).detach().to(p.device)


class SynflowImportance(GroupSynflowImportance):
    """
    Iterative Synaptic Flow Pruning (SynFlow).
    Pruning neural networks without any data by iteratively conserving synaptic flow

    SynFlow can identify highly sparse trainable subnetworks at initialization, without
    ever training, or indeed without ever looking at the data

    Reference:
    https://proceedings.neurips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html
    """

    pass


class GroupNTKSAPImportance(GroupGradientImportance):
    """
    NTK-SAP: Improving neural network pruning by aligning training dynamics

    Reference:
    https://openreview.net/forum?id=-5EWhW_4qWP
    https://github.com/YiteWang/NTK-SAP
    """

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",
        perturb_epsilon=1e-4,
        accu_ntk_steps: int = 1,
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        super(GroupNTKSAPImportance, self).__init__(group_reduction, normalizer, bias, target_types)

        self.perturb_epsilon = perturb_epsilon
        self.accu_ntk_steps = accu_ntk_steps

        self._accu_score = {}
        self._accu_ntk = {}
        self._counter = {}

    @torch.no_grad()
    def perturb(self, model_orig, model_copy):
        """Perturb copied model"""
        params_orig = dict(model_orig.named_parameters())
        params_copy = dict(model_copy.named_parameters())
        for (no, po), (nc, pc) in zip(params_orig.items(), params_copy.items()):
            if no != nc:
                raise ValueError(f"Parameter names do not match: {no} vs {nc}")
            # generate same random values across ranks
            delta = torch.randn_like(po.data)
            # parameter perturbation
            pc.data.copy_(po.data + self.perturb_epsilon * delta)

        for module, module_cpy in zip(model_orig.modules(), model_copy.modules()):
            if isinstance(module, _BatchNorm):
                module_cpy.running_mean.data.copy_(module.running_mean.data)
                module_cpy.running_var.data.copy_(module.running_var.data)
                module_cpy.num_batches_tracked.data.copy_(module.num_batches_tracked.data)

    @torch.no_grad()
    def initialize_weights(self, model):
        """initialize weights"""

        def kaiming_normal_init(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data, mean=0, std=0.2)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight.data, 1)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
                    # init.normal_(m.bias.data, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight.data, 1)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
                    # init.normal_(m.bias.data, mean=0, std=0.02)

        model.apply(kaiming_normal_init)

    @torch.no_grad()
    def hard_bn_momentum(self, model):
        """Disable momentum of BatchNorm"""

        def func(m):
            if isinstance(m, _BatchNorm):
                m.backup_momentum = m.momentum
                m.momentum = 1.0

        model.apply(func)

    @torch.no_grad()
    def reset_bn_momentum(self, model):
        """Disable momentum of BatchNorm"""

        def func(m):
            if isinstance(m, _BatchNorm) and hasattr(m, "backup_momentum"):
                m.momentum = m.backup_momentum

        model.apply(func)

    @torch.no_grad()
    def reset_bn_status(self, model):
        """Reset running status of BatchNorm"""

        def func(m):
            if isinstance(m, _BatchNorm):
                m.reset_running_stats()

        model.apply(func)

    def set_params_grads(self, models, enable=True, detach=True):
        """Set parameters gradients"""
        if not isinstance(models, list):
            models = [models]
        for model in models:
            for p in model.parameters():
                if detach:
                    p.detach_()
                p.requires_grad = enable

    def set_masks_grads(self, masks, enable=True, zero_grad=True):
        """Set mask gradients"""
        for m in masks.values():
            m.requires_grad = enable
            if m.grad is not None and zero_grad:
                m.grad.data.zero_()

    def apply_masks_on_detached_params(self, model, masks):
        """Apply masks on corresponding detached paramters"""
        params = dict(model.named_parameters())
        for n, m in masks.items():
            p = params[n]
            p.detach_().mul_(m)

    def accumulate_score(self, model, image):
        model_org = copy.deepcopy(model)
        self.initialize_weights(model_org)
        image = torch.randn_like(image)

        model_cpy = copy.deepcopy(model_org)
        model_masks = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        self.set_params_grads(models=[model_org, model_cpy], enable=False, detach=True)
        self.set_masks_grads(masks=model_masks, enable=True, zero_grad=True)

        self.hard_bn_momentum(model_org)
        self.hard_bn_momentum(model_cpy)

        self.reset_bn_status(model_org)

        with torch.no_grad():
            model_org.train()
            self.apply_masks_on_detached_params(model_org, model_masks)
            pred_org = model_org(image)

        model_org.eval()
        self.apply_masks_on_detached_params(model_org, model_masks)
        pred_org = model_org(image)

        self.perturb(model_org, model_cpy)

        model_cpy.eval()
        self.apply_masks_on_detached_params(model_cpy, model_masks)
        pred_cpy = model_cpy(image)

        jac_approx = 0
        if isinstance(pred_org, (list, tuple)):
            for po, pc in zip(pred_org, pred_cpy):
                jac_approx += (torch.norm(po - pc, dim=1) ** 2).sum()
        elif isinstance(pred_org, dict):
            for po, pc in zip(pred_org.values(), pred_cpy.values()):
                jac_approx += (torch.norm(po - pc, dim=1) ** 2).sum()
        else:
            jac_approx = (torch.norm(pred_org - pred_cpy, dim=1) ** 2).sum()

        grads = grad(jac_approx, list(model_masks.values()), create_graph=False, allow_unused=True)

        for (n, m), g in zip(model_masks.items(), grads):
            if g is None:
                continue
            g_detached = g.detach()
            # gradient accumulation
            if n not in self._accu_ntk:
                self._accu_ntk[n] = copy.deepcopy(g_detached)
                self._counter[n] = 1
            else:
                self._accu_ntk[n].data.add_(g_detached)
                self._counter[n] += 1
            # get ntk score
            if n in self._counter and self._counter[n] % self.accu_ntk_steps == 0:
                current_score = (self._accu_ntk[n]).abs_().detach()
                if n not in self._accu_score:
                    self._accu_score[n] = copy.deepcopy(current_score)
                else:
                    self._accu_score[n].data.add_(current_score)
                self._accu_ntk[n].data.zero_()

        del model_cpy, model_org

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_score) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_score:
                p.grad = torch.clone(self._accu_score[n] / self._counter[n]).detach().to(p.device)


class NTKSAPImportance(GroupNTKSAPImportance):
    """
    NTK-SAP: Improving neural network pruning by aligning training dynamics

    Reference:
    https://openreview.net/forum?id=-5EWhW_4qWP
    https://github.com/YiteWang/NTK-SAP
    """

    pass


class GroupPXImportance(GroupGradientImportance):
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
        group_reduction: str = "mean",
        normalizer: str = "mean",
        bias=False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.LayerNorm,
        ],
    ):
        super(GroupPXImportance, self).__init__(group_reduction, normalizer, bias, target_types)

        self._accu_score = {}

        self.orig_relu = F.relu
        self.orig_leakyrelu = F.leaky_relu

    def accumulate_score(self, model, image):
        def hook_activation(input, inplace=False):  # overrides F.relu
            self.activation_maps.append(input.clone().detach())
            return self.orig_relu(input, inplace)

        def apply_activation(input, inplace=False):  # overrides F.relu
            map = torch.where(self.activation_maps.pop(0) > 0, 1.0, 0.0)
            return input * map

        model.zero_grad()

        pk_model = copy.deepcopy(model)
        pk_model.train()
        pk_model.zero_grad()
        with torch.no_grad():
            for p in pk_model.parameters():
                p.pow_(2)

        jvf_model = copy.deepcopy(model)
        jvf_model.train()
        jvf_model.zero_grad()
        with torch.no_grad():
            for p in jvf_model.parameters():
                p.fill_(1.0)

        activation_model = copy.deepcopy(model)
        activation_model.eval()
        activation_model.zero_grad()
        with torch.no_grad():
            F.relu = hook_activation
            F.leaky_relu = hook_activation
            self.activation_maps = []
            activation_model(image)

            F.relu = apply_activation
            input_tensor = image.pow(2)
            pred_z1 = jvf_model(input_tensor)

        F.relu = lambda input, *args, **kwargs: input
        F.leaky_relu = lambda input, *args, **kwargs: input

        input_tensor = torch.ones_like(input_tensor)
        pred_z2 = pk_model(input_tensor)

        sum_val = 0
        if isinstance(pred_z1, (list, tuple)):
            for z1, z2 in zip(pred_z1, pred_z2):
                z1 = torch.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0)
                z2 = torch.nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)
                sum_val += (z1 * z2).sum()
        elif isinstance(pred_z1, dict):
            for z1, z2 in zip(pred_z1.values(), pred_z2.values()):
                z1 = torch.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0)
                z2 = torch.nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)
                sum_val += (z1 * z2).sum()
        else:
            sum_val = (pred_z1 * pred_z2).sum()

        sum_val.backward()

        for (n, p), pk_p in zip(model.named_parameters(), pk_model.parameters()):
            if pk_p.grad is None:
                score = torch.zeros_like(p)
            else:
                score = pk_p.grad.abs() * p.pow(2)

            if n not in self._accu_score:
                self._accu_score[n] = torch.clone(score).detach()
            else:
                self._accu_score[n].data.add_(score)

        F.relu = self.orig_relu
        F.leaky_relu = self.orig_leakyrelu

        del pk_model, jvf_model, activation_model

    def apply_to_model(self, model: nn.Module) -> None:
        if len(self._accu_score) == 0:
            raise RuntimeError("No accumulated gradients found. Call accumulate_score() first.")
        for n, p in model.named_parameters():
            if n in self._accu_score:
                p.grad = torch.clone(self._accu_score[n]).detach().to(p.device)


class PXImportance(GroupPXImportance):
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

    pass

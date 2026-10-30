# coding=utf-8

import copy
import math
import types
import warnings
from typing import Optional

import cvxpy as cp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tools import local_rank, named_params

from .rand_erk import RandERK

warnings.filterwarnings("ignore")

__all__ = [
    "NPB",
]


class IgnoreBatchnorm2d(nn.BatchNorm2d):
    """Ignore BatchNorm"""

    def forward(self, input):
        return input


class ScaleAvgPool2d(nn.AvgPool2d):
    """Scale Average pooling 2d"""

    def forward(self, input):
        if not isinstance(self.kernel_size, tuple):
            scale = self.kernel_size
        else:
            scale = sum(self.kernel_size)

        return scale * F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class ScaleAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """In NAS-Bench-201 adaptive average pooling layer has output size = 1
    scale = h * w
    """

    def forward(self, input):
        _, _, h, w = input.shape
        scale = h * w
        return scale * F.adaptive_avg_pool2d(input, self.output_size)


def copynet(net, replace_batchnorm=True, replace_avgpool=True):
    """To compute the number of effective paths we ignore the batchnorm and re-scale adaptive pooling
    This function use to change batchnorm to IgnoreBatchNorm and
    Adaptive to Scale Adaptive which are used to compute the number of paths
    """
    cloned_net = copy.deepcopy(net)
    if replace_batchnorm:
        for module in cloned_net.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.forward = types.MethodType(IgnoreBatchnorm2d.forward, module)
    if replace_avgpool:
        for module in cloned_net.modules():
            if isinstance(module, nn.AvgPool2d):
                module.forward = types.MethodType(ScaleAvgPool2d.forward, module)
            if isinstance(module, nn.AdaptiveAvgPool2d):
                module.forward = types.MethodType(ScaleAdaptiveAvgPool2d.forward, module)
    return cloned_net


def get_leaf_children(module, prefix=""):
    """Recursively obtain all leaf children and their names in a PyTorch model"""
    leaf_childs = dict()
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if len(list(child.children())) == 0:  # leaf child
            leaf_childs[full_name] = child
        else:
            leaf_childs.update(get_leaf_children(child, full_name))
    return leaf_childs


def get_intermediate_inputs(model, input):
    """Get intermediate input of each leaf child

    Note: Normalization layers and max pooling layers will be ignored
    """
    visualization = dict()

    def hook_fn(m, i, o):
        visualization[m._name] = i[0]

    # register hook function
    flatt_children = get_leaf_children(model)
    for name, child in flatt_children.items():
        child._name = name
        child.register_forward_hook(hook_fn)

    # model inference
    model(input)
    return visualization


def get_param_grads(model, input):
    """Get intermediate input of each leaf child

    Note: Normalization layers and max pooling layers will be ignored
    """
    # model inference
    model.zero_grad()
    y = model(input)
    # model backward
    y.sum().backward()
    return {k: p.grad for k, p in named_params(model).items()}


def get_num_unpruned_params(model, masks: dict):
    """Get the number of unpruned masks"""
    num_unpruned = 0
    for n, p in named_params(model).items():
        if n in masks.keys():
            num_unpruned += masks[n].sum()
        else:  # Pruning for this layer is disabled
            num_unpruned += p.numel()
    return num_unpruned


def layerwise_optimizing_mask(
    mask_shape,
    input_paths,
    density: float,
    alpha: float,
    beta: float,
    max_param_per_kernel: Optional[int] = None,
):
    """Layerwise optimization for NPB

    Params:
        mask_shape: mask shape of this layer
        input_paths: input of this layer
        density : the density of this layer
        alpha : is the balancing coefficient
        beta: is the regularizer coefficient
        max_param_per_kernel: max param in a kernel

    Reference:
        [1] Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?
        [2] github code: https://github.com/pvh1602/NPB
    """
    if len(mask_shape) == 2:  # linear layer
        C_out, C_in = mask_shape
        max_param_per_kernel = 1
        if len(input_paths.shape) == 1:
            P_in = input_paths.detach().cpu().numpy()
        elif len(input_paths.shape) > 1:
            P_in = input_paths.sum(dim=tuple(range(input_paths.dim() - 1))).detach().cpu().numpy()
        else:
            raise AttributeError("Wrong input dimension, tensor dimension must greater equal than 1")
        P_in = P_in / max(1, np.max(P_in))  # Normalize

    elif len(mask_shape) == 4:  # convolution layer
        C_out, C_in, kh, kw = mask_shape
        if max_param_per_kernel is None:
            max_param_per_kernel = kh * kw
        else:
            max_param_per_kernel = min(max_param_per_kernel, kh * kw)
        assert len(input_paths.shape) == 3, "Wrong input dimension, tensor with shape [C, H, W,] is expected"
        P_in = torch.sum(input_paths, dim=(1, 2)).detach().cpu().numpy()
        P_in = P_in / max(1, np.max(P_in))  # Normalize

    else:
        raise AttributeError("Unsupported mask shape, only linear and convolution layers are supported.")

    # params in this layer
    n_params = int(torch.ceil(density * torch.prod(torch.tensor(mask_shape))))

    # Mask variable
    M = cp.Variable((C_in, C_out), integer=True)

    # P_in * \sum_{j}^{ C_out } M_{ij}
    # \sum_{i}^{ C_in} M_{ij} * P_in[i]
    sum_in = cp.sum(M, axis=1) * P_in
    # sum_out = cp.sum(cp.multiply(M, P_in.reshape(-1, 1)), axis=0)
    sum_out = cp.sum(cp.diag(P_in) @ M, axis=0)

    # If eff_node_in is small which means there is a large number of input effective node
    sum_eff_node_in = C_in - cp.sum(cp.pos(1 - sum_in))
    sum_eff_node_out = C_out - cp.sum(cp.pos(1 - sum_out))

    # OPtimize nodes
    max_nodes = C_in + C_out
    A = (sum_eff_node_in + sum_eff_node_out) / max_nodes  # Scale to 1

    # Optimize paths
    min_out_node = int(n_params / (C_out * max_param_per_kernel))
    remainder = n_params - min_out_node * (C_out * max_param_per_kernel)
    try:
        max_path = (
            np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel))
            + remainder * np.sort(P_in)[-(min_out_node + 1)]
        )
    except:
        max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel))
    B = (cp.sum(P_in @ M)) / max_path

    # Regulaziration
    Reg = (n_params - cp.sum(cp.pos(1 - M))) / n_params  # encourage number of edges

    # Constraint the total activated params
    constraint = [cp.sum(M) <= n_params, M <= max_param_per_kernel, M >= 0]

    # Objective function
    obj = cp.Maximize(alpha * A + (1 - alpha) * B + beta * Reg)

    # Init problem
    prob = cp.Problem(obj, constraint)
    prob.solve()  # Solving

    return torch.tensor(M.value.transpose(1, 0), dtype=torch.int, device=input_paths.device)


class NPB(RandERK):
    """
    Node-Path Balancing Principle

    Reference:
    [1] Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    """

    def __init__(
        self,
        param_names: list,
        verbose=False,
        power_scale=1.0,
        alpha=0.01,
        beta=1,
        max_param_per_kernel=None,
        magnitude_based_mask=True,
        enable_chunking_strategy=True,
        linear_chunk_size=100,
        conv_chunk_size=32,
        finetuning_mask=False,
        input_tensor_key="image",
    ):
        super().__init__(param_names, verbose)
        self.power_scale = power_scale
        self.alpha = alpha
        self.beta = beta
        self.max_param_per_kernel = max_param_per_kernel
        self.magnitude_based_mask = magnitude_based_mask
        self.enable_chunking_strategy = enable_chunking_strategy
        self.linear_chunk_size = linear_chunk_size
        self.conv_chunk_size = conv_chunk_size
        self.finetuning_mask = finetuning_mask
        self.input_tensor_key = input_tensor_key

    def update_npb_masks(self, model, input_tensor, layerwise_densities, initial_params):
        """Update self.masks with NPB mask

        Params:
            model: nn.Module. A model which parameters are all ones
            input_tensor. torch.Tensor. A tensor which elements are all ones
            layerwise_densities: dict. The target layerwise param density rates
            initial_params: dict. The initial parameters of the input model
        """
        bar = tqdm(self.masks.items())
        for i, (name, mask) in enumerate(bar):
            self.apply_mask_on_params(model=model)
            intermediate = get_intermediate_inputs(model, input_tensor)

            # do weight pruning
            child_name = name[:-7]  # without `.weight`
            if name.endswith(".weight") and len(mask.shape) == 2:  # linear layers
                C_out, C_in = mask.shape
                if self.enable_chunking_strategy and C_out * C_in > 512 * 10:  # using chunking strategy
                    num_chunks = int(math.ceil(C_out / self.linear_chunk_size))
                    for c in range(num_chunks):
                        chunk_idx_start = c * self.linear_chunk_size
                        chunk_idx_end = min((c + 1) * self.linear_chunk_size, C_out)
                        chunked_mask = mask[chunk_idx_start:chunk_idx_end]
                        optimized_mask = layerwise_optimizing_mask(
                            chunked_mask.shape,
                            intermediate[child_name][0],
                            density=layerwise_densities[name],
                            alpha=self.alpha,
                            beta=0,
                        )
                        mask[chunk_idx_start:chunk_idx_end].copy_(torch.clone(optimized_mask))

                        bar.set_description(
                            f"[{self.__class__.__name__}] Pruning layer index: {i + 1}/{len(bar)}, "
                            f"chunk index: {c + 1}/{num_chunks}, "
                            f"chunk size: {chunk_idx_end - chunk_idx_start}"
                        )
                else:
                    optimized_mask = layerwise_optimizing_mask(
                        mask.shape,
                        intermediate[child_name][0],
                        density=layerwise_densities[name],
                        alpha=self.alpha,
                        beta=0,
                    )
                    mask.copy_(torch.clone(optimized_mask))

                    bar.set_description(f"[{self.__class__.__name__}] Pruning layer index: {i + 1}/{len(bar)}")

            elif name.endswith(".weight") and len(mask.shape) == 4:  # convolution layers
                C_out, C_in, kh, kw = mask.shape
                if self.enable_chunking_strategy and C_out * C_in > 128 * 128:  # using chunking strategy
                    num_chunks = int(math.ceil(C_out / self.conv_chunk_size))
                    for c in range(num_chunks):
                        chunk_idx_start = c * self.conv_chunk_size
                        chunk_idx_end = min((c + 1) * self.conv_chunk_size, C_out)
                        chunked_mask = mask[chunk_idx_start:chunk_idx_end]
                        optimized_mask = layerwise_optimizing_mask(
                            chunked_mask.shape,
                            intermediate[child_name][0],
                            density=layerwise_densities[name],
                            alpha=self.alpha,
                            beta=self.beta,
                            max_param_per_kernel=self.max_param_per_kernel,
                        )

                        kernel_arange = torch.arange(kh * kw, device=optimized_mask.device)
                        kernel_select = (kernel_arange < optimized_mask.unsqueeze(-1)).int()
                        if self.magnitude_based_mask:
                            param = initial_params[name]
                            param_mag = param.view(C_out, C_in, -1).abs()
                            param_mag = param_mag[chunk_idx_start:chunk_idx_end]
                            mag_indices = param_mag.argsort(dim=-1, descending=True)
                            binary_mask = torch.zeros_like(kernel_select)
                            binary_mask.scatter_(-1, index=mag_indices, src=kernel_select)
                        else:
                            rand_mag = torch.randn(*kernel_select.shape).to(mask.device)
                            rand_indices = rand_mag.argsort(dim=-1, descending=True)
                            binary_mask = torch.zeros_like(kernel_select)
                            binary_mask.scatter_(-1, index=rand_indices, src=kernel_select)

                        binary_mask = binary_mask.view(*chunked_mask.shape)
                        mask[chunk_idx_start:chunk_idx_end].copy_(torch.clone(binary_mask))

                        bar.set_description(
                            f"[{self.__class__.__name__}] Pruning layer index: {i + 1}/{len(bar)}, "
                            f"chunk index: {c + 1}/{num_chunks}, "
                            f"chunk size: {chunk_idx_end - chunk_idx_start}"
                        )
                else:
                    optimized_mask = layerwise_optimizing_mask(
                        mask.shape,
                        intermediate[child_name][0],
                        density=layerwise_densities[name],
                        alpha=self.alpha,
                        beta=self.beta,
                        max_param_per_kernel=self.max_param_per_kernel,
                    )

                    kernel_arange = torch.arange(kh * kw, device=optimized_mask.device)
                    kernel_select = (kernel_arange < optimized_mask.unsqueeze(-1)).int()
                    if self.magnitude_based_mask:
                        param = initial_params[name]
                        param_mag = param.view(C_out, C_in, -1).abs()
                        mag_indices = param_mag.argsort(dim=-1, descending=True)
                        binary_mask = torch.zeros_like(kernel_select)
                        binary_mask.scatter_(-1, index=mag_indices, src=kernel_select)
                    else:
                        rand_mag = torch.randn(*kernel_select.shape).to(mask.device)
                        rand_indices = rand_mag.argsort(dim=-1, descending=True)
                        binary_mask = torch.zeros_like(kernel_select)
                        binary_mask.scatter_(-1, index=rand_indices, src=kernel_select)

                    binary_mask = binary_mask.view(*mask.shape)
                    mask.copy_(torch.clone(binary_mask))

                    bar.set_description(f"[{self.__class__.__name__}] Pruning layer index: {i + 1}/{len(bar)}")

            else:  # other layers
                mask.copy_(torch.ones_like(mask).detach())

            assert mask.min() >= 0 and mask.max() <= 1, f"invalid mask for {name}"
            # dist.broadcast(mask, src=0)

    def fine_tune_masks(self, model, input_tensor):
        """Fine tuning the self.masks

        Params:
            model: nn.Module. A model which parameters are all ones
            input_tensor. torch.Tensor. A tensor which elements are all ones
        """
        self.apply_mask_on_params(model=model)
        named_grads = get_param_grads(model=model, input=input_tensor)

        num_ineff_params_after = 0

        bar = tqdm(self.masks.items())
        for i, (name, mask) in enumerate(bar):
            bar.set_description(
                f"[{self.__class__.__name__}] Fine-tuning layer index: {i + 1}/{len(bar)}, "
                f"num ineffective params: {int(num_ineff_params_after)}"
            )

            if name.endswith(".weight") and len(mask.shape) == 4:  # convolution layers
                C_out, C_in, kh, kw = mask.shape
                grad = named_grads[name]
                if grad is not None:
                    grad = grad * mask  # apply mask on grads
                    eff_mask = torch.where(grad != 0, 1, 0)
                    num_ineff_params = (mask - eff_mask).sum() + num_ineff_params_after

                    # add specific number of ones to the mask
                    new_mask = eff_mask.view(-1, kh, kw)
                    finetune_idx = torch.argsort(eff_mask.sum(dim=(2, 3)).view(-1), -1, True)
                    for idx in finetune_idx:
                        if num_ineff_params <= 0:
                            break
                        num_one_elems = new_mask[idx].sum()
                        num_finetuning_needed = kh * kw - num_one_elems
                        num_ineff_params = num_ineff_params - num_finetuning_needed
                        new_mask[idx].copy_(torch.ones_like(new_mask[idx]))
                    new_mask = new_mask.view(C_out, C_in, kh, kw)
                    mask.data.copy_(new_mask)

                    # check if still has ineff params
                    self.apply_mask_on_params(model=model)
                    named_grads = get_param_grads(model=model, input=input_tensor)
                    grad = named_grads[name]
                    grad = grad * mask  # apply mask on grads
                    eff_mask = torch.where(grad != 0, 1, 0)
                    num_ineff_params_after = (mask - eff_mask).sum()

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
        input_tensor_shape = next(iter(dataloader))[self.input_tensor_key].shape

        # Check parameters density globally
        params_dict = named_params(self._model)
        params_density = self.layerwise_density(density=density)

        num_params = sum([v.numel() for v in params_dict.values()])
        num_unpruned_needed = int(density * num_params)

        # copy net and set parameter value to ones
        if isinstance(self._model, DistributedDataParallel):
            model_cpy = copynet(self._model.module)
        else:
            model_cpy = copynet(self._model)
        for p in model_cpy.parameters():
            p.data.copy_(torch.ones_like(p))
        model_cpy = model_cpy.double()
        model_cpy.eval()

        # create a tensor with all ones
        input_tensor = torch.ones([1, *input_tensor_shape[1:]]).double()
        input_tensor = input_tensor.to(torch.device(local_rank()))

        # Get masks
        self.update_npb_masks(model_cpy, input_tensor, params_density, initial_params=params_dict)
        num_unpruned = get_num_unpruned_params(model_cpy, self.masks)
        if abs(num_unpruned_needed - num_unpruned) / num_params > tolerance:
            raise AttributeError(
                f"Out of {num_params} parameters, {num_params - num_unpruned_needed} parameters should be pruned, "
                f"but {num_params - num_unpruned} parameters has been pruned"
            )

        # copy mask
        if self.finetuning_mask:
            masks_cpy = copy.deepcopy(self.masks)

            # Fine-tuning mask.
            # Note that it causes the final overall density rate to deviate significantly from the target density rate.
            model_tmp = copy.deepcopy(model_cpy)
            model_tmp.eval()
            self.fine_tune_masks(model_tmp, input_tensor)
            num_unpruned = get_num_unpruned_params(model_cpy, self.masks)
            if abs((num_unpruned / num_params) / density - 1) > tolerance:
                if self.verbose:
                    print("Masks after fine-tuning is unacceptable, use masks without fine-tuning")
                for name, mask in self.masks.items():
                    mask.copy_(masks_cpy[name])

            del model_tmp

        # delete copied model
        del model_cpy

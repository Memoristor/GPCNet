# coding=utf-8

import copy

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tools import hard_bn_momentum, named_params, reset_bn_momentum, reset_bn_status, utils

__all__ = [
    "effective_paths",
    "effective_params",
    "effective_nodes",
    "trace_of_ntk",
]


def bypass_bn_norm(module, input, output):
    """
    Bypass BatchNorm layer
    """
    view_shape = torch.ones(len(input[0].shape)).int().tolist()
    view_shape[1] = -1
    weight = module.weight.view(*view_shape)
    bias = module.bias.view(*view_shape)
    return input[0] * weight + bias


def bypass_ln_norm(module, input, output):
    """
    Bypass LayerNorm layer
    """
    return input[0] * module.weight + module.bias


def copy_model(model):
    """Copy model in distributed data parallel"""
    if isinstance(model, DistributedDataParallel):
        return copy.deepcopy(model.module)
    else:
        return copy.deepcopy(model)


def copy_ddp_model(model):
    """Copy model in distributed data parallel"""
    if isinstance(model, DistributedDataParallel):
        return DistributedDataParallel(
            copy.deepcopy(model.module),
            device_ids=[utils.local_rank],
            output_device=utils.local_rank,
            # find_unused_parameters=True
        )
    else:
        return copy.deepcopy(model)


@torch.no_grad()
def effective_paths(pruner, dataloader, input_tensor_key="image"):
    """
    Effective Path: We define a path to be effective if it connects an input node to an output node without interruption.

    Reference:
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    """
    model = pruner._model

    # Copy model
    model_cpy = copy_model(model)

    # Set model to evaluation mode
    model_cpy.eval()
    model_cpy.double()

    # Set hard momentum of BatchNorm
    hard_bn_momentum(model_cpy)
    reset_bn_status(model_cpy)

    # Bypass Normalization layers
    for n, m in model_cpy.named_modules():
        if isinstance(m, _BatchNorm):
            m.register_forward_hook(bypass_bn_norm)
        elif isinstance(m, nn.LayerNorm):
            m.register_forward_hook(bypass_ln_norm)

    # Dense effective paths
    for n, p in named_params(model_cpy).items():
        p.data.copy_(torch.ones_like(p))

    truth = next(iter(dataloader))
    input_dim = list(truth[input_tensor_key][0, :].shape)
    input_tsr = torch.ones([1] + input_dim).double().to(p.device)
    output_tsr = model_cpy(input_tsr)
    dense_effective_paths = output_tsr.sum()

    # Pruned effective paths
    for n, p in named_params(model_cpy).items():
        if n in pruner.masks.keys():
            p.data.copy_(pruner.masks[n])
        else:
            p.data.copy_(torch.ones_like(p))

    output_tsr = model_cpy(input_tsr)
    pruned_effective_paths = output_tsr.sum()

    # Reset momentum of BatchNorm2d
    reset_bn_momentum(model_cpy)

    # Delete model
    del model_cpy

    return dense_effective_paths, pruned_effective_paths


def effective_params(pruner, dataloader, input_tensor_key="image"):
    """
    Effective Param: After having the sum of logits, we compute the gradients of this sum with respect to weights ∇wR.
        Then, if an unpruned weight has a non-zero gradient, it is effective and vice versa.

    Reference:
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    """
    model = pruner._model

    # Copy model
    model_cpy = copy_model(model)

    # Set model to evaluation mode
    model_cpy.eval()
    model_cpy.double()
    model_cpy.zero_grad(set_to_none=True)

    # Set hard momentum of BatchNorm
    hard_bn_momentum(model_cpy)
    reset_bn_status(model_cpy)

    # Bypass Normalization layers
    # for n, m in model_cpy.named_modules():
    #     if isinstance(m, _BatchNorm):
    #         m.register_forward_hook(bypass_bn_norm)
    #     elif isinstance(m, nn.LayerNorm):
    #         m.register_forward_hook(bypass_ln_norm)

    # Effective params
    for n, p in named_params(model_cpy).items():
        if n in pruner.masks.keys():
            p.data.copy_(pruner.masks[n])
        else:
            p.data.copy_(torch.ones_like(p))

    truth = next(iter(dataloader))
    input_dim = list(truth[input_tensor_key][0, :].shape)
    input_tsr = torch.ones([1] + input_dim).double().to(p.device)
    output_tsr = model_cpy(input_tsr)
    output_tsr = output_tsr.sum()
    output_tsr.backward()

    dense_effective_params = 0
    pruned_effective_params = 0
    for n, p in named_params(model_cpy).items():
        g = p.grad
        if n in pruner.masks.keys():
            m = pruner.masks[n]
            g = g * m
        dense_effective_params += torch.ones_like(g).sum()
        pruned_effective_params += (g != 0).int().sum()

    # Reset momentum of BatchNorm2d
    reset_bn_momentum(model_cpy)

    # Delete model
    del model_cpy

    return dense_effective_params, pruned_effective_params


@torch.no_grad()
def effective_nodes(pruner):
    """
    A node/channel is effective if at least one effective path goes through it. This concept is also considered in [36, 17].
    or convolutional layers, we consider a kernel as a connection, and a channel as a node, and then convert the convolutional
    layer into a fully connected layer.

    Reference:
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    """
    model = pruner._model

    # Copy model
    model_cpy = copy_model(model)

    # Effective nodes
    dense_effective_nodes = 0
    pruned_effective_nodes = 0

    for n, p in named_params(model_cpy).items():
        # Set parameter to ones tensor
        p.data.copy_(torch.ones_like(p))

        # Get parameter mask
        if n in pruner.masks.keys():
            m = pruner.masks[n]
        else:
            m = torch.ones_like(p)

        # Convolution kernel and linear layer
        if len(m.size()) == 4:
            dense_sum_along_in_channels = torch.sum(p, dim=[0, 2, 3])  # sum along input channels
            dense_effective_nodes += torch.where(dense_sum_along_in_channels > 0, 1, 0).sum()
            pruned_sum_along_in_channels = torch.sum(m, dim=[0, 2, 3])
            pruned_effective_nodes += torch.where(pruned_sum_along_in_channels > 0, 1, 0).sum()

        elif len(m.size()) == 2:
            dense_sum_along_in_channels = torch.sum(p, dim=[0])  # sum along input channels
            dense_effective_nodes += torch.where(dense_sum_along_in_channels > 0, 1, 0).sum()
            pruned_sum_along_in_channels = torch.sum(m, dim=[0])
            pruned_effective_nodes += torch.where(pruned_sum_along_in_channels > 0, 1, 0).sum()

    # Delete model
    del model_cpy

    return dense_effective_nodes, pruned_effective_nodes


@torch.no_grad()
def effective_nodes_v2(pruner):
    """
    A node/channel is effective if at least one effective path goes through it. This concept is also considered in [36, 17].
    or convolutional layers, we consider a kernel as a connection, and a channel as a node, and then convert the convolutional
    layer into a fully connected layer.

    Reference:
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html
    """
    model = pruner._model

    # Copy model
    model_cpy = copy_model(model)

    # Effective nodes
    dense_effective_nodes = 0
    pruned_effective_nodes = 0

    for n, p in named_params(model_cpy).items():
        # Set parameter to ones tensor
        p.data.copy_(torch.ones_like(p))

        # Get parameter mask
        if n in pruner.masks.keys():
            m = pruner.masks[n]
        else:
            m = torch.ones_like(p)

        # Convolution kernel and linear layer
        if len(m.size()) == 4:
            dense_effective_nodes += torch.where(torch.sum(p, dim=[0, 2, 3]) > 0, 1, 0).sum()
            dense_effective_nodes += torch.where(torch.sum(p, dim=[1, 2, 3]) > 0, 1, 0).sum()
            pruned_effective_nodes += torch.where(torch.sum(m, dim=[0, 2, 3]) > 0, 1, 0).sum()
            pruned_effective_nodes += torch.where(torch.sum(m, dim=[1, 2, 3]) > 0, 1, 0).sum()

        elif len(m.size()) == 2:
            dense_effective_nodes += torch.where(torch.sum(p, dim=[0]) > 0, 1, 0).sum()
            dense_effective_nodes += torch.where(torch.sum(p, dim=[1]) > 0, 1, 0).sum()
            pruned_effective_nodes += torch.where(torch.sum(m, dim=[0]) > 0, 1, 0).sum()
            pruned_effective_nodes += torch.where(torch.sum(m, dim=[1]) > 0, 1, 0).sum()

    # Delete model
    del model_cpy

    return dense_effective_nodes, pruned_effective_nodes


def trace_of_ntk(pruner, dataloader, hutch_iters, predict_func, logit_tensor_key="logit"):
    """
    Trace estimation of NTK based on Huchinson's method

    Reference:
    https://proceedings.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html
    """
    model = pruner._model

    # Copy model
    model_cpy = copy_ddp_model(model)
    model_cpy.zero_grad()

    # Get masks vector
    masks = {k: torch.ones_like(p) for k, p in named_params(model_cpy).items()}
    for k, m in pruner.masks.items():
        masks[k].copy_(m)
    masks_vec = torch.cat([m.flatten() for m in masks.values()])

    # Synchronizes all processes
    dist.barrier()

    # Set hard momentum of BatchNorm
    hard_bn_momentum(model_cpy)

    bar = tqdm(dataloader)
    for iter, truth in enumerate(bar):
        for k, v in truth.items():
            truth[k] = v.to(torch.device(utils.local_rank))

        reset_bn_status(model_cpy)

        # Init running parameters of BatchNorm
        with torch.no_grad():
            model_cpy.train()
            predict_func(model_cpy, truth)

        # Compute the true graph using eval mode
        for _ in range(hutch_iters):
            model_cpy.eval()
            pred = predict_func(model_cpy, truth)
            logit = pred[logit_tensor_key]
            coff = torch.randn_like(logit)
            dots = (coff * logit).sum()
            dots.backward()

        bar.set_description(
            f"Trace estimation of NTK, "
            f"Rank: {utils.local_rank}, "
            f"Density: {masks_vec.sum() / masks_vec.numel():5.5f}"
        )

    # Trace estimation
    grads_vec = torch.cat([p.grad.flatten() for p in model_cpy.parameters()])
    if dist.get_world_size() > 1:
        dist.all_reduce(grads_vec, op=dist.ReduceOp.SUM)

    grads_vec = grads_vec * masks_vec
    trace = (grads_vec * grads_vec).sum() / hutch_iters

    # Reset momentum of BatchNorm2d
    reset_bn_momentum(model_cpy)

    # Delete model
    del model_cpy

    return trace

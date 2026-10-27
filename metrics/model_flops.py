# coding=utf-8

import numpy as np
import torch
import torch.nn as nn

__all__ = ["flops_by_params"]


def flops_by_params(model: nn.Module, input_shape):
    """FLOPs estimation by weights and bias

    Reference:
        Github: https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Utils/metrics.py
    """

    total = {}

    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, nn.Linear):
                # in_features = module.in_features
                # out_features = module.out_features
                # flops["weight"] = in_features * out_features
                # if module.bias is not None:
                #     flops["bias"] = out_features
                in_features = module.in_features
                out_features = module.out_features
                num_elements = input[0].numel() // in_features
                flops["weight"] = in_features * out_features * num_elements
                if module.bias is not None:
                    flops["bias"] = out_features * num_elements
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops["weight"] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops["bias"] = out_channels * output_size
            if isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    flops["weight"] = module.num_features
                    flops["bias"] = module.num_features
            if isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops["weight"] = module.num_features * output_size
                    flops["bias"] = module.num_features * output_size
            if isinstance(module, nn.LayerNorm):
                output_size = output[0].numel()
                params_per_element = int(torch.prod(torch.tensor(module.normalized_shape)))
                if module.elementwise_affine:
                    flops["weight"] = params_per_element * output_size
                    flops["bias"] = params_per_element * output_size
            total[name] = flops

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    device = next(model.parameters()).device
    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total

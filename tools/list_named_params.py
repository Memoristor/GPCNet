# coding=utf-8


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from torch import nn

import models


def parse_args():
    parser = argparse.ArgumentParser(description="Check Named Parameters")
    parser.add_argument(
        "--model_class",
        type=str,
        help="The class name of the model, which needs to be included in the `models` package",
    )
    parser.add_argument("--num_classes", type=int, help="The number of classes")
    parser.add_argument("--except_conv", action="store_true", help="Except Convolution layers")
    parser.add_argument("--except_linear", action="store_true", help="Except Linear layers")
    parser.add_argument("--except_bn", action="store_true", help="Except Batch Norm layers")
    parser.add_argument("--except_ln", action="store_true", help="Except Layer Norm layers")
    parser.add_argument("--except_bias", action="store_true", help="Except Bias terms")
    parser.add_argument("--except_1d_tensor", action="store_true", help="Except 1-D tenosrs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        model = getattr(models, args.model_class)(num_classes=args.num_classes)
    except:
        try:
            model = getattr(models, args.model_class)(num_class=args.num_classes)
        except Exception as e:
            raise e

    params_names = [n for n, p in model.named_parameters()]

    if args.except_conv:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                if hasattr(m, "weight") and getattr(m, "weight") is not None:
                    if n + ".weight" in params_names:
                        params_names.remove(n + ".weight")
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    if n + ".bias" in params_names:
                        params_names.remove(n + ".bias")

    if args.except_linear:
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight") and getattr(m, "weight") is not None:
                    if n + ".weight" in params_names:
                        params_names.remove(n + ".weight")
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    if n + ".bias" in params_names:
                        params_names.remove(n + ".bias")

    if args.except_bn:
        for n, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if hasattr(m, "weight") and getattr(m, "weight") is not None:
                    if n + ".weight" in params_names:
                        params_names.remove(n + ".weight")
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    if n + ".bias" in params_names:
                        params_names.remove(n + ".bias")

    if args.except_ln:
        for n, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                if hasattr(m, "weight") and getattr(m, "weight") is not None:
                    if n + ".weight" in params_names:
                        params_names.remove(n + ".weight")
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    if n + ".bias" in params_names:
                        params_names.remove(n + ".bias")

    if args.except_bias:
        for n, m in model.named_modules():
            if hasattr(m, "bias") and getattr(m, "bias") is not None:
                if n + ".bias" in params_names:
                    params_names.remove(n + ".bias")

    if args.except_1d_tensor:
        for n, m in model.named_modules():
            if (
                hasattr(m, "weight")
                and getattr(m, "weight") is not None
                and len(m.weight.size()) == 1
            ):
                if n + ".weight" in params_names:
                    params_names.remove(n + ".weight")
            if hasattr(m, "bias") and getattr(m, "bias") is not None and len(m.bias.size()) == 1:
                if n + ".bias" in params_names:
                    params_names.remove(n + ".bias")

    for n in params_names:
        print(f"- {n}")

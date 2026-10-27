# coding=utf-8

from typing import Union

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    "load_init",
    "general_init",
    "kaiming_normal_init",
    "trunc_normal_init",
]


def load_init(model: nn.Module, pretrained_dict: Union[str, dict], verbose: bool = True):
    """Init model from checkpoint

    Usage:
        model = Model()
        load_init(model, path_to_cpt/cpt_dict)
    """
    if isinstance(pretrained_dict, str):
        if verbose:
            print(f"=> loading pretrained checkpoint from `{pretrained_dict}`.")
        pretrained_dict = torch.load(pretrained_dict, map_location="cpu")

    loaded_keys = []
    ignored_keys = []

    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                ignored_keys.append(k)

    if verbose:
        print(f"=> loading {len(loaded_keys)}/{len(model_dict)} states from pretrained checkpoint.")
        for k in ignored_keys:
            print(
                f"=> The shape of tensor `{k}` in the model ({model_dict[k].shape}) "
                f"does not match the shape ({v.shape}) in the pretrained checkpoint."
            )

    model.load_state_dict(model_dict, strict=False)


def general_init(m):
    """
    General weight init method

    Usage:
        model = Model()
        model.apply(general_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LayerNorm):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def kaiming_normal_init(m):
    """Weight init by kaiming normal

    Follow the settings of https://openreview.net/forum?id=-5EWhW_4qWP

    Usage:
        model = Model()
        model.apply(kaiming_normal_init)
    """
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


def trunc_normal_init(m):
    if isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

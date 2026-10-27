# coding=utf-8
from __future__ import absolute_import, division, print_function

import copy
import logging
import os

import torch
import torch.nn as nn

from .mamba_sys import VSSM

logger = logging.getLogger(__name__)


class MambaUnet(nn.Module):
    def __init__(self, num_classes=21843, zero_head=False, pretrained_path="models/pretrained/vmamba_tiny_e292.pth"):
        super(MambaUnet, self).__init__()

        self.zero_head = zero_head

        self.mamba_unet = VSSM(in_chans=3, num_classes=num_classes)

        self.load_from(pretrained_path=pretrained_path)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, pretrained_path=None):
        # pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            if not os.path.exists(pretrained_path):
                print("pretrained model not exists:{}".format(pretrained_path))
                return

            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict["model"]

            print("---start load pretrained model of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

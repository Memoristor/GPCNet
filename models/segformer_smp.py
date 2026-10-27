# coding=utf-8


import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Segformer
from torch import nn

from .mix_transformers import mix_transformer_encoders

__all__ = ["SegformerB0", "SegformerB1", "SegformerB2", "SegformerB5"]

smp.encoders.encoders.update(mix_transformer_encoders)


class SegformerB0(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Segformer(
            encoder_name="mit-b0",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class SegformerB1(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Segformer(
            encoder_name="mit-b1",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class SegformerB2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Segformer(
            encoder_name="mit-b2",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output


class SegformerB5(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.net = Segformer(
            encoder_name="mit-b5",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.net(x)
        return output

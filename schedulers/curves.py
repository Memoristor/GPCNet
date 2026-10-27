# coding=utf-8


import math

__all__ = [
    "linear",
    "cosine",
    "poly",
]


def linear(x1, y1, x2, y2, x):
    """Linear curve function"""
    ratio = (y1 - y2) / (x2 - x1)
    return y1 - ratio * (x - x1)


def cosine(x1, y1, x2, y2, x):
    """Cosine curve function"""
    cosine = math.cos((x - x1) / (x2 - x1) * math.pi)
    return y2 + (y1 - y2) / 2.0 * (1 + cosine)


def poly(x1, y1, x2, y2, x):
    """poly curve function"""
    alpha = math.exp(math.log(y1 / y2) / (x2 - x1))
    return y1 * alpha ** -(x - x1)

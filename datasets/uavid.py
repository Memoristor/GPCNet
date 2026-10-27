# coding=utf-8

import json
import os
import re

import numpy as np

from datasets.base_datasets import SegmentationDataset
from tools.utils import find_files

__all__ = ["UAVidDataset"]


class UAVidDataset(SegmentationDataset):
    """
    UAVid is a high-resolution UAV semantic segmentation dataset as a complement, which brings
    new challenges, including large scale variation, moving object recognition and temporal
    consistency preservation. The UAV dataset consists of 30 video sequences capturing 4K high-resolution
    images in slanted views. In total, 300 images have been densely labeled with 8 classes for
    the semantic labeling task.

    reference:
    https://uavid.nl/
    https://doi.org/10.1016/j.isprsjprs.2020.05.009
    https://paperswithcode.com/dataset/uavid
    """

    CLASSES = (
        "clutter",
        "building",
        "road",
        "tree",
        "low vegetation",
        "moving car",
        "static car",
        "human",
    )

    PALETTE = [
        [0, 0, 0],
        [128, 0, 0],
        [128, 64, 128],
        [0, 128, 0],
        [128, 128, 0],
        [64, 0, 128],
        [192, 0, 192],
        [64, 64, 0],
    ]

    def __init__(
        self,
        image_suffix="*.png",
        label_suffix="*.png",
        image_mean=(0.474, 0.494, 0.450),
        image_std=(0.221, 0.204, 0.223),
        **kwargs,
    ):
        super(UAVidDataset, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )


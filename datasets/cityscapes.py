# coding=utf-8


import json
import os

import numpy as np

from datasets.base_datasets import SegmentationDataset
from tools.utils import find_files

__all__ = ["CityscapesDataset"]


class CityscapesDataset(SegmentationDataset):
    """
    Cityscapes is a large-scale database which focuses on semantic understanding of urban
    street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30
    classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions,
    objects, nature, sky, and void). The dataset consists of around 5000 fine annotated
    images and 20000 coarse annotated ones. Data was captured in 50 cities during several
    months, daytimes, and good weather conditions. It was originally recorded as video so
    the frames were manually selected to have the following features: large number of dynamic
    objects, varying scene layout, and varying background.

    reference:
    https://www.cityscapes-dataset.com/
    https://paperswithcode.com/dataset/cityscapes
    """

    CLASSES = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(
        self,
        image_suffix="*_leftImg8bit.png",
        label_suffix="*_gtFine_color.png",
        image_mean=(0.287, 0.325, 0.284),
        image_std=(0.187, 0.190, 0.187),
        **kwargs,
    ):
        super(CityscapesDataset, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

# coding=utf-8

import cv2
import numpy as np

from datasets.base_datasets import SegmentationDataset

__all__ = ["PascalContextDataset"]


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class PascalContextDataset(SegmentationDataset):
    """
    The PASCAL Context dataset is an extension of the PASCAL VOC 2010 detection challenge, and it
    contains pixel-wise labels for all training images. It contains more than 400 classes (including
    the original 20 classes plus backgrounds from PASCAL VOC segmentation), divided into three categories
    (objects, stuff, and hybrids). Many of the object categories of this dataset are too sparse and;
    therefore, a subset of 59 frequent classes are usually selected for use.

    reference:
    https://paperswithcode.com/dataset/pascal-context
    https://cs.stanford.edu/~roozbeh/pascal-context/
    """

    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
        "bag",
        "bed",
        "bench",
        "book",
        "building",
        "cabinet",
        "ceiling",
        "cloth",
        "computer",
        "cup",
        "door",
        "fence",
        "floor",
        "flower",
        "food",
        "grass",
        "ground",
        "keyboard",
        "light",
        "mountain",
        "mouse",
        "curtain",
        "platform",
        "sign",
        "plate",
        "road",
        "rock",
        "shelves",
        "sidewalk",
        "sky",
        "snow",
        "bedclothes",
        "track",
        "tree",
        "truck",
        "wall",
        "water",
        "window",
        "wood",
    )

    PALETTE = []

    def __init__(
        self,
        image_suffix="*.jpg",
        label_suffix="*.png",
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.22),
        **kwargs,
    ):
        self.PALETTE = color_map(N=len(self.CLASSES))

        super(PascalContextDataset, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

    def encode_label(self, label):
        """
        Convert the label image into a trainable label according to the index value of RGB of each class.

        Params:
            label: 3-D numpy array. RGB images to be encoded. Note that the ignored category is encoded as 255

        Return:
            return the encoded 2-D label
        """
        return label

    def load_label(self, path):
        """load label by opencv"""
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # One channel only
        return label

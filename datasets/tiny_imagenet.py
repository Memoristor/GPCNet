# coding=utf-8

import os

from datasets.base_datasets import ClassificationDataset

__all__ = ["TinyImageNetDataset"]


class TinyImageNetDataset(ClassificationDataset):
    """
    Tiny ImageNet contains 100000 images of 200 classes (500 for each class) downsized to 64×64 colored
    images. Each class has 500 training images, 50 validation images and 50 test images.

    Reference:
    https://paperswithcode.com/dataset/tiny-imagenet
    https://tiny-imagenet.herokuapp.com/
    """

    CLASSES = ()

    def __init__(
        self,
        wnidx_path="wnids.txt",
        image_suffix="*.JPEG",
        image_mean=(0.480, 0.448, 0.397),
        image_std=(0.267, 0.269, 0.282),
        **kwargs,
    ):
        super(TinyImageNetDataset, self).__init__(
            image_suffix=image_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

        # Get CLASSES
        wnidx_path = os.path.join(self.root_path, wnidx_path)
        with open(os.path.expanduser(wnidx_path), "r") as f:
            self.CLASSES = tuple([line.strip() for line in f.readlines()])

        # get the number of classes
        self.num_classes = len(self.CLASSES)

    def load_label(self, path):
        """load label"""
        basename = os.path.basename(os.path.dirname(os.path.dirname(path)))
        label = self.CLASSES.index(basename)
        assert label >= 0 and label < len(self.CLASSES)
        return label

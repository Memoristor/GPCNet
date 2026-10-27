# coding=utf-8

import os

from datasets.base_datasets import ClassificationDataset

__all__ = ["ImageNetDataset"]


class ImageNetDataset(ClassificationDataset):
    """
    The ImageNet dataset contains 14,197,122 annotated images according to the WordNet hierarchy.
    Since 2010 the dataset is used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC),
    a benchmark in image classification and object detection. The publicly released dataset contains
    a set of manually annotated training images.

    Reference:
    https://paperswithcode.com/dataset/imagenet
    https://www.image-net.org/
    """

    CLASSES = ()

    def __init__(
        self,
        wnidx_path="wnids.txt",
        image_suffix="*.JPEG",
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        super(ImageNetDataset, self).__init__(
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
        basename = os.path.basename(os.path.dirname(path))
        label = self.CLASSES.index(basename)
        assert label >= 0 and label < len(self.CLASSES)
        return label

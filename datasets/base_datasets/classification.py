# coding=utf-8

import os
from typing import Callable, Optional

import cv2
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset

from tools.utils import find_files

__all__ = ["ClassificationDataset"]


class ClassificationDataset(Dataset):
    """
    Basic dataset for image classification. An example of file structure is as followings.

    .. data tree

        ├── ....
        │   ├── dataset_name
        │   │   ├── train
        │   │   │   ├── class_1
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── class_2
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   ├── val
        │   │   │   ├── class_1
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}

    Params:
        root_path: str. The root path of the dataset folder.
        transforms: callable. The augmentors for images.
        test_mode: bool. Labels will not be loaded if `test_mode` is true. Default: false
        image_dir: str. Image directory of `root_path`. Default: 'images'
        image_suffix str. Suffix of images. Default: '.jpg'
        image_mean: list. The RGB average value of whole dataset. Default (0.485, 0.456, 0.406)
        image_std: list. The RGB standard deviation value of whole dataset. Default (0.229, 0.224, 0.225)
        div_std: bool. Whether the image data will be divided by `std`. Default True
        chw_format: bool. If True, the image shape format is CxHxW, otherwise HxWxC. Default True
    """

    CLASSES = ()

    def __init__(
        self,
        root_path: str,
        transforms: Optional[Callable] = None,
        test_mode=False,
        image_dir="images",
        image_suffix=".jpg",
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        div_std=True,
        chw_format=True,
    ):
        self.root_path = os.path.expanduser(root_path)
        self.transforms = transforms
        self.test_mode = test_mode
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.image_mean = image_mean
        self.image_std = image_std
        self.div_std = div_std
        self.chw_format = chw_format

        # Load images
        self.image_path = find_files(
            directory=os.path.join(self.root_path, self.image_dir),
            pattern="*" + self.image_suffix,
        )
        self.image_path = sorted(list(self.image_path))

        # get the number of classes
        self.num_classes = len(self.CLASSES)

    def encode_label(self, label):
        """
        Convert the image label into a trainable onehot target.
        """
        result = np.zeros(self.num_classes, dtype=np.float32)
        result[label] = 1.0
        return result

    def load_image(self, path):
        """load image by opencv"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_label(self, path):
        """load label"""
        basename = os.path.basename(os.path.dirname(path))
        label = self.CLASSES.index(basename)
        assert label >= 0 and label < len(self.CLASSES)
        return label

    def __getitem__(self, index):
        """Get item by index"""
        if not self.test_mode:
            # Load image and label
            image_path = self.image_path[index]
            image = self.load_image(image_path)
            label = self.load_label(image_path)
            target = self.encode_label(label)

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Data augmentation
            if self.transforms is not None:
                image = self.transforms(image)

            # Normalization
            image = image / 255.0
            image = image - self.image_mean
            if self.div_std:
                image = image / self.image_std

            # Channel transpose
            if len(image.shape) == 3 and self.chw_format:
                image = image.transpose([2, 0, 1])

            return {
                "image": np.array(image, dtype=np.float32),
                "label": np.array(label, dtype=np.int64),
                "target": np.array(target, dtype=np.float32),
                "size": np.array(size, dtype=np.int32),
                "index": np.array(index, dtype=np.int32),
            }
        else:
            # Open image
            image_path = self.image_path[index]
            image = self.load_image(image_path)

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Data augmentation
            if self.transforms is not None:
                image = self.transforms(image)

            # Normalization
            image = image / 255.0
            image = image - self.image_mean
            if self.div_std:
                image = image / self.image_std

            # Channel transpose
            if len(image.shape) == 3 and self.chw_format:
                image = image.transpose([2, 0, 1])

            return {
                "image": np.array(image, dtype=np.float32),
                "size": np.array(size, dtype=np.int32),
                "index": np.array(index, dtype=np.int32),
            }

    def __len__(self):
        return len(self.image_path)

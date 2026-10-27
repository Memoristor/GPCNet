# coding=utf-8

import os
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from tools.utils import find_files

__all__ = ["SegmentationDataset"]


class SegmentationDataset(Dataset):
    """
    Basic dataset for semantic segmentation. An example of file structure is as followings.

    .. data tree

        ├── ....
        │   ├── dataset_name
        │   │   ├── image
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── label
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    Params:
        root_path: str. The root path of the dataset folder.
        transforms: callable. The augmentors for images and labels.
        test_mode: bool. Labels will not be loaded if `test_mode` is true. Default: false
        image_dir: str. Image directory of `root_path`. Default: 'images'
        image_suffix str. Suffix of images. Default: '.jpg'
        label_dir: str. Label directory of `root_path`. Default: 'labels'
        label_suffix str. Suffix of labels. Default: '.jpg'
        crop_grids: tuple. The number of horizontal and vertical grids for cropping the image.
        grid_size: tuple. The size of the cropped image (height, width). If None, the original image size will be used.
        image_mean: list. The RGB average value of whole dataset. Default (0.485, 0.456, 0.406)
        image_std: list. The RGB standard deviation value of whole dataset. Default (0.229, 0.224, 0.225)
        div_std: bool. Whether the image data will be divided by `std`. Default True
        chw_format: bool. If True, the image shape format is CxHxW, otherwise HxWxC. Default True
    """

    CLASSES = ()
    PALETTE = []

    def __init__(
        self,
        root_path: str,
        transforms: Optional[Callable] = None,
        test_mode=False,
        image_dir="images",
        image_suffix=".jpg",
        label_dir="labels",
        label_suffix=".jpg",
        crop_grids=(1, 1),
        grid_size=None,
        image_mean=(0.485, 0.456, 0.406),  # for imagenet
        image_std=(0.229, 0.224, 0.225),  # for imagenet
        div_std=True,
        chw_format=True,
    ):
        self.root_path = os.path.expanduser(root_path)
        self.transforms = transforms
        self.test_mode = test_mode
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.crop_grids = crop_grids
        self.grid_size = grid_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.div_std = div_std
        self.chw_format = chw_format

        # Load images and labels
        image_finder = find_files(os.path.join(self.root_path, self.image_dir), pattern="*" + self.image_suffix)
        self.image_path = sorted(list(image_finder))

        if not self.test_mode:
            label_finder = find_files(os.path.join(self.root_path, self.label_dir), pattern="*" + self.label_suffix)
            self.label_path = sorted(list(label_finder))
            # assert len(self.image_path) == len(self.label_path)
        else:
            self.label_path = list()

        # get the number of classes
        self.num_classes = len(self.CLASSES)

    def encode_label(self, label: np.ndarray):
        """Convert an RGB label image to a class-index map (ignore index = 255)."""
        r = label[:, :, 0].astype(np.int32)
        g = label[:, :, 1].astype(np.int32)
        b = label[:, :, 2].astype(np.int32)
        label_key = r * 65536 + g * 256 + b

        result = np.full(label.shape[:2], 255, dtype=np.int16)
        for i, color in enumerate(self.PALETTE):
            palette_key = color[0] * 65536 + color[1] * 256 + color[2]
            result[label_key == palette_key] = i
        return result

    def decode_label(self, label):
        """
        Convert the predict label into a label image according to the index value of RGB of each class.

        Params:
            label: 2-D numpy array. Label to de decoded. Note that the ignored category will be decoded as 255

        Return:
            return the decoded 3-D label
        """
        result = np.ones((*label.shape[0:2], 3), dtype=np.uint8) * 255
        for i, cls in enumerate(self.PALETTE):
            result[label == i, :] = self.PALETTE[i]
        return result

    def load_image(self, path):
        """load image by opencv"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_label(self, path):
        """load label by opencv"""
        label = cv2.imread(path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        return label

    def get_path_index(self, index):
        """Get the path index for the given index.

        Params:
            index: int. The index of the patch.

        Returns:
            path_idx: int. The index of the image path.
        """
        path_idx = index // (self.crop_grids[0] * self.crop_grids[1])
        return path_idx

    def index_to_position(self, index, image_size):
        """
        Given an index, compute the crop (top, left, height, width) directly.

        Params:
            index: int. The index of the patch.
            image_size: (H, W) of original image.

        Returns:
            (top, left, ch, cw): crop region in image.
        """
        H, W = image_size
        gh, gw = self.crop_grids
        ch, cw = self.grid_size if self.grid_size is not None else (H // gh, W // gw)
        # Calculate grid position
        row = (index % (gh * gw)) // gw
        col = (index % (gh * gw)) % gw
        # Compute offset
        oh = (H - gh * ch) / (gh - 1) if gh > 1 else 0
        ow = (W - gw * cw) / (gw - 1) if gw > 1 else 0
        assert oh == int(oh) and ow == int(ow), "invalid grid setting"
        top = int(row * ch + row * oh)
        left = int(col * cw + col * ow)
        return (top, left, ch, cw)

    def __getitem__(self, index):
        """Get item by index"""
        if not self.test_mode:
            # Get the path index
            path_index = self.get_path_index(index)

            # Load original image and label
            image = self.load_image(self.image_path[path_index])
            label = self.load_label(self.label_path[path_index])

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Get the grid position
            top, left, h, w = self.index_to_position(index, size)

            # Crop the image and label
            image = image[top : top + h, left : left + w]
            label = label[top : top + h, left : left + w]

            # Encode label
            label = self.encode_label(label)

            # Data augmentation
            if self.transforms is not None:
                image, label = self.transforms(image, label)

            # Normalization
            image = image / 255.0
            image = image - self.image_mean
            if self.div_std:
                image = image / self.image_std

            # Channel transpose
            if len(image.shape) == 3 and self.chw_format:
                image = image.transpose([2, 0, 1])

            if len(label.shape) == 3 and self.chw_format:
                label = label.transpose([2, 0, 1])

            return {
                "image": np.array(image, dtype=np.float32),
                "label": np.array(label, dtype=np.int64),
                "size": np.array(size, dtype=np.int32),
                "index": np.array(index, dtype=np.int32),
            }
        else:
            # Get the path index
            path_index = self.get_path_index(index)

            # Load original image and label
            image = self.load_image(self.image_path[path_index])

            assert len(image.shape) in [2, 3]
            size = image.shape[0:2]

            # Get the grid position
            top, left, h, w = self.index_to_position(index, size)

            # Crop the image and label
            image = image[top : top + h, left : left + w]

            # Data augmentation
            if self.transforms is not None:
                image, label = self.transforms(image, None)

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
        return len(self.image_path) * self.crop_grids[0] * self.crop_grids[1]

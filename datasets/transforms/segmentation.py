# coding=utf-8

from typing import Tuple

import albumentations as A
import cv2

__all__ = [
    "random_scale_crop_hflip",
    "resize_only",
]


class SegmentationTransform:
    def __init__(self, augments):
        self.augments = augments

    def __call__(self, image, label):
        output = self.augments(image=image, mask=label)
        image = output["image"]
        label = output["mask"]
        return image, label

    def __str__(self):
        return str(self.augments)


def random_scale_crop_hflip(
    image_size: Tuple[int, int],
    random_scale_prob: float = 1.0,
    random_scale_limit: Tuple[float, float] = (-0.5, 1.0),
    horizontal_flip_prob: float = 0.5,
):
    """Transforms which consist of random scale, random crop and random horizontal flip

    Params:
        image_size (int, int): The output image size.
        random_scale_prob (float): Probability of applying random scale.
        random_scale_limit (float, float): Scaling factor range of random scale.
        horizontal_flip_prob (float): Probability of applying horizontal flip.

    Returns:
        The `SegmentationTransform` object.

    Examples:
        >>> transforms = random_scale_crop_hflip()
        >>> image, label = transforms(image, label)
    """
    augments = A.Compose(
        [
            A.RandomScale(
                p=random_scale_prob,
                scale_limit=random_scale_limit,
                interpolation=cv2.INTER_NEAREST,
            ),
            A.PadIfNeeded(
                p=1.0,
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                value=cv2.BORDER_CONSTANT,
                mask_value=255,
            ),
            A.RandomCrop(p=1.0, height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=horizontal_flip_prob),
        ]
    )
    return SegmentationTransform(augments)


def resize_only(image_size: Tuple[int, int]):
    """Transforms that  consist of resize only.

    Params:
        image_size (int, int): The output image size.

    Returns:
        The `SegmentationTransform` object.

    Examples:
        >>> transforms = resize_only()
        >>> image, label = transforms(image, label)"""
    augments = A.Resize(p=1, height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST)
    return SegmentationTransform(augments)

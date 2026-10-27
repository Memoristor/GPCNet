# coding=utf-8

from typing import Tuple

import albumentations as A
import cv2

__all__ = [
    "random_resizedcrop_hflip",
    "resize_centercrop",
    "pad_random_crop_hflip",
    "image_resize_only",
]


class ClassificationTransform:
    def __init__(self, augments):
        self.augments = augments

    def __call__(self, image):
        output = self.augments(image=image)
        image = output["image"]
        return image

    def __str__(self):
        return str(self.augments)


def random_resizedcrop_hflip(
    image_size: Tuple[int, int],
    random_resizedcrop_prob: float = 1.0,
    random_resizedcrop_scale: Tuple[float, float] = (0.1, 1.0),
    random_resizedcrop_ratio: Tuple[float, float] = (0.8, 1.25),
    horizontal_flip_prob: float = 0.5,
):
    """Transforms which consist of random resized crop, and random horizontal flip

    Params:
        image_size (int, int): The output image size.
        random_resizedcrop_prob (float): Probability of applying random resized crop.
        random_resizedcrop_scale (float, float): Range of size of the origin size cropped.
        random_resizedcrop_ratio (float, float): range of aspect ratio of the origin aspect ratio cropped.
        horizontal_flip_prob (float): Probability of applying horizontal flip.

    Returns:
        The `ClassificationTransform` object.

    Examples:
        >>> transforms = random_resizedcrop_hflip(image_size=(224, 224))
        >>> image = transforms(image)
    """
    augments = A.Compose(
        [
            A.RandomResizedCrop(
                p=random_resizedcrop_prob,
                height=image_size[0],
                width=image_size[1],
                scale=random_resizedcrop_scale,
                ratio=random_resizedcrop_ratio,
                interpolation=cv2.INTER_CUBIC,
            ),
            A.HorizontalFlip(p=horizontal_flip_prob),
        ]
    )
    return ClassificationTransform(augments)


def resize_centercrop(
    image_size: Tuple[int, int],
    resized_image_size: Tuple[int, int] = (256, 256),
):
    """Transforms which consist of resize and center crop.

    Params:
        image_size (int, int): The output image size.
        resized_image_size (int, int): Desired height and width of the resized image.

    Returns:
        The `ClassificationTransform` object.

    Examples:
        >>> transforms = resize_centercrop()
        >>> image = transforms(image)
    """
    augments = A.Compose(
        [
            A.Resize(
                p=1,
                height=resized_image_size[0],
                width=resized_image_size[1],
                interpolation=cv2.INTER_NEAREST,
            ),
            A.CenterCrop(
                p=1,
                height=image_size[0],
                width=image_size[1],
            ),
        ]
    )
    return ClassificationTransform(augments)


def pad_random_crop_hflip(
    image_size: Tuple[int, int],
    padding_size: Tuple[int, int] = (4, 4),
    horizontal_flip_prob: float = 0.5,
):
    """Transforms which consist of pixel padding, random crop and horizontal flip.

    Params:
        image_size (int, int): The output image size.
        padding_size (int, int): The pixel size to extend along both sides in height and width.
        horizontal_flip_prob (float): Probability of applying horizontal flip.

    Returns:
        The `ClassificationTransform` object.

    Examples:
        >>> transforms = pad_random_crop_hflip()
        >>> image = transforms(image)
    """
    augments = A.Compose(
        [
            A.PadIfNeeded(
                p=1,
                min_height=image_size[0] + padding_size[0] * 2,
                min_width=image_size[1] + padding_size[1] * 2,
                border_mode=cv2.BORDER_WRAP,
            ),
            A.RandomCrop(
                p=1,
                height=image_size[0],
                width=image_size[1],
            ),
            A.HorizontalFlip(p=horizontal_flip_prob),
        ]
    )
    return ClassificationTransform(augments)


def image_resize_only(image_size: Tuple[int, int]):
    """Transforms that  consist of resize only.

    Params:
        image_size (int, int): The output image size.

    Returns:
        The `ClassificationTransform` object.

    Examples:
        >>> transforms = resize_only()
        >>> image, label = transforms(image, label)"""
    augments = A.Resize(
        p=1,
        height=image_size[0],
        width=image_size[1],
        interpolation=cv2.INTER_NEAREST,
    )
    return ClassificationTransform(augments)

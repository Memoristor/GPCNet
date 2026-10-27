# coding=utf-8

import numpy as np
from tqdm import tqdm

__all__ = [
    "get_image_mean_std",
    "get_class_proportion",
]


def get_image_mean_std(dataset):
    """
    Get the image mean and std value of the dataset.

    let u denote the `image_mean` of the dataset, while the s denote the `image_std`
    of the dataset, and E is the expectation function, i.e.
    Y = (X - u) / s

    E(Y) = (E(X) - u) / s
    S(Y) = E(Y - E(Y)) = E((X - u) / s - (E(X) - u) / s)
         = E(X - E(x)) / s
         = S(X) / s

    so the real `image_mean` and `image_std` are

    E(X) = E(Y) * s + u
    S(X) = S(Y) * s

    Params:
        dataset (CustomDataset): The dataset.
    """
    # average value
    rgb = np.zeros((3,), dtype=np.float32)
    pixels = 0

    mean_bar = tqdm(range(len(dataset)))
    for i in mean_bar:
        mean_bar.set_description("Mean value of RGB")
        data = dataset[i]
        image = data["image"]
        size = image.shape

        # sum pixels
        if dataset.chw_format:
            rgb += image.sum(
                axis=(
                    1,
                    2,
                )
            )
            pixels += size[1] * size[2]
        else:
            rgb += image.sum(
                axis=(
                    0,
                    1,
                )
            )
            pixels += size[0] * size[1]

    rgb_mean = rgb / pixels

    # standard deviation
    rgb = np.zeros((3,), dtype=np.float32)
    pixels = 0

    std_bar = tqdm(range(len(dataset)))
    for i in std_bar:
        std_bar.set_description("Std value of RGB")
        data = dataset[i]
        image = data["image"]
        size = image.shape

        # sum pixels
        if dataset.chw_format:
            rgb += ((image.transpose(1, 2, 0) - rgb_mean) ** 2).sum(
                axis=(
                    0,
                    1,
                )
            )
            pixels += size[1] * size[2]
        else:
            rgb += ((image - rgb_mean) ** 2).sum(
                axis=(
                    0,
                    1,
                )
            )
            pixels += size[0] * size[1]

    rgb_std = (rgb / pixels) ** 0.5

    if dataset.div_std:
        image_mean = rgb_mean * dataset.image_std + dataset.image_mean
        image_std = rgb_std * dataset.image_std
    else:
        image_mean = rgb_mean + dataset.image_mean
        image_std = rgb_std

    return image_mean, image_std


def get_class_proportion(dataset):
    """
    Get class proportion

    Params:
        dataset (CustomDataset): The dataset.
    """
    proportion = np.zeros((len(dataset.PALETTE),), dtype=np.float32)

    bar = tqdm(range(len(dataset)))
    for i in bar:
        bar.set_description("Class proportion")
        data = dataset[i]
        label = data["label"]

        for p in range(len(dataset.PALETTE)):
            proportion[p] += (label == p).astype(np.uint8).sum()

    proportion /= proportion.sum()
    return proportion

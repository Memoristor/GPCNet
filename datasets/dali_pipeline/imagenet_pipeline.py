# coding=utf-8

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import auto_augment, rand_augment
from nvidia.dali.pipeline import pipeline_def
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = [
    "imagenet_training_pipeline",
    "imagenet_training_autoaug_pipeline",
    "imagenet_training_randaug_pipeline",
    "imagenet_validation_pipeline",
]


@pipeline_def
def imagenet_training_pipeline(
    files,
    labels,
    num_shards=1,
    shard_id=0,
    random_seed=0,
    random_shuffle=False,
    image_size=(224, 224),
    random_aspect_ratio=[0.8, 1.25],
    random_area=[0.1, 1.0],
    hflip_prob=0.5,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """
    Defines a DALI pipeline for ImageNet dataset preprocessing and augmentation during training.

    Params:
        files: List of image file paths
        labels: Corresponding image index labels
        num_shards: Total number of parallel shards for distributed training (default: 1)
        shard_id: Index of the current shard in distributed training (default: 0)
        random_seed: Seed value for random operations (default: 0)
        random_shuffle: Whether to shuffle the dataset (default: False)
        image_size: Target output image size (H, W) (default: (224, 224))
        random_aspect_ratio: Range for random aspect ratio augmentation [min, max] (default: [0.8, 1.25])
        random_area: Range for random area scaling before crop [min_scale, max_scale] (default: [0.1, 1.0])
        hflip_prob: Probability of applying horizontal flip (default: 0.5)
        mean: Normalization mean values for RGB channels (default: IMAGENET_DEFAULT_MEAN)
        std: Normalization standard deviation values for RGB channels (default: IMAGENET_DEFAULT_STD)

    Returns:
        DALI pipeline object containing:
        - Decoded and augmented images
        - Corresponding preprocessed labels
    """
    jpegs, labels = fn.readers.file(
        files=files,
        labels=labels,
        random_shuffle=random_shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        pad_last_batch=True,
        name=f"Reader{shard_id}",
        seed=random_seed,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=image_size,
        dtype=types.FLOAT,
        random_aspect_ratio=random_aspect_ratio,
        random_area=random_area,
        num_attempts=100,
    )
    mirror_coin = fn.random.coin_flip(probability=hflip_prob)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=image_size,
        mean=(np.array(mean) * 255).tolist(),
        std=(np.array(std) * 255).tolist(),
        mirror=mirror_coin,
    )
    return images, labels


@pipeline_def(enable_conditionals=True)
def imagenet_training_autoaug_pipeline(
    files,
    labels,
    num_shards=1,
    shard_id=0,
    random_seed=0,
    random_shuffle=False,
    image_size=(224, 224),
    random_aspect_ratio=[0.8, 1.25],
    random_area=[0.1, 1.0],
    hflip_prob=0.5,
    re_prob=0.25,
    re_axis_names="HW",
    re_nregions=1,
    re_norm_anchor_range=(0.0, 1.0),
    re_norm_shape_range=(0.3, 0.7),
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """Defines a DALI pipeline for ImageNet training with AutoAugment-style preprocessing and Random Erasing.

    Params:
        files: List of image file paths
        labels: Corresponding image labels
        num_shards: Number of pipeline shards for distributed training (default: 1)
        shard_id: Shard ID for current pipeline instance (default: 0)
        random_seed: Seed for random number generation (default: 0)
        random_shuffle: Whether to shuffle images (default: False)
        image_size: Target output image dimensions (H, W) (default: (224, 224))
        random_aspect_ratio: Range for aspect ratio adjustment in random resized crop (default: [0.8, 1.25])
        random_area: Range for area selection in random resized crop (default: [0.1, 1.0])
        hflip_prob: Probability of horizontal flip (default: 0.5)
        re_prob: Probability of applying Random Erasing (default: 0.25)
        re_axis_names: Axes for Random Erasing regions (default: "HW")
        re_nregions: Number of regions to erase per image (default: 1)
        re_norm_anchor_range: Normalized coordinate range [min, max] for erase region anchors (default: (0.0, 1.0))
        re_norm_shape_range: Normalized size range [min, max] for erase region dimensions (default: (0.3, 0.7))
        mean: Normalization mean values for RGB channels (default: IMAGENET_DEFAULT_MEAN)
        std: Normalization standard deviation values for RGB channels (default: IMAGENET_DEFAULT_STD)

    Returns:
        DALI pipeline object containing:
        - Decoded and augmented images
        - Corresponding preprocessed labels
    """
    jpegs, labels = fn.readers.file(
        files=files,
        labels=labels,
        random_shuffle=random_shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        pad_last_batch=True,
        name=f"Reader{shard_id}",
        seed=random_seed,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=image_size,
        dtype=types.UINT8,
        random_aspect_ratio=random_aspect_ratio,
        random_area=random_area,
        num_attempts=100,
    )
    images = auto_augment.auto_augment(images, shape=image_size)
    random_anchor = fn.random.uniform(
        range=re_norm_anchor_range,
        shape=(re_nregions * len(re_axis_names),),
    )
    random_shape = fn.random.uniform(
        range=re_norm_shape_range,
        shape=(re_nregions * len(re_axis_names),),
    )
    if re_prob > 0:
        erase_coin = fn.random.coin_flip(probability=re_prob)
        images = fn.erase(
            images,
            anchor=random_anchor,
            shape=random_shape,
            axis_names=re_axis_names,
            normalized_anchor=True,
            normalized_shape=True,
        ) * erase_coin + images * (1 - erase_coin)
    mirror_coin = fn.random.coin_flip(probability=hflip_prob)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=image_size,
        mean=(np.array(mean) * 255).tolist(),
        std=(np.array(std) * 255).tolist(),
        mirror=mirror_coin,
    )
    return images, labels


@pipeline_def(enable_conditionals=True)
def imagenet_training_randaug_pipeline(
    files,
    labels,
    num_shards=1,
    shard_id=0,
    random_seed=0,
    random_shuffle=False,
    image_size=(224, 224),
    random_aspect_ratio=[0.8, 1.25],
    random_area=[0.1, 1.0],
    hflip_prob=0.5,
    randaug_n=2,
    randaug_m=9,
    monotonic_mag=True,
    re_prob=0.25,
    re_axis_names="HW",
    re_nregions=1,
    re_norm_anchor_range=(0.0, 1.0),
    re_norm_shape_range=(0.3, 0.7),
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """Defines a DALI pipeline for ImageNet training with RandAugment and Random Erasing.

    Params:
        files: List of image file paths
        labels: Corresponding image labels
        num_shards: Number of pipeline shards for distributed training (default: 1)
        shard_id: Shard ID for current pipeline instance (default: 0)
        random_seed: Seed for random number generation (default: 0)
        random_shuffle: Whether to shuffle images (default: False)
        image_size: Target output image dimensions (H, W) (default: (224, 224))
        random_aspect_ratio: Aspect ratio range for random resized crop (default: [0.8, 1.25])
        random_area: Area selection range for random resized crop (default: [0.1, 1.0])
        hflip_prob: Probability of horizontal flip (default: 0.5)
        randaug_n: Number of augmentation transformations to apply (default: 2)
        randaug_m: Magnitude level for RandAugment operations (default: 9)
        monotonic_mag: Use monotonically increasing magnitude per epoch (default: True)
        re_prob: Probability of applying Random Erasing (default: 0.25)
        re_axis_names: Axes for Random Erasing regions (default: "HW")
        re_nregions: Number of regions to erase per image (default: 1)
        re_norm_anchor_range: Normalized coordinate range [min, max] for erase region anchors (default: (0.0, 1.0))
        re_norm_shape_range: Normalized size range [min, max] for erase region dimensions (default: (0.3, 0.7))
        mean: Normalization mean values for RGB channels (default: IMAGENET_DEFAULT_MEAN)
        std: Normalization standard deviation values for RGB channels (default: IMAGENET_DEFAULT_STD)

    Returns:
        DALI pipeline object containing:
        - Decoded and augmented images
        - Corresponding preprocessed labels
    """
    jpegs, labels = fn.readers.file(
        files=files,
        labels=labels,
        random_shuffle=random_shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        pad_last_batch=True,
        name=f"Reader{shard_id}",
        seed=random_seed,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=image_size,
        dtype=types.UINT8,
        random_aspect_ratio=random_aspect_ratio,
        random_area=random_area,
        num_attempts=100,
    )
    images = rand_augment.rand_augment(
        images,
        shape=image_size,
        n=randaug_n,
        m=randaug_m,
        monotonic_mag=monotonic_mag,
    )
    random_anchor = fn.random.uniform(
        range=re_norm_anchor_range,
        shape=(re_nregions * len(re_axis_names),),
    )
    random_shape = fn.random.uniform(
        range=re_norm_shape_range,
        shape=(re_nregions * len(re_axis_names),),
    )
    erase_coin = fn.random.coin_flip(probability=re_prob)
    if re_prob > 0:
        images = fn.erase(
            images,
            anchor=random_anchor,
            shape=random_shape,
            axis_names=re_axis_names,
            normalized_anchor=True,
            normalized_shape=True,
        ) * erase_coin + images * (1 - erase_coin)
    mirror_coin = fn.random.coin_flip(probability=hflip_prob)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=image_size,
        mean=(np.array(mean) * 255).tolist(),
        std=(np.array(std) * 255).tolist(),
        mirror=mirror_coin,
    )
    return images, labels


@pipeline_def
def imagenet_validation_pipeline(
    files,
    labels,
    num_shards=1,
    shard_id=0,
    random_seed=0,
    random_shuffle=False,
    resize_image_size=(256, 256),
    image_size=(224, 224),
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    """Defines a DALI pipeline for ImageNet validation data preprocessing.

    Params:
        files: List of image file paths
        labels: Corresponding image labels
        num_shards: Number of pipeline shards for distributed evaluation (default: 1)
        shard_id: Shard ID for current pipeline instance (default: 0)
        random_seed: Seed for reproducible data ordering (default: 0)
        random_shuffle: Whether to shuffle images (not recommended for validation) (default: False)
        resize_image_size: Initial image dimensions for resizing (H, W) (default: (256, 256))
        image_size: Final output image dimensions after center crop (H, W) (default: (224, 224))
        mean: Normalization mean values for RGB channels (default: IMAGENET_DEFAULT_MEAN)
        std: Normalization standard deviation values for RGB channels (default: IMAGENET_DEFAULT_STD)

    Returns:
        DALI pipeline object containing:
        - Decoded and cropped images
        - Corresponding preprocessed labels
    """
    jpegs, labels = fn.readers.file(
        files=files,
        labels=labels,
        random_shuffle=random_shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        pad_last_batch=True,
        name=f"Reader{shard_id}",
        seed=random_seed,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(
        images,
        size=resize_image_size,
        mode="not_smaller",
        interp_type=types.INTERP_TRIANGULAR,
    )
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=image_size,
        mean=(np.array(mean) * 255).tolist(),
        std=(np.array(std) * 255).tolist(),
        mirror=0,
    )
    return images, labels

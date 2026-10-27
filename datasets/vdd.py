# coding=utf-8


from datasets.base_datasets import SegmentationDataset

__all__ = ["VDD"]


class VDD(SegmentationDataset):
    """Semantic segmentation of drone images is critical for various aerial vision tasks
    as it provides essential semantic details to understand scenes on the ground.
    Ensuring high accuracy of semantic segmentation models for drones requires access to
    diverse, large-scale, and high-resolution datasets, which are often scarce in the
    field of aerial image processing. While existing datasets typically focus on urban
    scenes and are relatively small, our Varied Drone Dataset (VDD) addresses these
    limitations by offering a large-scale, densely labeled collection of 400 high-resolution
    images spanning 7 classes. This dataset features various scenes in urban, industrial,
    rural, and natural areas, captured from different camera angles and under diverse
    lighting conditions. We also make new annotations to UDD and UAVid, integrating them
    under VDD annotation standards, to create the Integrated Drone Dataset (IDD). We train
    seven state-of-the-art models on drone datasets as baselines. It's expected that our
    dataset will generate considerable interest in drone image segmentation and serve as
    a foundation for other drone vision tasks. Datasets are publicly available at our
    website{https://github.com/RussRobin/VDD}."""

    CLASSES = (
        "Others",
        "Wall",
        "Road",
        "Vegetation",
        "Vehicle",
        "Roof",
        "Water",
    )

    PALETTE = [
        [0, 0, 0],
        [102, 102, 153],
        [128, 64, 128],
        [107, 142, 35],
        [0, 0, 142],
        [70, 70, 70],
        [0, 255, 255],
    ]

    def __init__(
        self,
        image_suffix="*.JPG",
        label_suffix="*.png",
        image_mean=(0.486, 0.487, 0.441),
        image_std=(0.190, 0.178, 0.213),
        **kwargs,
    ):
        super(VDD, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

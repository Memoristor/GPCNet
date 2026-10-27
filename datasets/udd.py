# coding=utf-8


from datasets.base_datasets import SegmentationDataset

__all__ = ["UDD5"]


class UDD5(SegmentationDataset):
    """This is a collection of drone image Dataset collected at Peking University,
    Huludao city, Henan University and Cangzhou city."""

    CLASSES = (
        "Vegetation",
        "Building",
        "Road",
        "Vehicle",
        "Other",
    )

    PALETTE = [
        [107, 142, 35],
        [102, 102, 156],
        [128, 64, 128],
        [0, 0, 142],
        [0, 0, 0],
    ]

    def __init__(
        self,
        image_suffix="*.JPG",
        label_suffix="*.png",
        **kwargs,
    ):
        super(UDD5, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            **kwargs,
        )


class UDD6(SegmentationDataset):
    """This is a collection of drone image Dataset collected at Peking University,
    Huludao city, Henan University and Cangzhou city."""

    CLASSES = (
        "Other",
        "Facade",
        "Road",
        "Vegetation",
        "Vehicle",
        "Roof",
    )

    PALETTE = [
        [0, 0, 0],
        [102, 102, 156],
        [128, 64, 128],
        [107, 142, 35],
        [0, 0, 142],
        [70, 70, 70],
    ]

    def __init__(
        self,
        image_suffix="*.JPG",
        label_suffix="*.png",
        **kwargs,
    ):
        super(UDD6, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            **kwargs,
        )

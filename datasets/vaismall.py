# coding=utf-8

from datasets.base_datasets.segmentation import SegmentationDataset

__all__ = ["VaismallDataset"]


class VaismallDataset(SegmentationDataset):
    """
    Vaihigen data set contains 33 patches (of different sizes), each consisting of a true
    orthophoto (TOP) extracted from a larger TOP mosaic. For further information about the
    original input data, please refer to the data description of the object detection and
    3d reconstruction benchmark.

    The dataset contains 33 tiles with spatial size varying from 1996 x 1995 to 3816 x 2550.
    There are 16 tiles fully annotated for the training phase while other unannotated 17 tiles
    for the testing phase. We followed the dataset settings of Mou [6], splitting the fully
    annotated 16 tiles into 11 tiles as the training set while the remaining (tile ID 11, 15,
    28, 30 and 34) as the validation set.

    reference:
    https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx
    """

    CLASSES = (
        "Impervious surface",
        "Buildings",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background",
    )

    PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ]

    def __init__(
        self,
        image_dir="image",
        image_suffix=".png",
        label_dir="label",
        label_suffix=".png",
        image_mean=(0.469, 0.323, 0.318),
        image_std=(0.220, 0.159, 0.151),
        **kwargs,
    ):
        super(VaismallDataset, self).__init__(
            image_dir=image_dir,
            image_suffix=image_suffix,
            label_dir=label_dir,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

# coding=utf-8


from datasets.base_datasets import SegmentationDataset

__all__ = ["FloodNet"]


class FloodNet(SegmentationDataset):
    """The dataset is a high-resolution natural disaster dataset captured by small UAV platforms
    after Hurricane Harvey near Texas and Louisiana. The dataset provides three computer vision
    tasks, including image classification, semantic segmentation, and visual question answering (VQA),
    to assess flooded damages to the affected areas. We only utilize the semantic segmentation
    benchmark, which consists of 1445 training images, 450 validation images, and 448 test images,
    to validate the model performance.
    """

    CLASSES = (
        "Background",
        "Building-flooded",
        "Building-non-flooded",
        "Road-flooded",
        "Road-non-flooded",
        "Water",
        "Tree",
        "Vehicle",
        "Pool",
        "Grass",
    )

    PALETTE = [
        [0, 0, 0],
        [255, 71, 0],
        [180, 120, 120],
        [160, 150, 20],
        [140, 140, 140],
        [61, 230, 250],
        [0, 82, 255],
        [255, 0, 245],
        [255, 0, 0],
        [4, 250, 7],
    ]

    def __init__(
        self,
        image_suffix="*.png",
        label_suffix="*.png",
        image_mean=(0.409, 0.447, 0.340),
        image_std=(0.207, 0.193, 0.208),
        **kwargs,
    ):
        super(FloodNet, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )


# class FloodNetOld(FloodNet):
#     CLASSES = (
#         "Background",
#         "Building-flooded",
#         "Building-non-flooded",
#         "Road-flooded",
#         "Road-non-flooded",
#         "Water",
#         "Tree",
#         "Vehicle",
#         "Pool",
#         "Grass",
#     )

#     PALETTE = [
#         [0, 0, 0],
#         [255, 71, 0],
#         [180, 120, 120],
#         [160, 150, 20],
#         [140, 140, 140],
#         [61, 230, 250],
#         [0, 82, 255],
#         [255, 0, 245],
#         [255, 0, 0],
#         [4, 250, 7],
#     ]

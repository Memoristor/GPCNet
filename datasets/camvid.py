# coding=utf-8

from datasets.base_datasets import SegmentationDataset

__all__ = ["CamVidDataset"]


class CamVidDataset(SegmentationDataset):
    """
    CamVid (Cambridge-driving Labeled Video Database) is a road/driving scene understanding database which
    was originally captured as five video sequences with a 960x720 resolution camera mounted on the dashboard
    of a car. Those sequences were sampled (four of them at 1 fps and one at 15 fps) adding up to 701 frames.
    Those stills were manually annotated with 32 classes: void, building, wall, tree, vegetation, fence, sidewalk,
    parking block, column/pole, traffic cone, bridge, sign, miscellaneous text, traffic light, sky, tunnel,
    archway, road, road shoulder, lane markings (driving), lane markings (non-driving), animal, pedestrian,
    child, cart luggage, bicyclist, motorcycle, car, SUV/pickup/truck, truck/bus, train, and other moving object

    reference:
    https://paperswithcode.com/dataset/camvid
    http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
    """

    CLASSES = (
        "Bicyclist",
        "Building",
        "Car",
        "Column_Pole",
        "Fence",
        "Pedestrian",
        "Road",
        "Sidewalk",
        "SignSymbol",
        "Sky",
        "Tree",
    )

    PALETTE = [
        [0, 128, 192],
        [128, 0, 0],
        [64, 0, 128],
        [192, 192, 128],
        [64, 64, 128],
        [64, 64, 0],
        [128, 64, 128],
        [0, 0, 192],
        [192, 128, 128],
        [128, 128, 128],
        [128, 128, 0],
    ]

    def __init__(
        self,
        image_suffix="*.png",
        label_suffix="*.png",
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        super(CamVidDataset, self).__init__(
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

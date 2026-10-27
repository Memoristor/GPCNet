# coding=utf-8

import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import itertools

import cv2
from tqdm import tqdm

from datasets.base_datasets import SegmentationDataset


class BasicCropper(SegmentationDataset):
    """
    Basic module for dataset crop and dataset merge.
    """

    def __init__(
        self,
        root_path: str,
        output_path: str,
        image_dir="images",
        image_suffix=".jpg",
        label_dir="labels",
        label_suffix=".jpg",
    ):
        super(BasicCropper, self).__init__(
            root_path,
            test_mode=False,
            image_dir=image_dir,
            image_suffix=image_suffix,
            label_dir=label_dir,
            label_suffix=label_suffix,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
            div_std=False,
            chw_format=False,
        )

        self.output_path = output_path

    def calc_grids(self, H, W):
        pass

    def crop_dataset(self):
        pass

    def merge_dataset(self):
        pass


class GridsCropper(BasicCropper):
    """
    Crop and merge semantic segmentation dataset with overlapped/non-overlapped grids.
    """

    def __init__(
        self,
        root_path: str,
        output_path: str,
        image_dir="images",
        image_suffix=".jpg",
        label_dir="labels",
        label_suffix=".jpg",
        image_size=(512, 512),
        grid_size=(3, 3),
    ):
        super(GridsCropper, self).__init__(root_path, output_path, image_dir, image_suffix, label_dir, label_suffix)

        self.image_size = image_size
        self.grid_size = grid_size

    def calc_grids(self, H, W):
        ih, iw = self.image_size
        gh, gw = self.grid_size
        oh = (H - gh * ih) / (gh - 1)
        ow = (W - gw * iw) / (gw - 1)
        tops = [int(ih * i + oh * i) for i in range(gh)]
        lefts = [int(iw * i + ow * i) for i in range(gw)]
        return list(itertools.product(tops, lefts))

    def _crop(self, path):
        rel_path = os.path.relpath(path, dataset_path)
        save_root = os.path.dirname(os.path.join(output_path, rel_path))
        os.makedirs(save_root, exist_ok=True)

        file_name, file_suffix = os.path.basename(path).split(".")

        # image_data = cv2.imread(path)
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            image_data = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        for top, left in self.calc_grids(*image_data.shape[0:2]):
            ih, iw = self.image_size
            sub_image = image_data[top : top + ih, left : left + iw]
            class_name = self.__class__.__name__
            format_loc = f"{top}_{left}_{top + ih}_{left + iw}"
            cv2.imwrite(os.path.join(save_root, file_name + f"@{class_name}_{format_loc}.{file_suffix}"), sub_image)

    def crop_dataset(self):
        for p in tqdm(self.image_path):
            self._crop(p)
        if not self.test_mode:
            for p in tqdm(self.label_path):
                self._crop(p)


if __name__ == "__main__":
    #########################
    # Crop UAVid
    #########################

    # # dataset_path = "/media/dandri/T7/Datasets/UAVid/uavid_image/uavid_v1.5_official_release_image/uavid_v1.5_official_release_image"
    # dataset_path = "/media/dandri/T7/Datasets/UAVid/uavid_image/"
    # output_path = "/media/dandri/T7/Datasets/UAVid/uavid_image/uavid_image_grids_2x3_2048x2048"

    # cropper = GridsCropper(
    #     root_path=os.path.join(dataset_path, "test_gt"),
    #     output_path=output_path,
    #     image_dir="",
    #     image_suffix="seq*/Images/*.png",
    #     label_dir="",
    #     label_suffix="seq*/Labels/*.png",
    #     image_size=(2048, 2048),
    #     grid_size=(2, 3),
    # )

    # cropper.crop_dataset()

    #########################
    # Crop VDD
    #########################

    # dataset_path = "/media/dandri/T7/Datasets/VDD/VDD/"
    # output_path = "/media/dandri/T7/Datasets/VDD/vdd_image_grids_2x3_2048x2048"

    # for folder in ["train", "val", "test"]:
    #     cropper = GridsCropper(
    #         root_path=os.path.join(dataset_path, folder),
    #         output_path=output_path,
    #         image_dir="src",
    #         image_suffix=".JPG",
    #         label_dir="gt-dec",
    #         label_suffix=".png",
    #         image_size=(2048, 2048),
    #         grid_size=(2, 3),
    #     )

    #     cropper.crop_dataset()

    #########################
    # Crop FloodNet
    #########################

    dataset_path = "/media/dandri/T7/Datasets/FloodNet/"
    output_path = "/media/dandri/T7/Datasets/FloodNet/floodnet_image_grids_2x3_2048x2048"

    for folder in ["train", "val", "test"]:
        cropper = GridsCropper(
            root_path=os.path.join(dataset_path, folder),
            output_path=output_path,
            image_dir="org-img",
            image_suffix=".jpg",
            label_dir="label-img-dec",
            label_suffix="_lab.png",
            image_size=(2048, 2048),
            grid_size=(2, 3),
        )

        cropper.crop_dataset()

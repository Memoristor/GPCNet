# coding=utf-8

import os

import torch
from PIL import Image
from tqdm import tqdm

from tools import utils

from .basic_engine import BasicEngine

__all__ = [
    "BasicTest",
    "SegmentationTest",
]


class BasicTest(BasicEngine):
    """The engine is used in the testing phase."""

    def __init__(self, *args, **kwags):
        super(BasicTest, self).__init__(*args, **kwags)

    def handle_output(self, output, truth, dataset):
        """The function to handle outputs."""
        raise NotImplementedError("function `handle_output` has not been implemented.")

    def run(self):
        """Test model"""
        self.model.eval()

        if utils.local_rank == 0:
            self.logger.info(f"[Test] model: {self.model_class}, dataset: {self.config.dataset.cls}")

        # Build dataloader for tests
        test_dataloader = self.build_dataloader(self.phase)

        # Run test
        with torch.no_grad():
            bar = tqdm(test_dataloader)
            for iter, truth in enumerate(bar):
                for k, v in truth.items():
                    truth[k] = v.to(torch.device(utils.local_rank))

                # Get predict results
                output = self.model(truth["image"])

                # Set description
                bar.set_description(f"[Test] Model: {self.model_class}, Rank: {utils.local_rank}")

                # Handle output
                self.handle_output(
                    output=output,
                    truth=truth,
                    dataset=test_dataloader.dataset,
                )


class SegmentationTest(BasicTest):
    """
    The module is used for the testing phase.
    """

    def handle_output(self, output, truth, dataset):
        """
        A function to determine how to handle outputs.

        Params:
            output. dict. Model output.
            truth. dict. Ground truth.
        """
        for k in output.keys():
            if k == "logit":
                z = torch.argmax(output[k], dim=1)
                z = z.cpu().detach().numpy()

                truth_index = truth["index"].cpu().numpy()
                truth_size = truth["size"].cpu().numpy()
                for i in range(z.shape[0]):
                    z_idx = truth_index[i]
                    x_path = dataset.image_path[z_idx]
                    x_dir = os.path.join(dataset.root_path, dataset.image_dir)
                    z_path = os.path.join(self.save_result, self.phase, os.path.relpath(x_path, x_dir))
                    z_dir = os.path.dirname(z_path)
                    os.makedirs(z_dir, exist_ok=True)

                    z_dec = dataset.decode_label(z[i])
                    # if self.test_dataset.chw_format:
                    #     z_dec = z_dec.transpose((1, 2, 0)).astype(np.uint8)

                    x_size = truth_size[i]
                    dec_img = Image.fromarray(z_dec)
                    dec_img.resize((x_size[1], x_size[0]))
                    dec_img.save(z_path)

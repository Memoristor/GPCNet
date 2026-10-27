# coding=utf-8

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torch.nn import functional as F
from torch.utils.data import Dataset

__all__ = ["ImagenetPipeloader"]


class ImagenetPipeloader:
    def __init__(
        self,
        dataset: Dataset,
        pipeline: Pipeline,
        reader_name: str,
        last_batch_policy=LastBatchPolicy.FILL,
        auto_reset=True,
    ):
        self.dali_iterator = DALIClassificationIterator(
            pipeline,
            reader_name=reader_name,
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.dali_iterator)
            image = data[0]["data"]
            label = data[0]["label"].squeeze(1).long()
            target = F.one_hot(label, self.dataset.num_classes)
            return {
                "image": image,
                "label": label,
                "target": target,
            }
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dali_iterator)

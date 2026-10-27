# coding=utf-8


from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.utils.data import Dataset

__all__ = ["SemSegPipeloader"]


class SemSegPipeloader:
    def __init__(
        self,
        dataset: Dataset,
        pipeline: Pipeline,
        last_batch_policy=LastBatchPolicy.FILL,
        auto_reset=True,
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.dali_iterator = DALIGenericIterator(
            self.pipeline,
            ["image", "label", "size", "index"],
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.dali_iterator)
            image = data[0]["image"]
            label = data[0]["label"].long()
            size = data[0]["size"].long()
            index = data[0]["index"].long()
            return {"image": image, "label": label, "size": size, "index": index}
        except StopIteration:
            raise StopIteration

    def __len__(self):
        callbacks = self.pipeline._input_callbacks
        assert len(callbacks) == 1, "only one external source callback can be allowed"
        return len(callbacks[0].callback)

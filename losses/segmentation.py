# coding=utf-8

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CrossEntropy2D",
    "BinaryCrossEntropy2D",
    "ImageBasedCrossEntropy2D",
    "BoundariesRelaxation2D",
    "BinaryBoundariesRelaxation2D",
    "OHEMCrossEntropy2D",
    "DiceLoss2D",
]


class BasicLossModule(nn.Module):
    """
    Basic loss module, please do not call this module before implementing `forward(...)`

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: 1D numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)
        upper_bound: retained for compatibility with the original weighting method
        alpha: float (default 1.0). Nonnegative strength of logarithmic class weighting
    """

    def __init__(
        self,
        ignore_index=255,
        custom_weight=None,
        batch_weight=True,
        size_average=True,
        batch_average=True,
        alpha=1.0,
    ):
        super(BasicLossModule, self).__init__()
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")

        # Init custom weight
        if custom_weight is not None:
            if isinstance(custom_weight, list):
                custom_weight = torch.from_numpy(np.array(custom_weight, dtype=np.float32)).float()
            else:
                custom_weight = torch.from_numpy(custom_weight).float()

        # Add to properties
        self.ignore_index = ignore_index
        self.custom_weight = custom_weight
        self.batch_weight = batch_weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.alpha = alpha

    def calculate_weights(self, target, num_classes):
        """Return 1 - alpha * log(class frequency) for classes in the batch.

        Ignore labels are excluded from the frequencies. Absent classes retain
        weight 1, including when the entire batch is ignored. No upper bound is
        applied to weights for present classes.
        """
        hist = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes - 1)
        if 0 <= self.ignore_index < num_classes:
            hist[self.ignore_index] = 0
        freq = hist / hist.sum().clamp_min(1)

        weights = torch.ones_like(freq)
        present = freq > 0
        weights[present] = 1 - self.alpha * torch.log(freq[present])
        return weights

    def forward(self, *args, **kwags):
        raise NotImplementedError("function has not been implemented")


class CrossEntropy2D(BasicLossModule):
    """
    The 2D cross entropy loss.

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        preds: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` before calling this loss module.
    """

    def __init__(self, label_smoothing=0.0, *args, **kwargs):
        super(CrossEntropy2D, self).__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def forward(self, preds, target):
        assert preds is not None, "Expect `preds` is not None, but None instead"
        assert target is not None, "Expect `target` is not None, but None instead"
        assert len(preds.shape) == 4 and len(target.shape) == 3

        # Get the size of `preds`
        n, c, h, w = preds.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(preds.device)
            else:
                weight = None

        # `size_average` and `reduce` of `nn.CrossEntropyLoss` are in the process of being deprecated
        loss = F.cross_entropy(
            preds,
            target.long(),
            weight,
            ignore_index=self.ignore_index,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class BinaryCrossEntropy2D(BasicLossModule):
    """
    The binary 2D cross entropy loss.
    Note that `F.binary_cross_entropy_with_logits` do not support `ignore_index`

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.8]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        preds: 3-D or 4-D torch.Tensor. The predict result without `sigmoid/softmax`, if `preds` is
            3-D/4D Tensor, the shape of `preds` should be (n,h,w)/(n,c,h,w) respectively
        target: torch.Tensor. The input target which shape should be same as `preds`

        Note that there's no need to use `softmax/sigmoid` before calling this loss module.
    """

    def __init__(self, *args, **kwargs):
        super(BinaryCrossEntropy2D, self).__init__(*args, **kwargs)

    def forward(self, preds, target):
        assert len(preds.shape) == len(target.shape)

        # Get the size of `preds`
        if len(preds.shape) == 4:
            n, c, h, w = preds.size()
        elif len(preds.shape) == 3:
            n, h, w = preds.size()
        else:
            raise AttributeError("Expect `preds` is a 3-D or 4-D Tensor, but {}-D instead".format(len(preds.shape)))

        # Reshape as (n, h*w*c)/(n, h*w)
        if len(preds.shape) == 4:
            preds_rsp = preds.transpose(1, 2).transpose(2, 3).reshape(1, -1)
            target_rsp = target.transpose(1, 2).transpose(2, 3).reshape(1, -1)
        else:
            preds_rsp = preds.reshape(1, -1)
            target_rsp = target.reshape(1, -1)

        # Get positive/negative/ignore index
        pos_index = target_rsp == 1
        neg_index = target_rsp == 0
        # ign_index = (target_rsp == self.ignore_index)
        ign_index = target_rsp > 1

        # Convert `target_rsp[ign_index]` to `0` first
        target_rsp[ign_index] = 0

        # Convert `positive/negative/ignore index` as `bool`
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ign_index = ign_index.data.cpu().numpy().astype(bool)

        # Calculate the weight
        weight = np.zeros(preds_rsp.size(), dtype=np.float32)
        if self.custom_weight is not None:
            weight[neg_index] = self.custom_weight[0] * 1.0
            weight[pos_index] = self.custom_weight[1] * 1.0
            weight[ign_index] = 0  # weight for `ignore_index` is 0 !

            weight = torch.from_numpy(weight.astype(np.float32)).to(preds.device)
        else:
            if self.batch_weight:
                pos_num = pos_index.sum()
                neg_num = neg_index.sum()
                sum_num = pos_num + neg_num
                if sum_num != 0:
                    weight[pos_index] = 1 + (neg_num * 1.0 / sum_num) if pos_num > 0 else 1
                    weight[neg_index] = 1 + (pos_num * 1.0 / sum_num) if neg_num > 0 else 1
                    weight[ign_index] = 0  # weight for `ignore_index` is 0 !
                else:
                    raise AttributeError("The sum of `pos_index` and `neg_index` is 0")

                weight = torch.from_numpy(weight.astype(np.float32)).to(preds.device)
            else:
                weight = None

        # Calculate binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(preds_rsp, target_rsp, weight=weight, reduction="sum")

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class ImageBasedCrossEntropy2D(BasicLossModule):
    """
    Image Weighted Cross Entropy Loss

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        preds: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` before calling this loss module.
    """

    def __init__(self, *args, **kwargs):
        super(ImageBasedCrossEntropy2D, self).__init__(*args, **kwargs)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 and len(target.shape) == 3

        # Get the size of `preds`
        n, c, h, w = preds.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(preds.device)
            else:
                weight = None

        # Calculate the loss
        loss = F.nll_loss(
            F.log_softmax(preds, dim=1),
            target.long(),
            weight,
            ignore_index=self.ignore_index,
            reduction="sum",
        )

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class BoundariesRelaxation2D(BasicLossModule):
    """
    The boundaries relaxation loss, which details can be seen here:
    https://ieeexplore.ieee.org/abstract/document/8954327

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

        window_size: int, list or tuple (default 3). The slide window size of boundaries relaxation loss
        stride: int, list or tuple (default 1). The strode of slide window

    forward:
        preds: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` before calling this loss module.
    """

    def __init__(self, window_size=3, stride=1, *args, **kwargs):
        super(BoundariesRelaxation2D, self).__init__(*args, **kwargs)

        # Init window size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        elif isinstance(window_size, list or tuple) and len(window_size) == 2:
            window_size = tuple(window_size)
        else:
            raise AttributeError("Expect type of `window_size`: int, 2-elem list or tuple")

        # Init stride
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, list or tuple) and len(stride) == 2:
            stride = tuple(stride)
        else:
            raise AttributeError("Expect type of `stride`: int, 2-elem list or tuple")

        self.window_size = window_size
        self.stride = stride
        self.pool2d = nn.AvgPool2d(kernel_size=self.window_size, stride=self.stride)

    def forward(self, preds, target):
        # Get the size of `preds`
        n, c, h, w = preds.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(preds.device)
            else:
                weight = None

        # Get soft output of `preds`
        preds_soft = F.softmax(preds, dim=1)

        # Get `ignore_index`
        ignore_index = target == self.ignore_index

        # Convert 3-D `target` Tensor to 4-D `onehot` Tensor
        target_clamps = target.clone()
        target_clamps[ignore_index] = c - 1
        target_onehot = F.one_hot(target_clamps.long(), c)  # n,h,w,c
        target_onehot[ignore_index] = 0  # n,h,w,c
        target_onehot_trans = target_onehot.transpose(2, 3).transpose(1, 2)  # n,c,h,w

        # Get the boundaries relaxation result of `preds` and `target`
        preds_br = self.pool2d(preds_soft)
        target_br = self.pool2d(target_onehot_trans.float())

        # Get loss, note that the loss' lower bound is 0
        loss = -target_br * (torch.log(preds_br + 1e-14) - torch.log(target_br + 1e-14))

        # Get new loss if `weight` is not None
        if weight is not None:
            weight_matrix = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1,c,1,1
            loss = loss * weight_matrix

        # Get sum of loss
        loss = loss.sum()

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class BinaryBoundariesRelaxation2D(BoundariesRelaxation2D):
    """
    The boundaries relaxation loss (version for binary cross-entropy loss)

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of two category. For example,
            [0.2, 0.8]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)
        window_size: int, list or tuple (default 3). The slide window size of boundaries relaxation loss
        stride: int, list or tuple (default 1). The strode of slide window

    forward:
        preds: 4-D torch.Tensor. The predict result without `sigmoid`, which shape is (n, 1, h, w)
        target: 3-D torch.Tensor. The input binary target which shape is (n, h, w)

        Note that there's no need to use `sigmoid` for `preds` before calling this loss module.
    """

    def forward(self, preds, target):
        # Get the size of `preds`
        n, c, h, w = preds.size()
        assert c == 1

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c + 1).to(preds.device)
            else:
                weight = None

        # Get soft output of `preds`
        preds_soft = torch.sigmoid(preds)

        # Get positive/negative/ignore index
        pos_index = target == 1
        neg_index = target == 0
        # ign_index = (target == self.ignore_index)
        ign_index = target > 1

        # Reset the values of `ignore index` to 0
        target[ign_index] = 0

        # Get the boundaries relaxation result of `preds` and `target`
        preds_br = self.pool2d(preds_soft)  # n,1,h,w
        target_br = self.pool2d(target.float().unsqueeze(1))  # n,1,h,w

        # Get binary boundary relaxation loss
        loss = -(target_br * torch.log(preds_br + 1e-14) + (1 - target_br) * torch.log(1 - preds_br + 1e-14))
        loss = loss.squeeze(1)  # n,h,w

        # Get new loss if `weight` is not None
        if weight is not None:
            weight_matrix = torch.zeros(target.size(), dtype=torch.float32)
            weight_matrix[neg_index] = weight[0]  # weight for positive index
            weight_matrix[pos_index] = weight[1]  # weight for negative index
            weight_matrix = weight_matrix.unsqueeze(1)  # n,1,h,w
            weight_matrix = self.pool2d(weight_matrix)
            weight_matrix = weight_matrix.squeeze(1)  # n,h,w

            loss = weight_matrix * loss

        # Get sum of loss
        loss = loss.sum()

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class OHEMCrossEntropy2D(BasicLossModule):
    """
    Online Hard Example Mining (OHEM) CrossEntropy Loss 2D for segmentation

    Params:
        ignore_index: int (default 255). Target categories to be ignored
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1]. If `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch will be used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

        score_threshold: float (default 0.7). Only the pixels with the confidence score lower than it will be trained.
        min_kept_examples: int (default 100000). The min number of pixels to be used during training.
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount of smoothing when computing
            the loss, where 0.0 means no smoothing.

    Forward:
        preds: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` before calling this loss module.
    """

    def __init__(self, score_threshold=0.7, min_kept_examples=100000, label_smoothing=0.0, *args, **kwargs):
        super(OHEMCrossEntropy2D, self).__init__(*args, **kwargs)
        self.score_threshold = score_threshold
        self.label_smoothing = label_smoothing
        self.min_kept_examples = max(1, min_kept_examples)

    def forward(self, preds, target):
        assert len(preds.shape) == 4 and len(target.shape) == 3

        # Get the size of `preds`
        n, c, h, w = preds.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(preds.device)
            else:
                weight = None

        # `size_average` and `reduce` of `nn.CrossEntropyLoss` are in the process of being deprecated
        loss = F.cross_entropy(
            preds,
            target.long(),
            weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        loss = loss.view(-1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        valid_loss = loss[valid_mask]

        num_hard = max(self.min_kept_examples, int(len(valid_loss) * self.score_threshold))
        num_hard = min(num_hard, len(valid_loss))
        if num_hard < len(valid_loss):
            _, hard_indices = torch.topk(valid_loss, k=num_hard)
            hard_loss = valid_loss[hard_indices]
        else:
            hard_loss = valid_loss

        loss = hard_loss.sum()

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= h * w

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        # for safe gradient backward
        if (target == self.ignore_index).all():
            loss = loss * 0

        return loss

class DiceLoss2D(BasicLossModule):
    """
    The Dice Loss for semantic segmentation. It is particularly effective for
    class imbalance problems. The loss is calculated as 1 - DiceCoefficient.

    Params:
        ignore_index: int (default 255). Target categories to be ignored.
        custom_weight: numpy array or list (default None). The weight for each class's loss.
        batch_weight: bool (default True). If true, calculates class weights based on the current batch.
                      Ignored if `custom_weight` is provided.
        eps: float (default 1e-5). A small constant to avoid division by zero.

        Note: The `size_average` and `batch_average` parameters from BasicLossModule are
        ignored in this implementation, as Dice Loss is a ratio-based metric and
        such averaging is not applicable. The loss is computed per-class and then
        averaged across classes.

    Forward:
        preds: 4-D torch.Tensor. The prediction result without `softmax`, with shape (n, c, h, w).
        target: 3-D torch.Tensor. The ground truth target with shape (n, h, w).

        Note that there's no need to use `softmax` before calling this loss module.
    """

    def __init__(self, eps=1e-5, *args, **kwargs):
        super(DiceLoss2D, self).__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, preds, target):
        assert len(preds.shape) == 4 and len(target.shape) == 3
        assert preds.shape[2:] == target.shape[1:], "Spatial dimensions of preds and target must match"

        n, c, h, w = preds.size()

        probs = F.softmax(preds, dim=1)
        valid_mask = target != self.ignore_index

        target_clamped = target.clone().long()
        target_clamped[~valid_mask] = 0
        target_onehot = F.one_hot(target_clamped, num_classes=c).permute(0, 3, 1, 2)

        # Apply the ignore mask to both probs and one-hot target
        # This ensures that ignored pixels contribute 0 to both numerator and denominator.
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(target_onehot, dim=dims)
        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss_per_class = 1.0 - dice_score

        if self.custom_weight is not None:
            weight = self.custom_weight.to(preds.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target.float(), c).to(preds.device)
            else:
                weight = None

        if weight is not None:
            assert weight.numel() == c, f"Weight tensor must have {c} elements, but got {weight.numel()}"
            dice_loss_per_class = dice_loss_per_class * weight

        loss = dice_loss_per_class.mean()
        return loss

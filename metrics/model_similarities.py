# coding=utf-8

from torch import nn

__all__ = [
    "model_structural_similarity",
]


def model_structural_similarity(input: nn.Module, target: nn.Module, significance_threshold=0.5):
    """
    Structural similarity is defined as the percentage of the common non-zero weight locations, i.e.,
    indices, in both the intermedia model and the final target model. NOTE: The locations of the
    `significance_threshold` most significant non-zero weights from `input` model are selected to
    calculate the structural similarity.

    Details are descripted in:
    https://openreview.net/pdf?id=493VFz-ZvDD

    Params:
        input_model (nn.Module): The input/intermedia model.
        target (nn.Module): The target/final model.
        significance_threshold (float, optional): The percentage threshold of the most significant weights. Defaults to 0.5.
    """
    count_numel = 0
    match_numel = 0
    for n, m in input.named_modules():
        if isinstance(m, nn.Conv2d):
            input_weight = m.weight
            target_weight = target.state_dict()[n + ".weight"]
            input_weight = input_weight.abs().view(-1)
            target_weight = target_weight.abs().view(-1)
            # get seleted number of elements
            nonzero_mask = (input_weight > 0).int()
            selected_numel = (nonzero_mask.sum() * significance_threshold).int()
            # sort weight by descending order.
            _, input_indices = input_weight.sort(descending=True)
            target_selected = target_weight[input_indices[:selected_numel]]
            # update `count_numel` and `match_numel`
            count_numel += selected_numel
            match_numel += (target_selected > 0).int().sum()

    return match_numel / count_numel

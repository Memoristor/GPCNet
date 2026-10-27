# coding=utf-8

import copy
import random

import numpy as np
import torch

from tools import named_params

from .pruner import Pruner

__all__ = [
    "PHEW",
]


def get_norm(p):
    p2 = p.detach().cpu().abs()
    if len(p2.shape) > 2:
        out_c, in_c = p2.size(0), p2.size(1)
        p2 = p2.view(out_c, in_c, -1)
        p2 = p2.sum(axis=-1)
    return p2.numpy()


def generate_probability(parameters):
    parameters = copy.deepcopy(parameters)
    prob = []
    reverse_prob = []
    kernel_prob = []
    for p in parameters:
        if len(p.data.size()) in [2, 4]:
            p2 = abs(p.detach().cpu().numpy())
            p1 = get_norm(p)

            p1 = np.array(p1)
            pvals = np.full(p1.shape, 1 / p1.shape[1])
            sum_p1 = p1.sum(axis=1).astype(float)
            mask_p1 = sum_p1 != 0
            pvals[mask_p1] = p1[mask_p1] / sum_p1[mask_p1, None]
            reverse_prob.append(pvals)

            p1 = np.transpose(np.array(p1))
            pvals = np.full(p1.shape, 1 / p1.shape[1])
            sum_p1 = p1.sum(axis=1).astype(float)
            mask_p1 = sum_p1 != 0
            pvals[mask_p1] = p1[mask_p1] / sum_p1[mask_p1, None]
            prob.append(pvals)

            p2_sum = np.abs(p2).reshape(p2.shape[0], p2.shape[1], -1).sum(axis=-1)
            p2_sum = p2_sum[:, :, None, None] if len(p2.shape) == 4 else p2_sum[:, :None]
            p2 = p2 / p2_sum

            cout, cin = p2.shape[:2]
            sum_p2 = p2.reshape(cout, cin, -1).sum(axis=-1)
            pvals = p2 / sum_p2[..., None, None] if len(p2.shape) == 4 else p2 / sum_p2
            kernel_prob.append(pvals)

    return prob, reverse_prob, kernel_prob


def generate_masks(parameters):
    weight_masks = []
    for p in parameters:
        if len(p.data.size()) in [2, 4]:
            retained_inds = p.data.abs() > 0
            weight_masks.append(retained_inds.float())

    bias_masks = []
    for i in range(len(weight_masks)):
        mask = torch.ones(len(weight_masks[i]))
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        bias_masks.append(mask)
    return weight_masks, bias_masks


def sum_masks(masks):
    num = 0
    for i in range(len(masks)):
        num = num + masks[i].sum().int()
    return num


def select_seed_unit(mask, robin_unit=-1, forward=True, round_robin=False):
    seed_unit = None
    if forward:
        length = mask[0].shape[1]
    elif not forward:
        length = mask[-1].shape[0]
    if not round_robin:
        seed_unit = random.choice(list(range(length)))
    elif round_robin:
        if robin_unit < length - 1:
            seed_unit = robin_unit + 1
        else:
            seed_unit = 0
    return seed_unit


def select_seed_unit_counter(mask, counter, forward=True):
    seed_unit = None
    if forward:
        length = mask[0].shape[1]
    elif not forward:
        length = mask[-1].shape[0]
    seed_unit = np.argmin(counter)
    return seed_unit


def get_param_options(mask, prev_unit, prob, kernel_prob, forward):
    prob_sum = sum(prob)
    prob = [x / prob_sum for x in prob]  # make sum prob to 1
    if forward:
        # print(len(prob), sum(prob))
        idx1 = int(np.random.choice(list(range(mask.shape[0])), 1, p=prob))
        idx2 = int(prev_unit)

        inds = np.random.choice(
            list(range(kernel_prob[idx1][idx2].shape[0] * kernel_prob[idx1][idx2].shape[1])),
            1,
            p=kernel_prob[idx1][idx2].reshape(-1),
        )
        idx3 = inds // kernel_prob[idx1][idx2].shape[0]
        idx4 = inds % kernel_prob[idx1][idx2].shape[0]
        return idx1, idx2, int(idx3), int(idx4), mask.shape[0]
    else:  # if not forward:
        idx1 = int(prev_unit)
        idx2 = int(np.random.choice(list(range(mask.shape[1])), 1, p=prob))
        inds = np.random.choice(
            list(range(kernel_prob[idx1][idx2].shape[0] * kernel_prob[idx1][idx2].shape[1])),
            1,
            p=kernel_prob[idx1][idx2].reshape(-1),
        )
        idx3 = inds // kernel_prob[idx1][idx2].shape[0]
        idx4 = inds % kernel_prob[idx1][idx2].shape[0]
        return idx1, idx2, int(idx3), int(idx4)


def get_unit_options(mask, prev_unit, prob, forward):
    prob_sum = sum(prob)
    prob = [x / prob_sum for x in prob]  # make sum prob to 1
    if forward:
        idx_options = list(range(mask.shape[0]))
        return np.random.choice(idx_options, 1, p=prob), prev_unit
    else:  # if not forward:
        idx_options = list(range(mask.shape[1]))
        return prev_unit, np.random.choice(idx_options, 1, p=prob), mask.shape[1]


def conv_to_linear_unit(mask, prev_unit, conv_length, linear_length, forward):
    if forward:
        idx = random.choice(list(range(int(mask.shape[1] / conv_length))))
        idx = idx + int(mask.shape[1] / conv_length) * prev_unit
        return idx
    elif not forward:
        factor = int(linear_length / conv_length)
        idx = int(prev_unit / factor)
        return idx


def phew_masks(
    parameters,
    num_weights,
    prob,
    reverse_prob,
    kernel_prob,
    weight_masks,
    bias_masks,
    verbose=True,
    kernel_conserved=False,
):
    params = copy.deepcopy(parameters)
    num = 0
    for p in params:
        p.data.fill_(0)
        if len(p.data.size()) in [2, 4]:
            num = num + 1

    input_robin_unit = -1
    input_counter = np.zeros(weight_masks[0].shape[1])
    output_robin_unit = -1
    output_counter = np.zeros(weight_masks[-1].shape[0])
    i = 0

    while sum_masks(weight_masks) < num_weights:
        if i % 2 == 0:
            conv_length = 0
            # prev_unit = select_seed_unit_counter(weight_masks, input_counter, forward=True)
            prev_unit = select_seed_unit(
                weight_masks,
                input_robin_unit,
                forward=True,
                round_robin=True,
            )
            input_robin_unit = prev_unit
            input_counter[prev_unit] = input_counter[prev_unit] + 1
            ctol_flag = 0
            k = 0

            while k < len(weight_masks):
                if len(weight_masks[k].shape) == 4:
                    if k + 3 < len(weight_masks) and len(weight_masks[k + 3].shape) == 4:
                        if weight_masks[k + 3].shape[2] == 1:
                            k = k + random.choice([0, 3])
                    # print(k)
                    idx1, idx2, idx3, idx4, conv_length = get_param_options(
                        weight_masks[k],
                        prev_unit,
                        prob[k][int(prev_unit)],
                        kernel_prob[k],
                        forward=True,
                    )
                    weight_masks[k][idx1, idx2, idx3, idx4] = 1
                    if kernel_conserved:
                        weight_masks[k][idx1, idx2].fill_(1)
                    bias_masks[k][idx1] = 1
                    prev_unit = idx1
                    ctol_flag = 1

                    if k + 1 < len(weight_masks) and len(weight_masks[k + 1].shape) == 4:
                        if weight_masks[k + 1].shape[2] == 1:
                            k = k + 1
                    # else:
                    k = k + 1
                    # print(k,'shreyas')

                elif len(weight_masks[k].shape) == 2:
                    if ctol_flag == 1:
                        prev_unit = conv_to_linear_unit(weight_masks[k], prev_unit, conv_length, 0, forward=True)
                        ctol_flag = 0
                    idx1, idx2 = get_unit_options(weight_masks[k], prev_unit, prob[k][int(prev_unit)], forward=True)
                    weight_masks[k][idx1, idx2] = 1
                    bias_masks[k][idx1] = 1
                    prev_unit = idx1
                    if k == num - 1:
                        output_counter[prev_unit] = output_counter[prev_unit] + 1
                    k = k + 1

        else:
            prev_unit = select_seed_unit(weight_masks, output_robin_unit, forward=False, round_robin=True)
            output_robin_unit = prev_unit
            output_counter[prev_unit] = output_counter[prev_unit] + 1
            ltoc_flag = 0
            linear_length = 0
            k = 0

            while k < len(weight_masks):
                if len(weight_masks[num - k - 1].shape) == 2:
                    idx1, idx2, linear_length = get_unit_options(
                        weight_masks[num - k - 1],
                        prev_unit,
                        reverse_prob[num - k - 1][int(prev_unit)],
                        forward=False,
                    )
                    weight_masks[num - k - 1][idx1, idx2] = 1
                    bias_masks[num - k - 1][idx1] = 1
                    prev_unit = idx2
                    ltoc_flag = 1
                    k = k + 1

                elif len(weight_masks[num - k - 1].shape) == 4:
                    if ltoc_flag == 1:
                        prev_unit = conv_to_linear_unit(
                            weight_masks[num - k - 1],
                            prev_unit,
                            conv_length,
                            linear_length,
                            forward=False,
                        )
                        ltoc_flag = 0

                    if weight_masks[num - k - 1].shape[2] == 1:
                        k = k + random.choice([0, 1])

                    idx1, idx2, idx3, idx4 = get_param_options(
                        weight_masks[num - k - 1],
                        prev_unit,
                        reverse_prob[num - k - 1][int(prev_unit)],
                        kernel_prob[num - k - 1],
                        forward=False,
                    )
                    weight_masks[num - k - 1][idx1, idx2, idx3, idx4] = 1
                    if kernel_conserved:
                        weight_masks[num - k - 1][idx1, idx2].fill_(1)
                    bias_masks[num - k - 1][idx1] = 1
                    prev_unit = idx2
                    if k == num - 1:
                        input_counter[prev_unit] = input_counter[prev_unit] + 1

                    if weight_masks[num - k - 1].shape[2] == 1:
                        k = k + 3
                    else:
                        k = k + 1

        i = i + 1
        if verbose:
            print(f"Enabled Weights: {sum_masks(weight_masks)} / {num_weights}", end="\r", flush=True)

    return weight_masks, bias_masks


class PHEW(Pruner):
    """
    PHEW : Constructing Sparse Networks that Learn Fast and Generalize Well without Training Data

    Reference:
        [1] paper: http://proceedings.mlr.press/v139/patil21a.html
        [2] code: https://github.com/ShreyasMalakarjunPatil/PHEW
        [2] code: https://github.com/VITA-Group/ramanujan-on-pai
    """

    def calc_score(self, dataloader, predict_func, loss_func):
        pass

    def local_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level parameter-wise."""
        raise NotImplementedError

    def global_mask(self, density, tolerance=0.005):
        """Updates masks of the parameters with scores by density level globally."""
        raise NotImplementedError

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
        params_dict = named_params(self._model)
        num_params = sum([p.numel() for p in params_dict.values()])
        num_unpruned = int(num_params * density)

        parameters = [params_dict[name] for name in self.masks.keys()]
        prob, reverse_prob, kernel_prob = generate_probability(parameters)

        weight_masks, bias_masks = generate_masks([torch.zeros_like(p) for p in parameters])

        weight_masks, bias_masks = phew_masks(
            parameters,
            num_unpruned,
            prob,
            reverse_prob,
            kernel_prob,
            weight_masks,
            bias_masks,
            verbose=self.verbose,
        )

        layer_idx = 0
        for name, mask in self.masks.items():
            if name.endswith(".weight") and len(mask.size()) in [2, 4]:
                mask.copy_(weight_masks[layer_idx])
                # check if bias mask is exist
                bias_name = name[:-7] + ".bias"
                if bias_name in self.masks.keys():
                    self.masks[bias_name].copy_(bias_masks[layer_idx])
                layer_idx += 1

        # check whether masks are valid
        num_params = sum([v.numel() for v in params_dict.values()])
        num_unpruned_needed = int(density * num_params)
        num_unpruned = 0
        for n, p in params_dict.items():
            if n in self.masks.keys():
                num_unpruned += self.masks[n].sum()
            else:  # Pruning for this layer is disabled
                num_unpruned += p.numel()

        if abs(num_unpruned_needed - num_unpruned) / num_params > tolerance:
            raise AttributeError(
                f"Out of {num_params} parameters, {num_params - num_unpruned_needed} parameters should be pruned, "
                f"but {num_params - num_unpruned} parameters has been pruned"
            )

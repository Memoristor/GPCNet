# coding=utf-8

import copy
from itertools import cycle

import torch
from torch import distributed as dist
from tqdm import tqdm

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = [
    "PAGS_SNIP",
]


def build_adjacency_matrix_from_mask(M):
    """Builds the adjacency matrix A of a bipartite graph from the mask matrix M.

    Args:
        M (torch.Tensor): Binary mask matrix of shape (Cout, Cin).

    Returns:
        torch.Tensor: Adjacency matrix A of shape (Cin + Cout, Cin + Cout).
    """
    Cin, Cout = M.shape[1], M.shape[0]
    A = torch.zeros((Cin + Cout, Cin + Cout), dtype=torch.int64, device=M.device)
    A[:Cin, Cin:] = M.T  # Top-right block: M^T (connections from input to output layer)
    A[Cin:, :Cin] = M  # Bottom-left block: M (connections from output to input layer)
    return A


def build_weighted_adjacency_matrix(M, P):
    """Build a weighted adjacency matrix W from a mask M and parameter matrix P.

    Args:
        M (torch.Tensor): Binary mask matrix of shape (Cout, Cin).
        P (torch.Tensor): Parameter matrix of shape (Cout, Cin) containing edge weights.

    Returns:
        torch.Tensor: Weighted adjacency matrix of shape (Cin + Cout, Cin + Cout).
    """
    Cout, Cin = M.shape
    P_masked = M * P  # Element-wise multiplication (Hadamard product)
    W = torch.zeros((Cin + Cout, Cin + Cout), dtype=P.dtype, device=P.device)
    W[:Cin, Cin:] = P_masked.T  # Transpose to match input->output mapping
    W[Cin:, :Cin] = P_masked  # Fill lower-left block (output to input connections)
    return W


def compute_ramanujan_gap(A: torch.Tensor) -> float:
    """
    Compute the Ramanujan Gap (Δr = 2√(d - 1) - μ̂(A)) for a symmetric adjacency matrix.

    Args:
        A (torch.Tensor): Symmetric adjacency matrix of shape (N, N).

    Returns:
        torch.Tensor: Ramanujan Gap value.
    """
    # Verify input symmetry
    assert torch.allclose(A, A.T), "Adjacency matrix must be symmetric"

    # Calculate average degree d
    A = A.float()
    degrees = A.sum(dim=1)
    d = degrees.mean()  # Average degree

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(A)  # Eigenvalues in ascending order
    abs_eigenvalues = torch.abs(eigenvalues)

    # Extract second-largest nontrivial eigenvalue
    sorted_abs, _ = torch.sort(abs_eigenvalues, descending=True)
    if sorted_abs.size(0) < 2:
        mu_hat = 0.0  # Handle edge cases (e.g., single-node graphs)
    else:
        mu_hat = sorted_abs[1]  # Skip the principal eigenvalue (sorted_abs[0])

    # Compute Ramanujan Gap
    ramanujan_gap = 2 * torch.sqrt(d - 1) - mu_hat
    return ramanujan_gap.item()


def compute_weighted_spectral_gap(W: torch.Tensor) -> float:
    """
    Compute the Weighted Spectral Gap (λ = μ₀(|W|) − μ̂(|W|)) for a weighted adjacency matrix.

    Args:
        W (torch.Tensor): Weighted adjacency matrix of shape (N, N).

    Returns:
        torch.Tensor: Weighted Spectral Gap λ.
    """
    # Step 1: Compute absolute value matrix |W|
    abs_W = torch.abs(W).float()

    # Step 2: Calculate eigenvalues of |W|
    eigenvalues = torch.linalg.eigvalsh(abs_W)

    # Step 3: Extract magnitudes of eigenvalues (handles complex eigenvalues)
    magnitudes = torch.abs(eigenvalues)

    # Step 4: Sort eigenvalues by magnitude in descending order
    sorted_magnitudes, _ = torch.sort(magnitudes, descending=True)

    # Step 5: Handle edge cases (e.g., single-node graphs)
    if sorted_magnitudes.size(0) < 2:
        return 0.0  # Default for trivial graphs

    # Step 6: Compute μ₀ (largest eigenvalue) and μ̂ (second-largest eigenvalue)
    mu0 = sorted_magnitudes[0]
    mu_hat = sorted_magnitudes[1]

    # Step 7: Compute weighted spectral gap λ
    lambda_gap = mu0 - mu_hat
    return lambda_gap.item()


def compute_spectrum_l2(initial_weights, weight_mask, weight_type="fc"):
    """
    Compute semi Full-Spectrum L2 Distance with GPU acceleration

    Args:
        weight_mask (torch.Tensor): Binary mask matrix of shape (Cout, Cin)/(Cout, Cin, kh, kw).
        initial_weights (torch.Tensor): Parameter matrix of shape (Cout, Cin)/(Cout, Cin, kh, kw) containing edge weights.

    Returns:
        Tensor: L2 distance in full-spectrum coordinate space
    """
    device = initial_weights.device

    # Handle convolutional layer reshaping
    if weight_type == "conv":
        assert len(initial_weights.shape) == 4, "This is not a weight of the convolution layer"
        cout, cin, kh, kw = initial_weights.shape
        weight_flat = initial_weights.view(cout, -1)
        mask_flat = weight_mask.view(cout, -1)
    else:
        assert len(initial_weights.shape) == 2, "This is not a weight of the fc layer"
        weight_flat = initial_weights
        mask_flat = weight_mask

    mask_adj = build_adjacency_matrix_from_mask(mask_flat)
    weight_adj = build_weighted_adjacency_matrix(mask_flat, weight_flat)

    # Calculate Ramanujan-related metrics and weighted spectral gaps
    delta_r = compute_ramanujan_gap(mask_adj)
    lambda_w = compute_weighted_spectral_gap(weight_adj)

    # Assemble vectors and compute L2 distance
    return torch.norm(torch.tensor([delta_r, lambda_w], device=device), p=2)


class PAGS_SNIP(Pruner):
    """
    ICLR 2019. SNIP: Single-shot Network Pruning based on Connection Sensitivity
    ICLR 2023. Don't just prune by magnitude! Your mask topology is a secret weapon

    Reference:
    [1] SNIP: https://arxiv.org/abs/1810.02340
    [1] PAGS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html
    """

    def __init__(self, param_names: list, verbose=False, population_size=1000):
        super().__init__(param_names, verbose)
        self.population_size = population_size

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
        params_dict = named_params(self._model)
        best_mask = None
        best_distance = None

        # Create infinite dataloader
        infinite_loader = cycle(dataloader)

        # Initial batch data
        last_bacth_data = next(infinite_loader)
        for k, v in last_bacth_data.items():
            last_bacth_data[k] = v.to(torch.device(local_rank()))

        # Synchronizes all processes
        dist.barrier()

        # Find best mask
        bar = tqdm(range(self.population_size))
        for _ in bar:
            curr_batch_data = next(infinite_loader)
            for k, v in curr_batch_data.items():
                curr_batch_data[k] = v.to(torch.device(local_rank()))

            # get a batch data with random indices
            comb_batch_data = dict()
            for k, lv in last_bacth_data.items():
                cv = curr_batch_data[k]
                v = torch.cat([lv, cv], dim=0)
                v = v[torch.randperm(v.shape[0])[: lv.shape[0]]]
                comb_batch_data[k] = v

            # Calculate snip-based saliency score
            self._model.zero_grad()
            pred = predict_func(self._model, comb_batch_data)
            losses = loss_func(pred, comb_batch_data)
            L = torch.sum(torch.stack(list(losses.values())))
            L.backward()  # all reduce

            for name, score in self.scores.items():
                param = params_dict[name]
                score.copy_(torch.clone(param.grad * param).detach().abs_())

            # Normalization
            sum_score = sum([v.sum() for v in self.scores.values()])
            for score in self.scores.values():
                score.div_(sum_score)

            # generate mask
            self.global_mask(density=density, tolerance=tolerance)

            total_distance = 0
            for n, m in self.masks.items():
                p = params_dict[n]

                try:
                    isvalid = m.sum() > 1000 and m.sum() / m.numel() > 1e-3
                    if n.endswith(".weight") and len(m.shape) == 4 and isvalid:
                        distance = compute_spectrum_l2(initial_weights=p, weight_mask=m, weight_type="conv")
                    elif n.endswith(".weight") and len(m.shape) == 2 and isvalid:
                        distance = compute_spectrum_l2(initial_weights=p, weight_mask=m, weight_type="fc")
                    else:
                        distance = torch.tensor(0.0, device=m.device, dtype=torch.float32)

                    if not torch.isnan(distance):
                        total_distance += distance
                    else:
                        if self.verbose:
                            print(f"the distance of layer {n} is nan")

                except torch._C._LinAlgError as e:
                    if self.verbose:
                        print(f"{n}: {str(e)}")

            if best_distance is None or best_distance < total_distance:
                best_mask = copy.deepcopy(self.masks)
                best_distance = total_distance

            # update last batch data
            last_bacth_data = curr_batch_data

            bar.set_description(
                f"[{self.__class__.__name__}] Rank: {local_rank()}, Loss: {L:5.5f}, "
                f"Best full spectrum distance: {best_distance:5.5f}"
            )

        # Copy best masks to self.masks
        for n, m in self.masks.items():
            m.copy_(torch.clone(best_mask[n]))

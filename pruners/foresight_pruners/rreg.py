# coding=utf-8


import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from tools import named_params

from .rand_erk import RandERK

__all__ = [
    "RReg",
]


def bipartite_d_regular_graph(n, d):
    """
    Build a d-regular bipartite graph with n nodes on the left and n nodes on the right,
    using Havel-Hakimi. Return an (n x n) adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes on each bipartite side.
    d : int
        Desired regular degree on each node (left and right).
        Require that (d <= n), otherwise it's impossible for each node to have d distinct neighbors.

    Returns
    -------
    mask : np.ndarray of shape (n, n)
        0/1 adjacency matrix for the bipartite graph.
        mask[i, j] = 1 means there's an edge between left-node i and right-node j.

    Raises
    ------
    NetworkXError
        If the degree sequence is not realizable as a simple bipartite graph.
    """
    if d > n:
        raise ValueError(f"d={d} cannot exceed n={n} in a bipartite d-regular graph.")

    # Prepare the degree sequences for the left and right sets
    seq_left = [d] * n
    seq_right = [d] * n  # same length, same total sum

    # Use bipartite Havel-Hakimi
    # This returns a graph G with nodes labeled from 0..(n_left+n_right - 1),
    #   - The first n nodes have attribute 'bipartite'=0 (left side)
    #   - The next n nodes have attribute 'bipartite'=1 (right side)
    G = nx.bipartite.havel_hakimi_graph(seq_left, seq_right)

    # We want an (n x n) adjacency matrix: rows = left nodes, columns = right nodes
    # By default, G has:
    #   left nodes labeled [0..n-1],
    #   right nodes labeled [n..2n-1].
    # We can fill the matrix accordingly.
    mask = np.zeros((n, n), dtype=int)

    # Get the edges from G (u, v). We expect u in [0..n-1], v in [n..2n-1].
    for u, v in G.edges():
        # Ensure (u, v) is from left->right
        if u < n and v >= n:
            i = u  # left index
            j = v - n  # right index
        elif v < n and u >= n:
            i = v
            j = u - n
        else:
            # If it somehow doesn't cross partitions (shouldn't happen in a valid bipartite construction),
            # we can skip or raise an error
            continue

        mask[i, j] = 1

    return mask


class RReg(RandERK):
    """
    Data-Free Model Pruning at Initialization via Expanders

    Reference:
    [1] https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Stewart_Data-Free_Model_Pruning_at_Initialization_via_Expanders_CVPRW_2023_paper.pdf#page=0.96
    """

    def __init__(self, param_names: list, verbose=False, power_scale=1.0, regular_d=15):
        super().__init__(param_names, verbose, power_scale)
        self.regular_d = regular_d

    def custom_mask(self, dataloader, predict_func, loss_func, density, tolerance=0.005):
        """Updates masks of the parameters in the custom manner."""
        # Check parameters density globally
        params_density = self.layerwise_density(density=density)

        # Get masks
        bar = tqdm(self.masks.items())
        for i, (name, mask) in enumerate(bar):
            bar.set_description(f"[{self.__class__.__name__}] Layer index: {i + 1}/{len(bar)}")

            d = params_density[name]
            if d < 1.0:
                if len(mask.shape) in [2, 4]:
                    cout, cin = mask.shape[:2]
                    kh = 1 if len(mask.shape) == 2 else mask.shape[2]
                    kw = 1 if len(mask.shape) == 2 else mask.shape[3]

                    max_vertices = min(cout, cin)
                    max_edges = int(d * cin * cout)

                    bool_mask = torch.zeros((cout, cin), dtype=torch.bool).to(mask.device)
                    while bool_mask.sum() < max_edges:
                        connections = int(max_edges - bool_mask.sum())

                        regular_d = min(self.regular_d, max_vertices)  # for most cases, max_vertices >= 3
                        vertices = max(min(int(connections / regular_d), max_vertices), regular_d)
                        regular_mask = bipartite_d_regular_graph(vertices, regular_d)
                        regular_mask = torch.from_numpy(regular_mask).bool().to(bool_mask.device)

                        out_sample_idx = torch.randperm(cout)[:vertices]
                        in_sample_idx = torch.randperm(cin)[:vertices]

                        bool_mask[out_sample_idx[:, None], in_sample_idx] = torch.logical_or(
                            regular_mask, bool_mask[out_sample_idx[:, None], in_sample_idx]
                        )

                    if len(mask.shape) == 2:
                        mask.copy_(bool_mask.float().detach())
                    else:
                        mask.copy_(bool_mask.float().unsqueeze(2).unsqueeze(3).expand(-1, -1, kh, kw).detach())

                else:
                    mask.copy_(torch.ones_like(mask).detach())

            else:
                mask.copy_(torch.ones_like(mask).detach())

        # Check masks
        params_dict = named_params(self._model)
        global_params = torch.cat([torch.flatten(v) for v in params_dict.values()])
        target_unpruned = int(density * global_params.numel())
        num_unpruned = 0
        for n, p in params_dict.items():
            if n in self.masks.keys():
                num_unpruned += self.masks[n].sum()
            else:  # Pruning for this layer is disabled
                num_unpruned += p.numel()

        if abs(target_unpruned - num_unpruned) / global_params.numel() > tolerance:
            raise AttributeError(
                f"Out of {global_params.numel()} parameters, {global_params.numel() - target_unpruned} parameters should be pruned, "
                f"but {global_params.numel() - num_unpruned} parameters has been pruned"
            )

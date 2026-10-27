# coding=utf-8


import torch
from torch import distributed as dist
from torch.autograd import grad
from tqdm import tqdm

from tools import local_rank, named_params

from .pruner import Pruner

__all__ = ["WideTopoNTK"]


class WideTopoNTK(Pruner):
    """
    Exploring Wide Neural Topologies of Foresight Pruning from the Perspective of Preserving Training Dynamics
    """

    def __init__(
        self,
        param_names: list,
        verbose: bool = False,
        hutch_iters: int = 1,
        accu_steps: int = 1,
        wide_topo: bool = True,
        compression: float = 1.0,
    ):
        super(WideTopoNTK, self).__init__(param_names, verbose)

        self.hutch_iters = hutch_iters
        self.accu_steps = accu_steps
        self.wide_topo = wide_topo
        self.compression = compression

    def calc_ntk_grads(self, model, truth, predict_func):
        """Calculate the grads of each parameter of unbiased estimation of NTK"""
        ntk_grads = dict()
        for k, v in self.masks.items():
            ntk_grads[k] = torch.zeros_like(v)

        # Compute the true graph using eval mode
        ntk_estimation = 0
        params = named_params(model)
        for _ in range(self.hutch_iters):
            model.train()
            preds = predict_func(model, truth)

            if isinstance(preds, torch.Tensor):
                dots = torch.randn_like(preds).mul_(preds).sum()
            elif isinstance(preds, (list, tuple)):
                dots = sum(torch.randn_like(out).mul_(out).sum() for out in preds)
            elif isinstance(preds, dict):
                dots = sum(torch.randn_like(out).mul_(out).sum() for out in preds.values())
            else:
                raise ValueError("Unsupported output type for NTK importance accumulation.")

            grads = grad(dots, list(params.values()), create_graph=True)  # type: ignore
            grads_flatten = torch.cat([g.flatten() for g in grads])
            dist.all_reduce(grads_flatten, op=dist.ReduceOp.SUM)

            params_masks = [self.masks[k] if k in self.masks else torch.ones_like(p) for k, p in params.items()]
            masks_flatten = torch.cat([m.flatten() for m in params_masks])

            grads_flatten = grads_flatten * masks_flatten
            trace = (grads_flatten * grads_flatten).sum()
            ntk_estimation += trace / self.hutch_iters

            grads = grad(trace, list(params.values()), create_graph=False, allow_unused=True)
            for g, (n, p) in zip(grads, params.items()):
                if n in self.masks.keys():
                    s = ntk_grads[n]
                    m = self.masks[n]
                    if g is not None:
                        g = g.contiguous()
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                        s.add_(torch.clone(g * m).detach())
                    else:
                        if self.verbose:
                            print(f"[{self.__class__.__name__}] param {n} is unused in second order derivate")

        return ntk_grads, ntk_estimation

    def calc_score(self, dataloader, predict_func, loss_func):
        """Calculate the score of each parameter"""
        accu_grads = {k: torch.zeros_like(v) for k, v in self.scores.items()}

        model = self._model
        params = named_params(model)

        # Synchronizes all processes
        dist.barrier()
        bar = tqdm(dataloader)
        for i, truth in enumerate(bar):
            for k, v in truth.items():
                truth[k] = v.to(torch.device(local_rank()))

            ntk_grads, ntk_estimation = self.calc_ntk_grads(model, truth, predict_func)
            for k in self.scores.keys():
                accu_grads[k].add_(ntk_grads[k])

            if (i + 1) % self.accu_steps == 0 or (i + 1) == len(dataloader):
                if self.verbose:
                    print(
                        f"[{self.__class__.__name__}] Rank: {local_rank()}, "
                        f"Accumulated NTK estimation at step {i + 1}"
                    )

                for k, s in self.scores.items():
                    g = accu_grads[k] / self.accu_steps
                    p = params[k]
                    s.add_(torch.clone((g * p).abs()).detach())
                    accu_grads[k].zero_()

            bar.set_description(
                f"[{self.__class__.__name__}] Computing scores, "
                f"Rank: {local_rank()}, "
                f"NTK: {ntk_estimation:5.5f}"
            )

        # Normalize score
        ntk_score = torch.norm(torch.cat([s.flatten() for s in self.scores.values()]))
        if self.verbose:
            print(f"[{self.__class__.__name__}] The norm of score is {ntk_score:4.4f} before normalization")

        for k, s in self.scores.items():
            s.div_(ntk_score)

        # Wide topology
        if self.wide_topo:
            masks_vec = torch.cat([m.flatten() for m in self.masks.values()])
            masks_density = masks_vec.sum() / masks_vec.numel()
            model_compression = -torch.log10(masks_density + 1e-10)

            for s, m in zip(self.scores.values(), self.masks.values()):
                if len(m.size()) == 4:  # convolution layers
                    out_deg = m.sum(dim=[1, 2, 3])
                    inp_deg = m.sum(dim=[0, 2, 3])
                    outer = torch.outer(out_deg, inp_deg)
                    outer = torch.unsqueeze(outer, 2)
                    outer = torch.unsqueeze(outer, 3)
                    sensitivity = (m / outer.clamp(min=1)) ** (model_compression / self.compression)
                    s.mul_(torch.clone(sensitivity).detach())
                elif len(m.size()) == 2:  # linear layers
                    out_deg = m.sum(dim=1)
                    inp_deg = m.sum(dim=0)
                    outer = torch.outer(out_deg, inp_deg)
                    sensitivity = (m / outer.clamp(min=1)) ** (model_compression / self.compression)
                    s.mul_(torch.clone(sensitivity).detach())

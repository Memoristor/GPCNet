# coding=utf-8

import fnmatch
import os
import random
import socket
from collections.abc import Mapping

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel

__all__ = [
    # Commonly used
    "local_rank",
    "to_device",
    "sync_state_dict",
    "get_by_path",
    "find_files",
    "seed_init",
    "named_params",
    "wandb_init",
    # For model pruning
    "lanczos_algorithm",
    "hard_bn_momentum",
    "reset_bn_momentum",
    "reset_bn_status",
    # For model re-parameterization
    #  https://github.com/THU-MIG/RepViT/blob/main/utils.py
    "replace_batchnorm",
]


def local_rank():
    """Get local rank of the current process in distributed training"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # elif "SLURM_LOCALID" in os.environ:
    #     return int(os.environ["SLURM_LOCALID"])
    else:
        raise EnvironmentError("Cannot find the local rank of the current process.")


@torch.no_grad()
def to_device(obj, device):
    """Convert obj to device recursively"""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return type(obj)(*(to_device(x, device) for x in obj))
        return tuple(to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, Mapping):
        out = obj.__class__()
        for k, v in obj.items():
            out[k] = to_device(v, device)  # type: ignore
        return out
    return obj


@torch.no_grad()
def sync_state_dict(obj, src=0, device=None):
    """sync optimizer state across ranks so they are all the same as src rank

    Note: obj can be model, optimizer, or any object with state_dict() and load_state_dict() methods.

    Params:
        obj: object to be synchronized
        src: source rank
        device: if not None, move the state_dict to this device after broadcasting
    """
    assert dist.is_initialized(), "torch.distributed is not initialized."
    assert hasattr(obj, "state_dict"), "obj must have state_dict() method."
    assert hasattr(obj, "load_state_dict"), "obj must have load_state_dict() method."

    obj_list = [to_device(obj.state_dict(), "cpu") if local_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    obj_state = obj_list[0]
    if device is not None:
        obj_state = to_device(obj_state, device)
    obj.load_state_dict(obj_state)


def get_by_path(data, path: str, default=None):
    """Get value by path from a nested dictionary or list"""
    if path is None:
        return None

    keys = path.split(".")
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list):
            try:
                idx = int(key)
                data = data[idx]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return data


def find_files(directory, pattern):
    """Find all files that matches the pattern under the directory"""
    for root, dirs, files in os.walk(os.path.expanduser(directory)):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(os.path.join(root, basename), pattern):
                filename = os.path.join(root, basename)
                yield filename


def seed_init(seed=0, cuda_deterministic=True):
    """Init seed of numpy, random, torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def named_params(model):
    """Get named parameters of model"""
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        return dict([(n, w) for (n, w) in model.module.named_parameters()])
    else:
        return dict([(n, w) for (n, w) in model.named_parameters()])


def wandb_init(wandb_path, wandb_config: DictConfig, project_name: str, experiment_name: str):
    """Init weight&bias. If the previous run does not exist, a new run will be created.
    Otherwise, the existing run will be loaded."""
    wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
    runid_txt = os.path.join(wandb_path, "run_id.txt")
    if os.path.exists(runid_txt):
        with open(runid_txt, "r") as f:
            run_id = f.read().strip()
        run = wandb.init(
            config=wandb_config,
            project=project_name,
            notes=socket.gethostname(),
            name=experiment_name,
            dir=wandb_path,
            id=run_id,
            resume="allow",
        )
    else:
        run = wandb.init(
            config=wandb_config,
            project=project_name,
            notes=socket.gethostname(),
            name=experiment_name,
            dir=wandb_path,
            resume="auto",
        )
        with open(os.path.join(wandb_path, "run_id.txt"), "w") as f:
            f.write(run.id)


def lanczos_algorithm(mvp_func, dim: int, neigs: int, ncv=None, *args, **kwargs):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products).

    Modified from repo: https://github.com/locuslab/edge-of-stability/blob/github/src/utilities.py
    """
    device = torch.device(local_rank())

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return mvp_func(gpu_vec).detach().cpu().numpy()  # matrix vector product function

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs, ncv=ncv, *args, **kwargs)
    evals = torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float().to(device)
    evecs = torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float().to(device)
    return evals, evecs


@torch.no_grad()
def hard_bn_momentum(model):
    """Disable momentum of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm):
            m.backup_momentum = m.momentum
            m.momentum = 1.0

    model.apply(func)


@torch.no_grad()
def reset_bn_momentum(model):
    """Disable momentum of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum

    model.apply(func)


@torch.no_grad()
def reset_bn_status(model):
    """Reset running status of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm):
            m.reset_running_stats()

    model.apply(func)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

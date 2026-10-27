# coding=utf-8

import copy
import os
import typing
from functools import partial

import torch
import torch_pruning as tp
import wandb
from torch import distributed as dist
from torch.backends.cuda import sdp_kernel
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch_pruning.pruner.importance import MagnitudeImportance
from tqdm import tqdm

import losses
from pruners.foresight_pruners.widetopo_ntk import WideTopoNTK
from pruners.structural_pruners.analyzer import MaskedSparsityAnalyzer, SparsityAnalyzer
from pruners.structural_pruners.importance import *
from pruners.structural_pruners.pruner import ModuleSparsityPruner
from tools import utils

from .basic_engine import BasicEngine

__all__ = ["BasicPrune"]


class BasicPrune(BasicEngine):
    """The engine is used in the model pruning phase."""

    def __init__(self, *args, **kwags):
        super(BasicPrune, self).__init__(*args, **kwags)

        # Load criterion
        self.criteria = dict()
        self.criteria_weight = dict()
        for k, v in self.config.train.losses.items():  # train losses
            self.criteria[k] = getattr(losses, v.cls)(**v.get("kwargs", {}))
            self.criteria_weight[k] = v.get("weight", 1.0)

    def losses(self, pred: dict, truth: dict):
        """Calculate the losses of each prediction and corresponding ground truth.

        Params:
            pred (dict): The predict data, e.g. {'body': body, 'edge': edge}
            truth (dict): The truth data, e.g. {'body': body, 'edge': edge}

        Returns:
            return the dict of losses, e.g. {'body': body_loss, 'edge': edge_loss}
        """
        losses = dict()
        for k, v in self.config.train.losses.items():  # train losses
            tensors = list()
            for pairs in v.forward:
                for pk, pv in pairs.items():
                    if pk == "inputs":
                        tensors.append(utils.get_by_path(truth, pv))
                    elif pk == "outputs":
                        tensors.append(utils.get_by_path(pred, pv))
                    else:
                        raise AttributeError(f"Unknown keys: {pk}, not in `inputs` or `outputs`")
            losses[k] = self.criteria[k](*tensors) * self.criteria_weight[k]

        return (losses,)

    def sum_losses(self, losses):
        """Sum all losses"""
        if isinstance(losses, torch.Tensor):
            return losses
        elif isinstance(losses, (list, tuple)):
            if len(losses) == 0:
                return torch.tensor(0.0, device=torch.device(utils.local_rank()))
            return torch.sum(torch.stack([self.sum_losses(loss) for loss in losses]))
        elif isinstance(losses, dict):
            if len(losses) == 0:
                return torch.tensor(0.0, device=torch.device(utils.local_rank()))
            return torch.sum(torch.stack([self.sum_losses(v) for v in losses.values()]))
        else:
            raise ValueError("The type of losses is not supported.")

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        wandb.define_metric(f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}
        return wandb_info

    def predict(self, model, truth):
        """Get model predictions

        Params:
            truth (dict). Ground truth.
        """
        input_tensors = list()
        for k in self.config.model.input_keys:
            input_tensors.append(truth[k])

        output_tensors = model(*input_tensors)
        if isinstance(output_tensors, (list, tuple)):
            pred = dict([(k, v) for k, v in zip(self.config.model.output_keys, output_tensors)])
        else:
            pred = {self.config.model.output_keys[0]: output_tensors}
        return pred

    def hook_before_run(self):
        """Hook function before run"""
        pass

    def hook_for_input(self, truth):
        """Hook function for input data"""
        # for semseg-depth tasks
        if "frames" in truth.keys():
            truth["image"] = truth["frames"][:, 0]
        return truth

    def hook_for_output(self, pred):
        """Hook function for output data"""
        return pred

    @staticmethod
    def _get_modules_by_names(model, module_names):
        """Get the modules from the model based on the module names"""
        if module_names is not None:
            if isinstance(module_names, str):
                module_names = [module_names]
            elif isinstance(module_names, typing.Iterable):
                module_names = list(module_names)
            else:
                raise ValueError("module_names should be str or iterable of str.")
            modules = [model.get_submodule(name) for name in module_names]
        else:
            modules = [model]
        return modules

    def _prune_with_magnitude(self, model, prune_config, device):
        """Prune model with ntk-based method (best version)"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        # batch_size = prune_config.batch_size
        # sample_size = prune_config.sample_size
        # dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        # forward_fn = prune_config.get("forward_fn", None)
        # if forward_fn is not None:
        #     forward_fn = getattr(model, forward_fn)
        # else:
        #     forward_fn = model

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=MagnitudeImportance(**prune_config.importance.get("kwargs", {})),
                ignored_layers=ignored_layers,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def _prune_with_ntk(self, model, prune_config, device):
        """Prune model with ntk-based method (best version)"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = model

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate NTK importance
            importance = NTKImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating NTK gradients: {i + 1}/{len(dataloader)}")
                image = truth["image"].to(device)
                model.train()
                with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        outputs = forward_fn(image)
                importance.accumulate_score(model, outputs)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def resize_label(self, label, scale_factor):
        return F.interpolate(label.unsqueeze(1).float(), scale_factor=scale_factor, mode="nearest").squeeze(1).long()

    def _prune_with_snip(self, model, prune_config, device):
        """Prune model with SNIP"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = partial(self.predict, model)

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate NTK importance
            importance = SNIPImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating SNIP gradients: {i + 1}/{len(dataloader)}")
                for k, v in truth.items():
                    truth[k] = v.to(device)
                if "label" in truth.keys():
                    label = truth["label"]
                    label_x2 = self.resize_label(label, scale_factor=0.5)
                    label_x4 = self.resize_label(label, scale_factor=0.25)
                    label_x8 = self.resize_label(label, scale_factor=0.125)
                    truth["label_x2"] = label_x2
                    truth["label_x4"] = label_x4
                    truth["label_x8"] = label_x8
                model.train()
                with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    outputs = forward_fn(truth)
                importance.accumulate_score(model, outputs, truth, loss_func=self.losses, sum_losses=self.sum_losses)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def _prune_with_synflow(self, model, prune_config, device):
        """Prune model with Synflow"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = model

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate NTK importance
            importance = SynflowImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating Synflow scores: {i + 1}/{len(dataloader)}")
                image = truth["image"].to(device)
                model.eval()
                importance.accumulate_score(model, image)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def _prune_with_grasp(self, model, prune_config, device):
        """Prune model with GraSP"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = partial(self.predict, model)

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate GraSP importance
            importance = GraSPImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating GraSP gradients: {i + 1}/{len(dataloader)}")
                for k, v in truth.items():
                    truth[k] = v.to(device)
                if "label" in truth.keys():
                    label = truth["label"]
                    label_x2 = self.resize_label(label, scale_factor=0.5)
                    label_x4 = self.resize_label(label, scale_factor=0.25)
                    label_x8 = self.resize_label(label, scale_factor=0.125)
                    truth["label_x2"] = label_x2
                    truth["label_x4"] = label_x4
                    truth["label_x8"] = label_x8
                model.train()
                with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    outputs = forward_fn(truth)
                importance.accumulate_score(model, outputs, truth, loss_func=self.losses, sum_losses=self.sum_losses)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def _prune_with_ntksap(self, model, prune_config, device):
        """Prune model with NTKSAP"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = model

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate NTK importance
            importance = NTKSAPImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating NTKSAP scores: {i + 1}/{len(dataloader)}")
                image = truth["image"].to(device)
                model.eval()
                importance.accumulate_score(model, image)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def _prune_with_px(self, model, prune_config, device):
        """Prune model with NTKSAP"""
        # Init sparsity analyzer
        analyzer = SparsityAnalyzer()
        analyzer.record_original(model)

        # Get target module and ignored layers
        target_module = self._get_modules_by_names(model, prune_config.pruner.get("target_module", None))[0]
        ignored_layers = self._get_modules_by_names(model, prune_config.pruner.get("ignored_layers", []))

        # Get sparsity of the target module
        target_sparsity = prune_config.target_sparsity
        prune_iters = prune_config.prune_iters

        # p * (1 - alpha)^T = p * (1 - S) => alpha = 1 - (1 - S)^(1/T)
        alpha = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_iters)
        iterative_sparsity = [1.0 - (1.0 - alpha) ** (t + 1) for t in range(prune_iters)]

        target_module_params = tp.utils.count_params(target_module)
        target_module_params_remained = [(1.0 - s) * target_module_params for s in iterative_sparsity]

        # Build dataloader
        batch_size = prune_config.batch_size
        sample_size = prune_config.sample_size
        dataloader = self.build_dataloader(phase=self.phase, batch_size=batch_size, num_samples=sample_size)

        # Forward function
        forward_fn = prune_config.get("forward_fn", None)
        if forward_fn is not None:
            forward_fn = getattr(model, forward_fn)
        else:
            forward_fn = model

        # Iterative pruning process
        prune_bar = tqdm(range(prune_iters))
        for t in prune_bar:
            prune_bar.set_description(f"Pruning iteration: {t + 1}/{prune_iters}")

            target_module_params = tp.utils.count_params(target_module)
            pruning_ratio = 1 - target_module_params_remained[t] / target_module_params

            assert pruning_ratio < 1.0, "Pruning ratio must be less than 1.0"

            if pruning_ratio <= 0.0:
                continue

            # Calculate PX importance
            importance = PXImportance(**prune_config.importance.get("kwargs", {}))

            acc_bar = tqdm(dataloader)
            for i, truth in enumerate(acc_bar):
                acc_bar.set_description(f"Accumulating PX scores: {i + 1}/{len(dataloader)}")
                image = truth["image"].to(device)
                model.eval()
                importance.accumulate_score(model, image)

            importance.apply_to_model(model)
            pruner = ModuleSparsityPruner(
                model=model,
                target_module=target_module,
                pruning_ratio=pruning_ratio,
                example_inputs=torch.randn((1, 3, *prune_config.pruner.example_input_size)).to(device),
                importance=importance,
                ignored_layers=ignored_layers,
                # ignore_mlp_fc2=True,
                **prune_config.pruner.get("kwargs", {}),
            )
            pruner.step()

            results = analyzer.compute_sparsity(model)
            analyzer.print_report(results)
            analyzer.print_summary(model, hierarchy_depth=2)

        return model

    def run(self):
        """Train model"""
        if self.is_resumed:
            self.logger.info("Model is resumed, pruning skipped.")
            return

        if dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError("Distributed pruning is currently not supported.")

        # Init weight&bias
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            # wandb.define_metric("epoch")
            # wandb.define_metric("losses/*", step_metric="epoch")
            # wandb.define_metric("metrics/*", step_metric="epoch")
            # wandb.define_metric("learning_rate/*", step_metric="epoch")
            # wandb.define_metric("confusion_mat/*", step_metric="epoch")

        # Hook before run
        self.hook_before_run()

        # Structural pruning process
        prune_config = self.config[self.phase]
        device = torch.device(utils.local_rank())

        if prune_config.importance.type == "mag":
            model = self._prune_with_magnitude(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "ntk":
            model = self._prune_with_ntk(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "snip":
            model = self._prune_with_snip(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "synflow":
            model = self._prune_with_synflow(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "grasp":
            model = self._prune_with_grasp(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "ntksap":
            model = self._prune_with_ntksap(self.model.module, prune_config, device=device)
        elif prune_config.importance.type == "px":
            model = self._prune_with_px(self.model.module, prune_config, device=device)
        else:
            raise ValueError(f"Unsupported importance type: {prune_config.importance.type}")

        # Save final pruned model
        save_full_model = self.config[self.phase].get("save_full_model", False)
        self.save(os.path.join(self.save_checkpoints, "last_epoch.pth"), 0, save_full_model, model=model)

        # Mark the run as finished
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()

    def save(self, path: str, epoch: int, save_full_model: bool = False, model=None) -> None:
        """Save checkpoint including epoch and model only

        Params:
            path: The file path where the checkpoint will be saved.
            epoch: The current epoch number.
            save_full_model: If True, saves the complete model object instead
                of just the state_dict. This is useful for sparse/pruned models
                where the architecture may have changed.
            model: The model to be saved. If None, uses self.model.
        """
        if utils.local_rank() == 0:
            state_dict = {"epoch": epoch}

            if model is None:
                model = self.model

            # Save model state dict
            if not save_full_model:
                if isinstance(model, (DataParallel, DistributedDataParallel)):
                    state_dict["model_state_dict"] = model.module.state_dict()
                else:
                    state_dict["model_state_dict"] = model.state_dict()
            else:
                if isinstance(model, (DataParallel, DistributedDataParallel)):
                    state_dict["model"] = copy.deepcopy(model.module).to("cpu")
                else:
                    state_dict["model"] = copy.deepcopy(model).to("cpu")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

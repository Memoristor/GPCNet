# coding=utf-8

import json
import os

import numpy as np
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from tqdm import tqdm

import losses
import metrics
from datasets import mixup
from tools import utils

from .basic_engine import BasicEngine

__all__ = [
    "BasicTrain",
    "ClassificationTrain",
    "SegmentationTrain",
]


class BasicTrain(BasicEngine):
    """The engine is used in the training phase."""

    def __init__(self, *args, **kwags):
        super(BasicTrain, self).__init__(*args, **kwags)

        # Init metrics
        self.metrics = dict()
        if isinstance(self.config[self.phase].metrics, (dict, DictConfig)):
            for k, v in self.config[self.phase].metrics.items():
                self.metrics[k] = getattr(metrics, v.cls)(**v.get("kwargs", {}))
                self.metrics[k].to(torch.device(utils.local_rank()))
                self.metrics[k].reset()

        # Load criterion
        self.criteria = dict()
        self.criteria_weight = dict()
        for k, v in self.config[self.phase].losses.items():  # train losses
            self.criteria[k] = getattr(losses, v.cls)(**v.get("kwargs", {}))
            self.criteria_weight[k] = v.get("weight", 1.0)

        # Create mixup/cutmix
        self.mixup = None
        if "mixup" in self.config[self.phase].keys():
            self.mixup = getattr(mixup, self.config[self.phase].mixup.cls)(
                **self.config[self.phase].mixup.get("kwargs", {})
            )

    def losses(self, pred: dict, truth: dict):
        """Calculate the losses of each prediction and corresponding ground truth.

        Params:
            pred (dict): The predict data, e.g. {'body': body, 'edge': edge}
            truth (dict): The truth data, e.g. {'body': body, 'edge': edge}

        Returns:
            return the dict of losses, e.g. {'body': body_loss, 'edge': edge_loss}
        """
        losses = dict()
        for k, v in self.config[self.phase].losses.items():  # train losses
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

    def gradient_backward(self, losses):
        """Gradient backward"""
        self._optimizer.zero_grad()
        # loss_all.backward()

        loss_all = self.sum_losses(losses)
        self._grad_scaler.scale(loss_all).backward()

        # Gradient clipping
        if isinstance(self.optimizer, dict):
            if "gradient_norm_clipping" in self.config[self.phase].optimizer.keys():
                # Unscales the gradients of optimizer's assigned params in-place
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self._grad_scaler.unscale_(self._optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config[self.phase].optimizer.gradient_norm_clipping,
                )
        else:
            if "gradient_norm_clipping" in self.config.optimizer.keys():
                # Unscales the gradients of optimizer's assigned params in-place
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self._grad_scaler.unscale_(self._optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.optimizer.gradient_norm_clipping,
                )

        # self._optimizer.step()
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

        # Update model EMA
        if self._model_ema is not None:
            self._model_ema.update(self.model)

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

    def update_metrics(self, pred, truth):
        """Update metrics

        Params:
            pred (dict): The predict data, e.g. {'body': body, 'edge': edge}
            truth (dict): The truth data, e.g. {'body': body, 'edge': edge}
        """
        for k, v in self.metrics.items():
            update_tensors = list()
            for pairs in self.config[self.phase].metrics[k].tensor_keys:
                for pk, pv in pairs.items():
                    if pk == "inputs":
                        update_tensors.append(utils.get_by_path(truth, pv))
                    elif pk == "outputs":
                        update_tensors.append(utils.get_by_path(pred, pv))
                    else:
                        raise AttributeError("Unknown keys, not in `inputs` or `outputs`")
            v.update(*update_tensors)

    @staticmethod
    def normalize_cm(cm_count, mode: str = "true"):
        """Normalize confusion matrix

        Params:
            cm_count (torch.Tensor): Confusion matrix counts. Shape (num_classes, num_classes).
            mode (str): Normalization mode, one of {'true','pred','all'}.
        """
        cm = cm_count.to(dtype=torch.float64)
        if mode == "true":
            denom = cm.sum(dim=1, keepdim=True).clamp_min(1)
            return cm / denom
        elif mode == "pred":
            denom = cm.sum(dim=0, keepdim=True).clamp_min(1)
            return cm / denom
        elif mode == "all":
            s = cm.sum().clamp_min(1)
            return cm / s
        else:
            raise ValueError("mode must be one of {'true','pred','all'}")

    @staticmethod
    def plot_heatmap(mat: np.ndarray, class_names, title, fmt=".2f", cmap="Blues", cbar_label=None):
        """Plot heatmap"""
        fig, ax = plt.subplots(figsize=(0.9 * len(class_names), 0.8 * len(class_names)))
        sns.heatmap(
            mat,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": cbar_label} if cbar_label else None,
            annot_kws={"fontsize": 9},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def run_epoch(self, dataloader, epoch, phase="train", mixup=None, summary_freq=10):
        """Run one epoch, and get the metrics and losses.

        Params:
            dataloader (DataLoader): The data loader for the dataset.
            epoch (int): The current epoch number.
            phase (str): The phase of training, options are `train` or `valid`.
            mixup (callable): A callable function or object that applies Mixup or CutMix augmentation.
            summary_freq (int): Frequency of logging summaries. Defaults to 10.
        """
        if phase == "train":
            self.model.train()
            grad_func = torch.enable_grad()
            # Set epoch for distributed sampler
            if isinstance(dataloader, DataLoader):
                dataloader.sampler.set_epoch(epoch)
        elif phase == "valid":
            self.model.eval()
            grad_func = torch.no_grad()
        else:
            raise AttributeError(f"Unknown phase: {phase}, not in `train`, `valid`")

        # Check is model synced
        self.check_model_synced()
        if self._model_ema is not None:
            self.check_model_synced(self._model_ema.ema)

        # freeze some modules of the model
        self.freeze_modules(epoch=epoch)

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # add confusion matrix
        num_classes = len(dataloader.dataset.CLASSES)  # type: ignore
        class_names = dataloader.dataset.CLASSES  # type: ignore
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes, ignore_index=255)
        confmat = confmat.to(torch.device(utils.local_rank()))

        # Run epochs
        epoch_losses = dict()
        epoch_metrics = dict()
        with grad_func:
            # Synchronizes all processes
            dist.barrier()
            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):
                # Epoch start from `1`
                iterations = i + (epoch - 1) * len(bar)

                for k, v in truth.items():
                    truth[k] = v.to(torch.device(utils.local_rank()))

                # Hook for input data
                truth = self.hook_for_input(truth)

                # Mixup/Cutmix
                if phase == "train" and mixup is not None:
                    assert {
                        "image",
                        "label",
                    } <= truth.keys(), "The key `image` and `label` must in the `truth`"
                    image = truth["image"]
                    label = truth["label"]
                    image, target = mixup(image, label)
                    truth["target"] = target.to(torch.device(utils.local_rank()))
                    truth["label"] = torch.argmax(target, dim=1)

                # Get predict results and losses, update metrics
                if phase == "train":
                    with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                        pred = self.predict(model=self.model, truth=truth)
                        pred = self.hook_for_output(pred)
                        losses = self.losses(pred=pred, truth=truth)

                    # Gradient backward
                    self.gradient_backward(losses=losses)

                    # Update learning rate scheduler by `iteration`
                    if isinstance(self.lr_scheduler, dict):
                        if self.config[self.phase].learning_rate.step_mode == "iteration":
                            self._lr_scheduler.step()
                    else:
                        if self.config.learning_rate.step_mode == "iteration":
                            self._lr_scheduler.step()
                else:
                    pred = self.predict(model=self.model, truth=truth)
                    pred = self.hook_for_output(pred)
                    losses = self.losses(pred=pred, truth=truth)

                self.update_metrics(pred=pred, truth=truth)

                # update confusion matrix
                if "label" in truth.keys() and "logit" in pred.keys():
                    pred_label = torch.argmax(pred["logit"], dim=1)
                    confmat.update(pred_label, truth["label"])

                # flush wandb summary
                if utils.local_rank() == 0 and not self.config.disable_wandb and iterations % summary_freq == 0:
                    wandb_info = self.get_wandb_info(
                        phase, iterations, pred=pred, truth=truth, dataloader=dataloader, epoch=epoch
                    )
                    wandb.log(wandb_info)

                # Reduces the loss data across all machines
                for loss in losses:
                    for k, v in loss.items():
                        v = torch.clone(v.detach())
                        dist.all_reduce(v, op=dist.ReduceOp.SUM)
                        v.div_(dist.get_world_size())
                        if k not in epoch_losses.keys():
                            epoch_losses[k] = v.detach().cpu().numpy() / len(dataloader)
                        else:
                            epoch_losses[k] += v.detach().cpu().numpy() / len(dataloader)

                # Set bar description
                last_lr = [g["lr"] for g in self._optimizer.param_groups]
                loss_all = self.sum_losses(losses)
                bar.set_description(
                    f"[{phase}] Epoch: {epoch}/{self.config[self.phase].num_epoch}, "
                    f"Rank: {utils.local_rank()}, Loss: {loss_all:5.6f}, "
                    f"LR: {', '.join([f'{lr:6.6f}' for lr in last_lr])}"
                )

            # Get metric of this epoch
            for k, v in self.metrics.items():
                epoch_metrics[k] = v.compute().cpu().numpy()

        # update the wandb log for confusion matrix
        cm_count = confmat.compute().detach()
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            cm_norm_true = self.normalize_cm(cm_count, mode="true").cpu().numpy()
            cm_norm_all = self.normalize_cm(cm_count, mode="all").cpu().numpy()

            fig_true = self.plot_heatmap(
                cm_norm_true,
                class_names,
                title=f"{phase} Confusion Matrix (Normalized by True), epoch {epoch}",
                fmt=".2f",
                cmap="Blues",
                cbar_label="Proportion",
            )

            fig_all = self.plot_heatmap(
                cm_norm_all,
                class_names,
                title=f"{phase} Confusion Matrix (Normalized by All), epoch {epoch}",
                fmt=".2f",
                cmap="Greens",
                cbar_label="Proportion",
            )

            wandb_info = {}
            wandb_info[f"confusion_mat/{phase}_true"] = wandb.Image(fig_true)
            wandb_info[f"confusion_mat/{phase}_all"] = wandb.Image(fig_all)
            wandb.log(wandb_info)

            plt.close(fig_true)
            plt.close(fig_all)

        # Update learning rate scheduler by `epoch`
        if isinstance(self.lr_scheduler, dict):
            if phase == "train" and self.config[self.phase].learning_rate.step_mode == "epoch":
                self._lr_scheduler.step()
        else:
            if phase == "train" and self.config.learning_rate.step_mode == "epoch":
                self._lr_scheduler.step()

        return epoch_losses, epoch_metrics

    def freeze_modules(self, epoch=None):
        """Freeze some modules of the model"""
        if "freeze" in self.config[self.phase].keys():
            # get the real model
            if isinstance(self.model, (DistributedDataParallel, DataParallel)):
                model = self.model.module
            else:
                model = self.model

            # freeze modules
            for m in self.config[self.phase].freeze:
                if isinstance(m, (list, tuple, ListConfig)) and len(m) == 2:
                    module_name = m[0]
                    num_epochs = m[1]
                else:
                    module_name = m
                    num_epochs = None

                assert isinstance(module_name, str), "the module name must be a string"

                if num_epochs is None or epoch > num_epochs:
                    if hasattr(model, module_name):
                        mod = getattr(model, module_name)
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False
                        if utils.local_rank() == 0:
                            self.logger.info(f"Freeze the module: {module_name}")
                    else:
                        raise AttributeError(f"The model has no attribute named {module_name}")

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

    def run(self):
        """Train model"""
        if self.start_epoch >= self.config[self.phase].num_epoch:
            return

        # Init weight&bias
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("losses/*", step_metric="epoch")
            wandb.define_metric("metrics/*", step_metric="epoch")
            wandb.define_metric("learning_rate/*", step_metric="epoch")
            wandb.define_metric("confusion_mat/*", step_metric="epoch")

        # Hook before run
        self.hook_before_run()

        # Load the best validation metric records
        best_valid_path = os.path.join(self.save_checkpoints, "best_valid.json")
        try:
            with open(best_valid_path, "r") as f:
                best_valid = json.loads(f.read())

            for k in self.metrics.keys():
                if k not in best_valid.keys():
                    best_valid[k] = {"value": 0.0, "epoch": 0}
        except:
            best_valid = dict([(k, {"value": 0.0, "epoch": 0}) for k in self.metrics.keys()])

        # Create dataloader
        train_dataloader = self.build_dataloader(phase=self.phase)
        valid_dataloader = self.build_dataloader(phase="valid")

        for epoch in range(self.start_epoch + 1, self.config[self.phase].num_epoch + 1):
            # Log information
            last_lr = [g["lr"] for g in self._optimizer.param_groups]
            if utils.local_rank() == 0:
                self.logger.info(f"[Train] exp name: {self.experiment_name}")
                self.logger.info(f"[Train] current learning rate: {', '.join([f'{lr:6.6f}' for lr in last_lr])}")

            # Run epoch
            train_loss, train_metric = self.run_epoch(
                train_dataloader,
                epoch=epoch,
                phase="train",
                mixup=self.mixup,
                summary_freq=self.config[self.phase].summary_freq,
            )
            valid_loss, valid_metric = self.run_epoch(
                valid_dataloader,
                epoch=epoch,
                phase="valid",
                summary_freq=self.config.valid.summary_freq,
            )

            # Save and record results
            if utils.local_rank() == 0:
                if not self.config.disable_wandb:
                    # Add epoch, learning rate, losses and metrics to wandb
                    wandb_info = {
                        "epoch": epoch,
                        **{f"learning_rate/last_lr_{i}": lr for i, lr in enumerate(last_lr)},
                    }

                    for k, v in train_loss.items():
                        wandb_info[f"losses/train_{k}"] = v
                    for k, v in valid_loss.items():
                        wandb_info[f"losses/valid_{k}"] = v
                    for k, v in train_metric.items():
                        wandb_info[f"metrics/train_{k}"] = v
                    for k, v in valid_metric.items():
                        wandb_info[f"metrics/valid_{k}"] = v

                    wandb.log(wandb_info)

                # Save lastest model every 10 epochs
                if epoch == 1 or epoch % 1 == 0 or epoch == self.config[self.phase].num_epoch:
                    save_full_model = self.config[self.phase].get("save_full_model", False)
                    self.save(os.path.join(self.save_checkpoints, "last_epoch.pth"), epoch, save_full_model)

                # Save best model and corresponding metric results
                for k, v in valid_metric.items():
                    if best_valid[k]["value"] < v:
                        best_valid[k]["value"] = float(v)
                        best_valid[k]["epoch"] = epoch

                        save_full_model = self.config[self.phase].get("save_full_model", False)
                        self.save(os.path.join(self.save_checkpoints, f"best_{k}.pth"), epoch, save_full_model)

                        with open(os.path.join(self.save_checkpoints, f"best_{k}.txt"), "w") as f:
                            f.write(
                                f"[Valid] Model: {self.model_class}, Optimizer: {self._optimizer.__class__.__name__}, Epoch: {epoch}\n"
                            )
                            f.write("\n")
                            f.write("[Valid] The train losses\n")
                            for n, s in train_loss.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The train metric\n")
                            for n, s in train_metric.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The valid losses\n")
                            for n, s in valid_loss.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The valid metric\n")
                            for n, s in valid_metric.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")

                    self.logger.info(
                        f"[Valid] {k}: {valid_metric[k]:5.5f}, best {k}: {best_valid[k]['value']:5.5f}/{best_valid[k]['epoch']}"
                    )

                # Save the best valid metrics
                with open(best_valid_path, "w") as f:
                    f.write(json.dumps(best_valid))

        # Mark the run as finished
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()


class ClassificationTrain(BasicTrain):
    """The engine is used in the classification training phase."""

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        # Set parameters of wandb
        wandb.define_metric(f"iteration/{phase}")
        wandb.define_metric("table/*", step_metric=f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}

        if {"truth", "pred", "dataloader", "epoch"} <= kwargs.keys():
            truth = kwargs["truth"]
            pred = kwargs["pred"]
            dataloader = kwargs["dataloader"]
            epoch = kwargs["epoch"]

            if {"image", "label"} <= truth.keys() and "logit" in pred.keys():
                index = np.random.randint(0, truth["image"].shape[0])
                image = truth["image"][index].cpu().numpy().transpose((1, 2, 0))
                image_std = dataloader.dataset.image_std
                image_mean = dataloader.dataset.image_mean
                if dataloader.dataset.div_std is True:
                    image = image * image_std + image_mean
                else:
                    image = image + image_mean
                image = np.asarray(image * 255, dtype=np.uint8)

                label = truth["label"][index].cpu().numpy()
                label = dataloader.dataset.CLASSES[int(label)]

                softmax = torch.softmax(pred["logit"][index], dim=0)
                argmax = torch.argmax(softmax, dim=0).cpu().detach().numpy()
                result = dataloader.dataset.CLASSES[argmax]
                accuracy = softmax[argmax]

                columns = ["Image", "Label", "Result", "Accuracy"]
                data = [
                    [
                        wandb.Image(image, caption=f"epoch_{epoch}_iters_{iters}", file_type="jpg"),
                        label,
                        result,
                        accuracy,
                    ],
                ]

                table = wandb.Table(data=data, columns=columns)
                wandb_info[f"table/{phase}"] = table

        return wandb_info


class SegmentationTrain(BasicTrain):
    """The engine is used in the segmentation training phase."""

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        # Set parameters of wandb
        wandb.define_metric(f"iteration/{phase}")
        wandb.define_metric("table/*", step_metric=f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}

        if {"truth", "pred", "dataloader", "epoch"} <= kwargs.keys():
            truth = kwargs["truth"]
            pred = kwargs["pred"]
            dataloader = kwargs["dataloader"]
            epoch = kwargs["epoch"]

            if {"image", "label"} <= truth.keys() and "logit" in pred.keys():
                index = np.random.randint(0, truth["image"].shape[0])

                image = truth["image"][index].cpu().numpy().transpose((1, 2, 0))
                image_std = dataloader.dataset.image_std
                image_mean = dataloader.dataset.image_mean
                if dataloader.dataset.div_std is True:
                    image = image * image_std + image_mean
                else:
                    image = image + image_mean
                image = np.asarray(image * 255, dtype=np.uint8)

                label = truth["label"][index].cpu().numpy()
                label = dataloader.dataset.decode_label(label)

                result = torch.argmax(pred["logit"][index], dim=0)
                result = result.cpu().detach().numpy()
                result = dataloader.dataset.decode_label(result)

                columns = ["Image", "Label", "Result"]
                data = [
                    [
                        wandb.Image(image, caption=f"epoch_{epoch}_iters_{iters}", file_type="jpg"),
                        wandb.Image(label, caption=f"epoch_{epoch}_iters_{iters}", file_type="jpg"),
                        wandb.Image(result, caption=f"epoch_{epoch}_iters_{iters}", file_type="jpg"),
                    ],
                ]

            table = wandb.Table(data=data, columns=columns)
            wandb_info[f"table/{phase}"] = table

        return wandb_info

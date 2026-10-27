# coding=utf-8

import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import wandb
from omegaconf import OmegaConf
from torch.nn import functional as F

import metrics
from tools import utils

from ..base_engines.basic_train import SegmentationTrain

__all__ = [
    "HighResoSemSegTrain",
]


class HighResoSemSegTrain(SegmentationTrain):
    """The engine is used in the training phase of high-resolution image semantic segmentation.

    Params:
        config: dict. The runtime environment configuration.
    """

    def run(self):
        """(Overwrite) Train model"""
        if self.start_epoch >= self.config.train.num_epoch:
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
            wandb.define_metric("iteration")
            wandb.define_metric("losses/*", step_metric="epoch")
            wandb.define_metric("metrics/*", step_metric="epoch")
            wandb.define_metric("learning_rate/*", step_metric="epoch")
            wandb.define_metric("confusion_mat/*", step_metric="epoch")
            wandb.define_metric("model_density/*", step_metric="epoch")

        # Load the best valid metric data
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
        train_dataloader = self.build_dataloader(phase="train")
        valid_dataloader = self.build_dataloader(phase="valid")

        # Start training
        for epoch in range(self.start_epoch + 1, self.config.train.num_epoch + 1):
            # Log information
            last_lr = [g["lr"] for g in self._optimizer.param_groups]  # type: ignore
            if utils.local_rank() == 0:
                self.logger.info(f"[Train] exp name: {self.experiment_name}")
                self.logger.info(f"[Train] current learning rate: {', '.join([f'{lr:6.6f}' for lr in last_lr])}")

            # Run train epoch and valid epoch
            train_loss, train_metric = self.run_epoch(
                train_dataloader,
                epoch=epoch,
                phase="train",
                mixup=self.mixup,
                summary_freq=self.config.train.summary_freq,
            )
            valid_loss, valid_metric = self.run_epoch(
                valid_dataloader,
                epoch=epoch,
                phase="valid",
                summary_freq=self.config.valid.summary_freq,
            )

            # Save and record result
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

                    model_density = metrics.model_density(self.model, named_parameters=None)
                    model_density = float(model_density.cpu().detach().numpy())  # type: ignore
                    wandb_info["model_density/density"] = model_density

                    wandb_info["training_finished"] = int(epoch == self.config.train.num_epoch)
                    wandb.log(wandb_info)

                # Save lastest model every 10 epochs
                if epoch == 1 or epoch % 1 == 0 or epoch == self.config.train.num_epoch:
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
                                f"[Valid] Model: {self.model_class}, Optimizer: {self.optimizer.__class__.__name__}, Epoch: {epoch}\n"
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

    def hook_for_input(self, truth):
        """Hook function for input data"""
        # for semseg-depth tasks
        if "frames" in truth.keys():
            truth["image"] = truth["frames"][:, 0]

        if "label" in truth.keys():
            label = truth["label"]
            label_x2 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.5, mode="nearest").squeeze(1).long()
            label_x4 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.25, mode="nearest").squeeze(1).long()
            label_x8 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.125, mode="nearest").squeeze(1).long()
            truth["label_x2"] = label_x2
            truth["label_x4"] = label_x4
            truth["label_x8"] = label_x8

        return truth

    def hook_for_output(self, pred):
        """Hook function for output data"""

        if "logits" in pred.keys():
            logits = pred["logits"]
            if isinstance(logits, torch.Tensor):
                logit, aux_logits = logits, None
            elif isinstance(logits, (list, tuple)):
                logit, aux_logits = logits[0], logits[1:] if len(logits) > 1 else None
            elif isinstance(logits, (dict, OrderedDict)):
                logit = logits.pop("0")
                aux_logits = logits if len(logits) > 0 else None
            else:
                raise TypeError(f"logits should be Tensor, list or dict, but got {type(logits)}")

            pred["logit"] = logit
            pred["aux_logits"] = aux_logits

        return pred

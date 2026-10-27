# coding=utf-8

import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import cv2
import thop
import torch
import wandb
from fvcore.nn import FlopCountAnalysis
from omegaconf import OmegaConf
from prettytable import prettytable
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

from tools import op_flops, utils

from ..base_engines.basic_valid import BasicValid

__all__ = ["HighResoSemSegValid"]


class HighResoSemSegValid(BasicValid):
    """The engine is used in the validation phase.

    Params:
        config: dict. The configuration from runtime environment.
    """

    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def run_epoch(self, dataloader, reload_name):
        """Run one epoch, and get the metrics and losses.

        Params:
            dataloader: DataLoader.
        """
        self.model.eval()

        # Check is model synced
        self.check_model_synced()

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # Run epochs
        valid_preds = dict()
        valid_losses = dict()
        valid_metrics = dict()
        with torch.no_grad():
            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):
                for k, v in truth.items():
                    truth[k] = v.to(torch.device(utils.local_rank()))

                truth = self.hook_for_input(truth)
                with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    pred = self.predict(model=self.model, truth=truth)
                    pred = self.hook_for_output(pred)

                    losses = self.losses(pred=pred, truth=truth)
                    loss_all = self.sum_losses(losses)

                # Reduces the loss data across all machines
                for loss in losses:
                    for k, v in loss.items():
                        dist.all_reduce(v, op=dist.ReduceOp.SUM)
                        v.div_(dist.get_world_size())
                        if k not in valid_losses.keys():
                            valid_losses[k] = v.detach().cpu().numpy() / len(dataloader)
                        else:
                            valid_losses[k] += v.detach().cpu().numpy() / len(dataloader)

                # merge all logits as the whole image size
                logit = pred["logit"]
                prob = F.softmax(logit, dim=1)
                image = truth["image"]
                label = truth["label"]
                index = truth["index"]
                size = truth["size"]

                for j in range(prob.shape[0]):
                    path_idx = dataloader.dataset.get_path_index(int(index[j]))
                    top, left, h, w = dataloader.dataset.index_to_position(index[j], size[j][:2])

                    if path_idx not in valid_preds.keys():
                        canvas = torch.zeros((prob.size(1), *size[j][:2]), dtype=prob.dtype, device=prob.device)
                        ground = torch.zeros((*size[j][:2],), dtype=label.dtype, device=label.device)
                        source = torch.zeros_like(ground)
                        counter = 0
                    else:
                        canvas = valid_preds[path_idx]["canvas"]
                        ground = valid_preds[path_idx]["ground"]
                        source = valid_preds[path_idx]["source"]
                        counter = valid_preds[path_idx]["counter"]

                    prob_j = prob[j]
                    label_j = label[j]

                    prob_shape = prob_j.size()[-2:]
                    if prob_shape != (h, w):
                        prob_j = prob_j.unsqueeze(0)
                        prob_j = F.interpolate(prob_j, size=(h, w), mode="bilinear", align_corners=False)
                        prob_j = prob_j.squeeze(0)

                        label_j = label_j.unsqueeze(0).unsqueeze(0).float()
                        label_j = F.interpolate(label_j, size=(h, w), mode="nearest").squeeze(0).squeeze(0).long()

                    canvas[:, top : top + h, left : left + w] += prob_j
                    ground[top : top + h, left : left + w] += label_j
                    source[top : top + h, left : left + w] += torch.ones_like(label_j)
                    counter += 1
                    valid_preds[path_idx] = {
                        "canvas": canvas,
                        "source": source,
                        "ground": ground,
                        "counter": counter,
                    }

                    # merging is completed
                    if counter == (dataloader.dataset.crop_grids[0] * dataloader.dataset.crop_grids[1]):
                        assert source.min() > 0, "There are empty logits in the canvas."
                        whole_prob = canvas / source.unsqueeze(0)
                        whole_label = ground / source
                        whole_pred = {"logit": whole_prob.unsqueeze(0)}
                        # whole_truth = {"label": whole_label.unsqueeze(0)}

                        # load label from dataset to avoid interpolation error
                        label_path = dataloader.dataset.label_path[path_idx]
                        jth_label = dataloader.dataset.load_label(label_path)
                        jth_label = dataloader.dataset.encode_label(jth_label)
                        jth_label = torch.from_numpy(jth_label).long().to(logit.device)
                        whole_truth = {"label": jth_label.unsqueeze(0)}

                        self.update_metrics(pred=whole_pred, truth=whole_truth)

                        # get the relative path compared to the root path
                        image_path = dataloader.dataset.image_path[path_idx]
                        relpath = os.path.relpath(image_path, dataloader.dataset.root_path)
                        relpath = os.path.splitext(relpath)[0] + ".png"

                        # save predictions
                        pred_label = torch.argmax(whole_prob, dim=0)
                        pred_color = dataloader.dataset.decode_label(pred_label.cpu().numpy())
                        pred_path = os.path.join(self.save_result, self.phase, reload_name, "predictions", relpath)
                        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                        cv2.imwrite(pred_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                        # # save heatmap
                        # valid_label = whole_label * (whole_label != 255)
                        # softmax = F.softmax(whole_prob, dim=0)
                        # heatmap = 1 - torch.gather(softmax, dim=0, index=valid_label.long().unsqueeze(0)).squeeze(0)
                        # whole_image = dataloader.dataset.load_image(dataloader.dataset.image_path[path_idx])

                        # heatmap = heatmap.cpu().float().numpy()
                        # jetmap = cv2.applyColorMap((heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
                        # jetmap = jetmap * (valid_label.unsqueeze(2).cpu().numpy() != 255)
                        # fusion = cv2.addWeighted(whole_image, 0.5, jetmap, 0.5, 0)

                        # fusion_path = os.path.join(self.save_result, self.phase, reload_name, "heatmap", relpath)
                        # fusion_dir = os.path.dirname(fusion_path)
                        # os.makedirs(fusion_dir, exist_ok=True)
                        # cv2.imwrite(fusion_path, cv2.cvtColor(fusion, cv2.COLOR_RGB2BGR))

                        # # save error map
                        # error_mask = (pred_label != whole_label) & (whole_label != 255)
                        # error_mask_np = error_mask.cpu().numpy().astype(np.uint8)
                        # error_vis = np.zeros_like(whole_image)
                        # error_vis[error_mask_np == 1.0] = pred_color[error_mask_np == 1.0]
                        # error_overlay = cv2.addWeighted(whole_image, 0.3, error_vis, 0.7, 0)
                        # error_overlay[error_mask_np == 0.0] = whole_image[error_mask_np == 0.0]

                        # mask_uint8 = error_mask_np.astype(np.uint8) * 255
                        # contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # cv2.drawContours(error_overlay, contours, -1, (255, 0, 0), 1)

                        # error_path = os.path.join(self.save_result, self.phase, reload_name, "errormap", relpath)
                        # error_dir = os.path.dirname(error_path)
                        # os.makedirs(error_dir, exist_ok=True)
                        # cv2.imwrite(error_path, cv2.cvtColor(error_overlay, cv2.COLOR_RGB2BGR))

                        # pop the record to release memory
                        valid_preds.pop(path_idx)

                # Set bar description
                bar.set_description(f"[{self.phase}] Rank: {utils.local_rank()}, Loss: {loss_all:5.5f}")

            # Check wheather all predictions have been merged
            assert len(valid_preds) == 0, "There are some predictions not merged."

            # Get metric of this epoch
            for k, v in self.metrics.items():
                valid_metrics[k] = v.compute().cpu().numpy()

        return valid_losses, valid_metrics

    def run(self):
        """Valid model"""
        self.model.eval()

        if not self.is_resumed or self.start_epoch == 0:
            raise AttributeError("Validation only supports resumed trained model.")

        if dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError("Distributed validation is currently not supported.")

        reload_pth = self.config[self.phase].reload_pth
        if reload_pth is None or not reload_pth.endswith(".pth"):
            raise ValueError("Please provide a valid trained model path for validation.")

        reload_name = os.path.basename(reload_pth)[:-4]
        if utils.local_rank() == 0:
            self.logger.info(
                f"[{self.phase}] model: {self.model_class}, dataset: {self.config.dataset.cls}, reload: {reload_name}"
            )

        # Init weight&bias
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            wandb.define_metric("losses/*", step_metric="epoch")
            wandb.define_metric("fine_metrics/*", step_metric="epoch")
            wandb.define_metric("efficiency/*", step_metric="epoch")

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # Run epoch
        valid_dataloader = self.build_dataloader(self.phase, batch_size=1)  # self.config[self.phase].batch_size)
        valid_losses, valid_metrics = self.run_epoch(valid_dataloader, reload_name=reload_name)

        # Get FPS, MACs, parameter size
        warmup_iters = 50
        max_iters = 200

        with torch.no_grad():
            device = torch.device(utils.local_rank())
            truth = next(iter(valid_dataloader))
            image = truth["image"].to(device)
            image = image[:1]  # use one image for efficiency evaluation

            # GPU wramup
            dist.barrier()
            bar = tqdm(range(warmup_iters))
            bar.set_description(f"[{self.phase}] Rank: {utils.local_rank()}, Warming up")
            for _ in bar:
                with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    self.model(image)  # type: ignore

            torch.cuda.synchronize()

            # Calculate time cost
            dist.barrier()
            start_time = time.time()

            bar = tqdm(range(max_iters - warmup_iters))
            bar.set_description(f"[{self.phase}] Rank: {utils.local_rank()}, Calculating FPS")
            for _ in bar:
                with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                    self.model(image)  # type: ignore

            torch.cuda.synchronize()

            end_time = time.time()
            dist.barrier()

            # Calculate FPS
            total_time_seconds = end_time - start_time
            fps = (max_iters - warmup_iters) / total_time_seconds * image.size(0)

            # model profiling
            flop_analyzer = FlopCountAnalysis(self.model, inputs=(image,))
            flop_analyzer.set_op_handle("aten::scaled_dot_product_attention", op_flops.sdpa_flop_jitter)
            flop_analyzer.set_op_handle("prim::PythonOp.SelectiveScanFn", op_flops.selective_scan_flop_jit)
            flop_analyzer.set_op_handle("prim::PythonOp.SpatialVariableConvolutionFunction", op_flops.svc_flop_jit)
            flop_analyzer.set_op_handle(
                "prim::PythonOp.MambaInnerFnNoOutProj", op_flops.mamba_inner_fn_no_out_proj_flop_jit
            )
            flops = flop_analyzer.total()
            flops /= image.size(0)

            macs, params = thop.profile(self.model, inputs=(image,), verbose=False)  # type: ignore
            macs /= image.size(0)

            # print(f"MACs: {macs}, Params: {params}, FPS: {fps}")

        macs = torch.tensor(macs, dtype=torch.float64).to(device)
        dist.all_reduce(macs, op=dist.ReduceOp.SUM)
        macs = macs / dist.get_world_size()
        macs = float(macs.cpu().numpy())

        flops = torch.tensor(flops, dtype=torch.float64).to(device)
        dist.all_reduce(flops, op=dist.ReduceOp.SUM)
        flops = flops / dist.get_world_size()
        flops = float(flops.cpu().numpy())

        params = torch.tensor(params, dtype=torch.float64).to(device)
        dist.all_reduce(params, op=dist.ReduceOp.SUM)
        params = params / dist.get_world_size()
        params = float(params.cpu().numpy())

        fps = torch.tensor(fps, dtype=torch.float64).to(device)
        dist.all_reduce(fps, op=dist.ReduceOp.SUM)
        fps = fps / dist.get_world_size()
        fps = float(fps.cpu().numpy())

        # Save and record result
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            wandb_info = dict()

            # losses
            for k, v in valid_losses.items():
                wandb_info[f"losses/{self.phase}_{reload_name}_{k}"] = v
            # metrics
            for k, v in valid_metrics.items():
                labels = list(valid_dataloader.dataset.CLASSES)
                values = [float(x) for x in v]
                wandb_info[f"fine_metrics/{self.phase}_{reload_name}_{k}"] = wandb.plot.bar(
                    wandb.Table(
                        data=[[f"{i + 1}:{lb}", vl] for i, (lb, vl) in enumerate(zip(labels, values))],
                        columns=["metric label", "metric value"],
                    ),
                    label="metric label",
                    value="metric value",
                    title=f"{self.phase}_{reload_name}_{k}",
                )
                wandb_info[f"fine_metrics/{self.phase}_{reload_name}_m{k.upper()}"] = sum(values) / len(values)

            # FPS, Params, MACs
            wandb_info[f"efficiency/{self.phase}_{reload_name}_fps"] = fps
            wandb_info[f"efficiency/{self.phase}_{reload_name}_params"] = params
            wandb_info[f"efficiency/{self.phase}_{reload_name}_MACs"] = macs
            wandb_info[f"efficiency/{self.phase}_{reload_name}_FLOPs"] = flops
            wandb.log(wandb_info)

        # Save configirations and metrics to files.
        if utils.local_rank() == 0:
            os.makedirs(os.path.join(self.save_result, self.phase, reload_name), exist_ok=True)
            config_file = os.path.join(self.save_result, self.phase, reload_name, "config.json")
            result_file = os.path.join(self.save_result, self.phase, reload_name, "result.txt")

            with open(config_file, "w") as f:
                f.write(json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=2))
            self.logger.info(f"[{self.phase}] Configurations have been saved to: {config_file}")

            table = prettytable.PrettyTable()
            filed = [""]
            filed.extend(valid_dataloader.dataset.CLASSES)
            table.field_names = filed
            for k, v in valid_metrics.items():
                table.add_row([k, *[f"{x:4.4f}" for x in v]])

            with open(result_file, "w") as f:
                f.write(f"[{self.phase}] Model: {self.model_class}\n")
                f.write("\n")
                f.write(f"[{self.phase}] The {self.phase} losses\n")
                for n, s in valid_losses.items():
                    f.write(f"{n}: {s:4.4f}\n")
                f.write("\n")
                f.write(f"[{self.phase}] The {self.phase} metric table\n")
                f.write(str(table))
                f.write("\n")

            self.logger.info(f"[{self.phase}] Results have been saved to: {result_file}")
            self.logger.info(f"{self.phase} metric table:\n{str(table)}")

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

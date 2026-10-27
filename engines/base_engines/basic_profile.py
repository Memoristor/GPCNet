# coding=utf-8

import os
import time

import thop
import torch
import wandb
from fvcore.nn import FlopCountAnalysis
from torch import distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from tools import op_flops, utils

from .basic_engine import BasicEngine

__all__ = ["BasicProfile"]


class BasicProfile(BasicEngine):
    """The engine is used in the profiling phase.

    Params:
        config: dict. The configuration from runtime environment.
    """

    def __init__(self, *args, **kwargs):
        super(BasicProfile, self).__init__(*args, **kwargs)

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
            # wandb.define_metric("losses/*", step_metric="epoch")
            # wandb.define_metric("fine_metrics/*", step_metric="epoch")
            wandb.define_metric("efficiency/*", step_metric="epoch")

        # Run epoch
        valid_dataloader = self.build_dataloader(self.phase, batch_size=1)  # self.config[self.phase].batch_size)

        # Get FPS, MACs, parameter size
        warmup_iters = 50
        max_iters = 200

        with torch.no_grad():
            device = torch.device(utils.local_rank())
            truth = next(iter(valid_dataloader))
            image = truth["image"].to(device)

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

            if self.config.disable_wandb and utils.local_rank() == 0:
                print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}, FPS: {fps}")

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

            # FPS, Params, MACs
            wandb_info[f"efficiency/{self.phase}_{reload_name}_fps"] = fps
            wandb_info[f"efficiency/{self.phase}_{reload_name}_params"] = params
            wandb_info[f"efficiency/{self.phase}_{reload_name}_MACs"] = macs
            wandb_info[f"efficiency/{self.phase}_{reload_name}_FLOPs"] = flops
            wandb.log(wandb_info)

        # Mark the run as finished
        if utils.local_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()

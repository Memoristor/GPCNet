# coding=utf-8

import argparse
import os
import warnings

import torch
from omegaconf import OmegaConf
from torch import distributed as dist

import engines
from tools import local_rank, seed_init

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment entrance")
    parser.add_argument(
        "-c",
        "--yaml_config",
        type=str,
        help="Yaml configuration file required for the experiment",
    )
    parser.add_argument(
        "-p",
        "--proc_phase",
        type=str,
        default="train",
        help="The current phase of the program",
    )
    parser.add_argument(
        "-s",
        "--convert_syncbn",
        action="store_true",
        help="Whether convert BatchNorm to SyncBatchNorm",
    )
    parser.add_argument(
        "-w",
        "--wandb_project",
        type=str,
        default="Foresight-Pruner",
        help="Project name for wandb",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        default=2024,
        type=int,
        help="Random seed for numpy, torch, random",
    )
    parser.add_argument(
        "-i",
        "--disable_wandb",
        action="store_true",
        help="Do not use wandb to record experimental data",
    )
    parser.add_argument(
        "-d",
        "--delete_history",
        action="store_true",
        help="Whether delete the historical experimental data of the input configuration",
    )
    parser.add_argument(
        "-dc",
        "--delete_checkpoints",
        action="store_true",
        help="Whether delete the historical experimental data of the input configuration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.yaml_config)

    # Add configurations
    config.convert_syncbn = args.convert_syncbn
    config.random_seed = args.random_seed
    config.disable_wandb = args.disable_wandb

    # Set wandb url and api key
    os.environ["WANDB_BASE_URL"] = "YOUR_WANDB_URL"  # Replace with your actual WandB URL
    os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"  # Replace with your actual WandB API key
    
    # Set cuda device to avoid unblanced distributed data parallel
    torch.cuda.set_device(local_rank())

    # Init seeds and process group
    dist.init_process_group(backend="nccl")
    seed_init(config.random_seed + local_rank(), cuda_deterministic=(args.proc_phase == "train"))

    # Run engine
    if args.proc_phase.endswith("_profile"):
        experiment_name = ".".join(os.path.basename(args.yaml_config).split(".")[:-1])
        experiment_name += f"_seed_{config.random_seed}"
        args.proc_phase = args.proc_phase.replace("_profile", "")
        engines.BasicProfile(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config=config,
            phase=args.proc_phase,
            delete_history=args.delete_history,
            delete_checkpoints=args.delete_checkpoints,
        ).run()
    else:
        experiment_name = ".".join(os.path.basename(args.yaml_config).split(".")[:-1])
        experiment_name += f"_seed_{config.random_seed}"
        getattr(engines, config[args.proc_phase].engine.cls)(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config=config,
            phase=args.proc_phase,
            delete_history=args.delete_history,
            delete_checkpoints=args.delete_checkpoints,
            **config[args.proc_phase].engine.get("kwargs", dict()),
        ).run()

    # Cleanup
    dist.destroy_process_group()

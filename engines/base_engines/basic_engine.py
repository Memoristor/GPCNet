# coding=utf-8

import copy
import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple, Union

import timm
import torch
from omegaconf import DictConfig, OmegaConf
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma
from torch import distributed as dist
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import datasets
import models
import schedulers
from datasets import dali_pipeline, transforms
from datasets.base_datasets.classification import ClassificationDataset
from datasets.base_datasets.segmentation import SegmentationDataset
from tools import get_logger, utils

__all__ = ["BasicEngine"]


class BasicEngine:
    """The basic engine is all you need :).

    This engine manages the complete lifecycle of model training, including
    checkpoint loading/saving, distributed training setup, and learning rate
    scheduling. It supports loading both state_dict-based checkpoints and
    complete model object checkpoints.

    The preparation order is designed to handle checkpoints containing complete
    model objects (saved via ``torch.save(model)``) rather than just state_dicts.
    When a complete model is saved (e.g., for sparse/pruned models), the optimizer
    and scheduler must be created after the model is loaded to ensure parameter
    compatibility.

    Important:
        In distributed training, the checkpoint is broadcast to all ranks before
        any component preparation. This ensures that all ranks have identical
        model structures, which is critical when loading pruned/sparse models
        where the architecture may differ from the original configuration.

    Preparation Order:
        1. Directories and logger setup
        2. Load checkpoint file on rank 0
        3. Broadcast checkpoint to all ranks
        4. Model preparation (all ranks use the same checkpoint)
        5. Model EMA preparation
        6. Optimizer creation (must be after model is finalized)
        7. Learning rate scheduler creation (must be after optimizer)
        8. Gradient scaler creation
        9. Load remaining states and finalize setup (SyncBN, DDP)

    Attributes:
        project_name (str): The name of the project for Weights & Biases (wandb).
        experiment_name (str): The name of the experiment within the current project.
        config (DictConfig): The runtime environment configuration.
        phase (str): The current phase of the program (``train``, ``valid``, or ``test``).
        delete_history (bool): Whether to delete historical experiment data.
        model: The neural network model (wrapped in DDP after setup).
        optimizer: The optimizer or dict of optimizers.
        lr_scheduler: The learning rate scheduler or dict of schedulers.
        grad_scaler: The gradient scaler or dict of scalers for mixed precision.
        model_ema: Dict of Model EMA instances.
        start_epoch (int): The starting epoch (loaded from checkpoint or 0).
        is_resumed (bool): Whether training is resumed from a checkpoint.

    Params:
        project_name: The name of the project for Weights & Biases (wandb).
        experiment_name: The name of the experiment within the current project.
        config: The runtime environment configuration as a dictionary.
        phase: The current phase of the program. Options are ``train``, ``valid``, or ``test``.
        delete_history: Determines whether to delete historical experiment data.
            Defaults to False.

    Example:
        >>> config = OmegaConf.load("config.yaml")
        >>> engine = BasicEngine(
        ...     project_name="image_classification",
        ...     experiment_name="resnet50_baseline",
        ...     config=config,
        ...     phase="train",
        ...     delete_history=False
        ... )
        >>> print(f"Starting from epoch {engine.start_epoch}")
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: DictConfig,
        phase: str,
        delete_history: bool = False,
        delete_checkpoints: bool = False,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.phase = phase
        self.delete_history = delete_history
        self.delete_checkpoints = delete_checkpoints

        self.runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Initialize attributes that will be set during preparation
        self.model = None
        self.model_ema: Dict[str, ModelEma] = {}
        self.optimizer: Union[optim.Optimizer, Dict[str, optim.Optimizer], None] = None
        self.lr_scheduler = None
        self.grad_scaler: Dict[str, GradScaler] = {}
        self.start_epoch: int = 0
        self.is_resumed: bool = False

        # Prepare all required attributions
        self._prepare_all()

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: Any = "auto", validate_bf16: bool = False) -> torch.dtype:
        """Resolve ``globals.amp_dtype`` into a torch dtype.

        ``auto`` keeps the previous behavior: prefer bf16 on supported CUDA
        devices and fall back to fp16 otherwise.
        """
        if amp_dtype is None:
            amp_dtype = "auto"

        if isinstance(amp_dtype, torch.dtype):
            dtype = amp_dtype
        else:
            amp_dtype = str(amp_dtype).strip().lower()
            dtype_map = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "half": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }

            if amp_dtype in {"auto", "default"}:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif amp_dtype in dtype_map:
                dtype = dtype_map[amp_dtype]
            else:
                raise ValueError("globals.amp_dtype must be one of: auto, fp16, float16, half, bf16, bfloat16")

        if validate_bf16 and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("globals.amp_dtype='bf16' was requested, but CUDA bf16 is not supported.")

        return dtype

    def _prepare_all(self) -> None:
        """Prepare all required attributions in the correct order.

        The preparation order is critical when dealing with checkpoints that contain
        complete model objects (saved via ``torch.save(model)``) rather than state dicts.
        In such cases, optimizer and scheduler must be created after loading the model
        to ensure parameter compatibility.

        Important:
            The checkpoint is broadcast to all ranks BEFORE any model/optimizer
            preparation. This ensures all ranks have identical model structures,
            which is essential for:
            - Sparse/pruned models saved as complete objects
            - Models with dynamically modified architectures
            - Ensuring optimizer parameter groups match across ranks

        The method orchestrates the following preparation sequence:
            1. Directory structure creation
            2. Logger initialization
            3. Checkpoint file loading (rank 0 only)
            4. Checkpoint broadcast to all ranks
            5. Model preparation (all ranks use broadcasted checkpoint)
            6. Model EMA initialization
            7. Optimizer creation and state loading
            8. Learning rate scheduler creation and state loading
            9. Gradient scaler creation and state loading
            10. Finalize setup (SyncBN, DDP)
        """
        self.enable_amp = self.config.globals.get("enable_amp", False)
        self.amp_dtype = self._resolve_amp_dtype(
            self.config.globals.get("amp_dtype", "auto"),
            validate_bf16=self.enable_amp,
        )

        self.prepare_dir()
        self.prepare_logger()

        checkpoint_path = os.path.join(self.save_checkpoints, self.config[self.phase].reload_pth)
        checkpoint = self._load_checkpoint_file(checkpoint_path)

        # This is critical for ensuring all ranks have identical model structures
        checkpoint = self._broadcast_checkpoint(checkpoint)

        self.prepare_model(checkpoint)
        self.prepare_model_ema(checkpoint)
        self.prepare_optimizer()
        self.prepare_lr_scheduler()
        self.prepare_grad_scaler()

        if checkpoint is not None:
            self.load_optimizer_state(checkpoint)  # type: ignore
            self.load_lr_scheduler_state(checkpoint)  # type: ignore
            try:
                self.load_grad_scaler_state(checkpoint)  # type: ignore
            except Exception:
                self.logger.warning("Failed to load grad_scaler state from checkpoint.")

        self.start_epoch, self.is_resumed = self._finalize_setup(checkpoint)

    def _load_checkpoint_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint file from disk on rank 0.

        This method only loads the checkpoint file without applying any states.
        The checkpoint will be broadcast to other ranks in :meth:`_broadcast_checkpoint`.

        Note:
            Only rank 0 loads the checkpoint to avoid redundant I/O operations.
            The checkpoint is broadcast to other ranks separately.

        Params:
            path: The file path of the checkpoint to load.

        Returns:
            The checkpoint dictionary if the file exists (rank 0), None otherwise.
            The dictionary may contain:
                - ``epoch``: The epoch number when the checkpoint was saved
                - ``model_state_dict`` or ``model``: Model weights or complete model
                - ``optimizer_state_dict``: Optimizer state
                - ``lr_scheduler_state_dict``: Learning rate scheduler state
                - ``grad_scaler_state_dict``: Gradient scaler state
                - ``model_ema_state_dict`` or ``model_ema``: Model EMA weights
        """
        checkpoint = None
        if utils.local_rank() == 0:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location="cpu")
                epoch_info = checkpoint.get("epoch", "unknown")
                self.logger.info(f"Checkpoint file loaded from `{path}`, epoch: {epoch_info}")
            else:
                self.logger.info(f"{self.model_class} checkpoint not found at `{path}`, will start from scratch")
        return checkpoint

    def _broadcast_checkpoint(self, checkpoint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Broadcast checkpoint from rank 0 to all other ranks.

        This method ensures all ranks have identical checkpoint data, which is
        critical for:
            - Loading complete model objects (pruned/sparse models)
            - Ensuring identical model structures across ranks
            - Proper optimizer state loading on all ranks

        Params:
            checkpoint: The checkpoint dictionary (only valid on rank 0).

        Returns:
            The checkpoint dictionary on all ranks, or None if no checkpoint exists.
        """
        assert dist.is_initialized(), "torch.distributed is not initialized."

        # Broadcast checkpoint to all ranks using object broadcast
        # This ensures all ranks have identical checkpoint data
        object_list = [checkpoint]
        dist.broadcast_object_list(object_list, src=0)
        checkpoint = object_list[0]

        # Log on non-zero ranks
        if utils.local_rank() != 0 and checkpoint is not None:
            epoch_info = checkpoint.get("epoch", "unknown")
            self.logger.info(f"Received checkpoint via broadcast, epoch: {epoch_info}")

        return checkpoint

    def prepare_dir(self) -> None:
        """Prepare and create required directories.

        Creates the following directory structure under the output root::

            {output_root}/
            └── {project_name}/
                └── {dataset_class}/
                    └── {model_class}/
                        └── {experiment_name}/
                            ├── checkpoints/
                            ├── wandb/
                            ├── logs/
                            └── results/

        If ``delete_history`` is True, existing directories will be deleted
        after a 15-second countdown (allowing user intervention).

        Warning:
            When ``delete_history=True``, all previous experiment data including
            checkpoints, logs, and results will be permanently deleted.
        """
        if "cls" in self.config.model:
            self.model_class = self.config.model.cls
        elif "timm" in self.config.model:
            self.model_class = self.config.model.timm.model_name
        else:
            raise ValueError("Cannot determine model class from configuration.")

        self.dataset_class = self.config.dataset.cls

        output_root = self.config.globals.output_root
        output_dir = os.path.join(
            output_root,
            self.project_name,
            self.dataset_class,
            self.model_class,
            self.experiment_name,
        )

        self.save_checkpoints = os.path.join(output_dir, "checkpoints")
        self.save_wandb = os.path.join(output_dir, "wandb")
        self.save_logger = os.path.join(output_dir, "logs")
        self.save_result = os.path.join(output_dir, "results")

        if self.delete_history:
            if utils.local_rank() == 0:
                print(
                    "Warning! Historical experimental data will be deleted in 15 seconds! "
                    "Press Ctrl+C if you want this stop!"
                )
                for remaining in range(15, 0, -1):
                    print(f"\rCounting down: {remaining} seconds", end="")
                    time.sleep(1)

                shutil.rmtree(self.save_checkpoints, ignore_errors=True)
                shutil.rmtree(self.save_wandb, ignore_errors=True)
                shutil.rmtree(self.save_logger, ignore_errors=True)
                shutil.rmtree(self.save_result, ignore_errors=True)

        if self.delete_checkpoints:
            if utils.local_rank() == 0:
                print(
                    "Warning! Historical checkpoints will be deleted in 15 seconds! Press Ctrl+C if you want this stop!"
                )
                for remaining in range(15, 0, -1):
                    print(f"\rCounting down: {remaining} seconds", end="")
                    time.sleep(1)

                shutil.rmtree(self.save_checkpoints, ignore_errors=True)

        dist.barrier()  # Ensure all ranks wait for directory preparation

        os.makedirs(self.save_checkpoints, exist_ok=True)
        os.makedirs(self.save_wandb, exist_ok=True)
        os.makedirs(self.save_logger, exist_ok=True)
        os.makedirs(self.save_result, exist_ok=True)

    def prepare_logger(self) -> None:
        """Prepare and initialize the logger.

        Creates a logger instance that writes to a file named with the rank
        and runtime information. The log level is determined by
        ``config.globals.log_level``.
        """
        self.logger = get_logger.Logger(
            filename=os.path.join(self.save_logger, f"rank_{utils.local_rank()}_runtime_{self.runtime}.log"),
            level=self.config.globals.log_level,
        ).logger

    def prepare_model(self, checkpoint: Optional[Dict[str, Any]] = None) -> None:
        """Prepare the model, optionally loading from checkpoint.

        This method handles two scenarios:
            1. If checkpoint contains a complete model object (``model`` key),
               the model is loaded directly from the checkpoint.
            2. Otherwise, a new model is created from the configuration,
               and state_dict is loaded if available.

        Important:
            Since the checkpoint has been broadcast to all ranks before this
            method is called, all ranks will have identical model structures.
            This is essential for proper DDP synchronization.

        Note:
            For sparse/pruned models saved as complete objects, using the
            ``model`` key in checkpoint ensures architecture compatibility.
            The optimizer will be created based on this loaded model's parameters.

        Params:
            checkpoint: Optional checkpoint dictionary (available on all ranks
                after broadcast). If provided and contains a ``model`` key,
                the complete model object will be used. If it contains
                ``model_state_dict``, those weights will be loaded into a
                newly created model.
        """
        model_loaded_from_checkpoint = False

        if checkpoint is not None:
            if "model" in checkpoint:
                # All ranks use the same model object from checkpoint
                self.model = copy.deepcopy(checkpoint["model"])
                model_loaded_from_checkpoint = True
                self.logger.info(f"{self.model_class} loaded complete model object from checkpoint")

        # If model not loaded from checkpoint, create a new one
        if not model_loaded_from_checkpoint:
            if "cls" in self.config.model:
                self.model = getattr(models, self.model_class)(**self.config.model.get("kwargs", {}))
            elif "timm" in self.config.model:
                self.model = timm.create_model(**OmegaConf.to_container(self.config.model.timm))
            else:
                raise ValueError("Cannot determine model class from configuration.")

            # Load state_dict if available
            if checkpoint is not None and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                self.logger.info(f"{self.model_class} loaded model state_dict from checkpoint")

        # Move model to device
        device = torch.device(utils.local_rank())
        self.model = self.model.to(device)

    def prepare_model_ema(self, checkpoint: Optional[Dict[str, Any]] = None) -> None:
        """Prepare Model Exponential Moving Average (EMA) instances.

        Creates ModelEma instances for phases that have ``model_ema`` configuration.
        If a checkpoint is provided, EMA states are loaded from it.

        Note:
            Model EMA is created after the model is fully prepared to ensure
            it captures the correct model state. Since the checkpoint is
            available on all ranks, EMA states are loaded identically.

        Params:
            checkpoint: Optional checkpoint dictionary containing EMA states
                under ``model_ema_state_dict`` or ``model_ema`` keys.
        """
        self.model_ema = {}

        for key in self.config.keys():
            if isinstance(self.config[key], (dict, DictConfig)) and "model_ema" in self.config[key].keys():
                self.model_ema[key] = ModelEma(
                    self.model,
                    decay=self.config[key].model_ema.ema_decay,
                    device=torch.device(utils.local_rank()),
                    resume="",
                )

        # Load EMA states from checkpoint (checkpoint is available on all ranks)
        if checkpoint is not None and self.model_ema:
            if "model_ema_state_dict" in checkpoint:
                for k, v in self.model_ema.items():
                    if k in checkpoint["model_ema_state_dict"]:
                        v.ema.load_state_dict(checkpoint["model_ema_state_dict"][k], strict=True)
                    else:
                        self.logger.warning(
                            f"{self.model_class} cannot reload {k} of model EMA, key not exist in checkpoint"
                        )
            elif "model_ema" in checkpoint:
                for k, v in self.model_ema.items():
                    if k in checkpoint["model_ema"]:
                        # Replace EMA model with the one from checkpoint
                        v.ema = copy.deepcopy(checkpoint["model_ema"][k])
                        v.ema = v.ema.to(torch.device(utils.local_rank()))
                    else:
                        self.logger.warning(
                            f"{self.model_class} cannot reload {k} of model EMA, key not exist in checkpoint"
                        )
            else:
                # Initialize EMA with current model state
                for k, v in self.model_ema.items():
                    v.ema.load_state_dict(self.model.state_dict(), strict=True)

        elif self.model_ema:
            # No checkpoint, initialize EMA with current model state
            for k, v in self.model_ema.items():
                v.ema.load_state_dict(self.model.state_dict(), strict=True)

    def prepare_optimizer(self) -> None:
        """Prepare the optimizer(s) based on the current model.

        This method supports multiple optimizer configurations:
            1. Global optimizer defined in ``config.optimizer``
            2. Per-phase optimizers defined in ``config[phase].optimizer``
            3. Module-specific learning rates via ``config.optimizer.modules``
            4. timm-style optimizer creation via ``config.optimizer.timm``

        Important:
            This method must be called after :meth:`prepare_model` to ensure
            the optimizer is created with the correct model parameters,
            especially when loading sparse/pruned models from checkpoints.
            Since all ranks have identical model structures (via checkpoint
            broadcast), optimizer parameter groups will match across ranks.

        Params:
            checkpoint: Optional checkpoint dictionary containing optimizer state
                under ``optimizer_state_dict`` key.
        """
        if "optimizer" in self.config.keys():
            timm_cfg = self.config.optimizer.get("timm", None)
            if timm_cfg is not None:
                self.optimizer = create_optimizer(timm_cfg, self.model)
            else:
                if "modules" in self.config.optimizer.keys():
                    default_lr = self.config.optimizer.kwargs.lr
                    default_wd = (
                        self.config.optimizer.kwargs.weight_decay
                        if "weight_decay" in self.config.optimizer.kwargs.keys()
                        else 0.0
                    )

                    params = []
                    params_ids = []
                    for module in self.config.optimizer.modules:
                        assert isinstance(module, (dict, DictConfig)) and len(module) == 1, (
                            f"Invalid module config: {module}"
                        )
                        module_name = str(list(module.keys())[0])

                        module_lr = module[module_name].get("lr", default_lr)
                        module_wd = module[module_name].get("weight_decay", default_wd)

                        m = self.model
                        for name in module_name.split("."):
                            m = getattr(m, name)

                        params.append({"params": m.parameters(), "lr": module_lr, "weight_decay": module_wd})
                        params_ids += list(id(p) for p in m.parameters())

                    other_params = []
                    for p in self.model.parameters():
                        if id(p) not in params_ids:
                            other_params.append(p)

                    params.append({"params": other_params, "lr": default_lr, "weight_decay": default_wd})
                    self.optimizer = getattr(optim, self.config.optimizer.cls)(params=params)

                    if utils.local_rank() == 0:
                        self.logger.info("Optimizer parameter groups:")
                        for i, param_group in enumerate(self.optimizer.param_groups):
                            group_lr = param_group["lr"]
                            group_wd = param_group["weight_decay"]
                            group_size = len(param_group["params"])
                            self.logger.info(f"Group {i}: lr={group_lr}, weight_decay={group_wd}, size={group_size}")
                else:
                    self.optimizer = getattr(optim, self.config.optimizer.cls)(
                        params=self.model.parameters(),
                        **self.config.optimizer.get("kwargs", {}),
                    )

        else:
            self.optimizer = {}
            for key in self.config.keys():
                if isinstance(self.config[key], (dict, DictConfig)) and "optimizer" in self.config[key].keys():
                    timm_cfg = self.config[key].optimizer.get("timm", None)
                    if timm_cfg is not None:
                        self.optimizer[key] = create_optimizer(timm_cfg, self.model)
                    else:
                        if "modules" in self.config[key].optimizer.keys():
                            default_lr = self.config[key].optimizer.kwargs.lr
                            default_wd = (
                                self.config[key].optimizer.kwargs.weight_decay
                                if "weight_decay" in self.config[key].optimizer.kwargs.keys()
                                else 0.0
                            )

                            params = []
                            params_ids = []
                            for module in self.config[key].optimizer.modules:
                                assert isinstance(module, (dict, DictConfig)) and len(module) == 1, (
                                    f"Invalid module config: {module}"
                                )
                                module_name = str(list(module.keys())[0])

                                module_lr = module[module_name].get("lr", default_lr)
                                module_wd = module[module_name].get("weight_decay", default_wd)

                                m = self.model
                                for name in module_name.split("."):
                                    m = getattr(m, name)

                                params.append({"params": m.parameters(), "lr": module_lr, "weight_decay": module_wd})
                                params_ids += list(id(p) for p in m.parameters())

                            other_params = []
                            for p in self.model.parameters():
                                if id(p) not in params_ids:
                                    other_params.append(p)

                            params.append({"params": other_params, "lr": default_lr, "weight_decay": default_wd})
                            self.optimizer[key] = getattr(optim, self.config[key].optimizer.cls)(params=params)

                            if utils.local_rank() == 0:
                                self.logger.info(f"Optimizer parameter groups for {key}:")
                                for i, param_group in enumerate(self.optimizer[key].param_groups):
                                    group_lr = param_group["lr"]
                                    group_wd = param_group["weight_decay"]
                                    group_size = len(param_group["params"])
                                    self.logger.info(
                                        f"Group {i}: lr={group_lr}, weight_decay={group_wd}, size={group_size}"
                                    )
                        else:
                            self.optimizer[key] = getattr(optim, self.config[key].optimizer.cls)(
                                params=self.model.parameters(),
                                **self.config[key].optimizer.get("kwargs", {}),
                            )

    def load_optimizer_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""

        if "optimizer" in self.config.keys():
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.logger.info(f"{self.model_class} loaded optimizer state from checkpoint")

        else:
            if "optimizer_state_dict" in checkpoint:
                for k, v in self.optimizer.items():
                    if k in checkpoint["optimizer_state_dict"]:
                        v.load_state_dict(checkpoint["optimizer_state_dict"][k])
                        self.logger.info(f"{self.model_class} loaded {k} optimizer state from checkpoint")
                    else:
                        self.logger.warning(
                            f"{self.model_class} cannot reload {k} optimizer, key not exist in checkpoint"
                        )

    def prepare_lr_scheduler(self) -> None:
        """Prepare learning rate scheduler(s) based on the optimizer(s).

        This method supports multiple scheduler configurations:
            1. Global scheduler defined in ``config.learning_rate``
            2. Per-phase schedulers defined in ``config[phase].learning_rate``
            3. Warmup + main scheduler via ``SequentialLR``
            4. timm-style scheduler creation via ``config.learning_rate.timm``

        Important:
            This method must be called after :meth:`prepare_optimizer` to ensure
            the scheduler is created with the correct optimizer reference.

        Params:
            checkpoint: Optional checkpoint dictionary containing scheduler state
                under ``lr_scheduler_state_dict`` key.
        """
        if "learning_rate" in self.config.keys():
            assert not isinstance(self.optimizer, dict), (
                "When the learning rate scheduler is specified in the global scope, "
                "the optimizer should also be specified in the global scope."
            )

            timm_cfg = self.config.learning_rate.get("timm", None)
            if timm_cfg is not None:
                timm_cfg.epochs = self.config.train.num_epoch
                self.lr_scheduler, _ = create_scheduler(timm_cfg, self.optimizer)
            else:
                if self.config.learning_rate.warmup_steps > 0:
                    self.lr_scheduler = SequentialLR(
                        optimizer=self.optimizer,
                        schedulers=[
                            LinearLR(
                                self.optimizer,  # type: ignore
                                start_factor=0.01,
                                end_factor=1.0,
                                total_iters=self.config.learning_rate.warmup_steps,
                            ),
                            getattr(schedulers, self.config.learning_rate.cls)(
                                optimizer=self.optimizer,
                                **self.config.learning_rate.get("kwargs", {}),
                            ),
                        ],
                        milestones=[self.config.learning_rate.warmup_steps],
                    )
                else:
                    self.lr_scheduler = getattr(schedulers, self.config.learning_rate.cls)(
                        optimizer=self.optimizer,
                        **self.config.learning_rate.get("kwargs", {}),
                    )

        else:
            assert isinstance(self.optimizer, dict), (
                "When the learning rate scheduler is specified in the phase scope, "
                "the optimizer should also be specified in the phase scope."
            )

            self.lr_scheduler = {}
            for key in self.config.keys():
                if isinstance(self.config[key], (dict, DictConfig)) and "learning_rate" in self.config[key].keys():
                    timm_cfg = self.config[key].learning_rate.get("timm", None)
                    if timm_cfg is not None:
                        timm_cfg.epochs = self.config[key].num_epoch
                        lr_scheduler, _ = create_scheduler(timm_cfg, self.optimizer[key])
                        self.lr_scheduler[key] = lr_scheduler
                    else:
                        if self.config[key].learning_rate.warmup_steps > 0:
                            self.lr_scheduler[key] = SequentialLR(
                                optimizer=self.optimizer[key],
                                schedulers=[
                                    LinearLR(
                                        self.optimizer[key],
                                        start_factor=0.01,
                                        end_factor=1.0,
                                        total_iters=self.config[key].learning_rate.warmup_steps,
                                    ),
                                    getattr(schedulers, self.config[key].learning_rate.cls)(
                                        optimizer=self.optimizer[key],
                                        **self.config[key].learning_rate.get("kwargs", {}),
                                    ),
                                ],
                                milestones=[self.config[key].learning_rate.warmup_steps],
                            )
                        else:
                            self.lr_scheduler[key] = getattr(schedulers, self.config[key].learning_rate.cls)(
                                optimizer=self.optimizer[key],
                                **self.config[key].learning_rate.get("kwargs", {}),
                            )

    def load_lr_scheduler_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load learning rate scheduler state from checkpoint."""

        if "learning_rate" in self.config.keys():
            # Load lr_scheduler state from checkpoint
            if "lr_scheduler_state_dict" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                self.logger.info(f"{self.model_class} loaded lr_scheduler state from checkpoint")

        else:
            # Load lr_scheduler states from checkpoint
            if "lr_scheduler_state_dict" in checkpoint:
                for k, v in self.lr_scheduler.items():
                    if k in checkpoint["lr_scheduler_state_dict"]:
                        v.load_state_dict(checkpoint["lr_scheduler_state_dict"][k])
                        self.logger.info(f"{self.model_class} loaded {k} lr_scheduler state from checkpoint")
                    else:
                        self.logger.warning(
                            f"{self.model_class} cannot reload {k} lr_scheduler, key not exist in checkpoint"
                        )

    def prepare_grad_scaler(self) -> None:
        """Prepare gradient scaler(s) for mixed precision training.

        Creates GradScaler instances for phases that have ``enable_amp``
        configuration set to True.

        Params:
            checkpoint: Optional checkpoint dictionary containing scaler state
                under ``grad_scaler_state_dict`` key.
        """
        self.grad_scaler = {}
        for key in self.config.keys():
            if isinstance(self.config[key], (dict, DictConfig)):
                self.grad_scaler[key] = GradScaler()
                self.grad_scaler[key]._enabled = self.enable_amp and self.amp_dtype is torch.float16
                # self.logger.info(f"{self.model_class} GradScaler for {key} enabled: {self.grad_scaler[key]._enabled}")

    def load_grad_scaler_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load gradient scaler state from checkpoint."""

        if "grad_scaler_state_dict" in checkpoint and self.grad_scaler:
            for k, v in self.grad_scaler.items():
                if k in checkpoint["grad_scaler_state_dict"]:
                    v.load_state_dict(checkpoint["grad_scaler_state_dict"][k])
                    self.logger.info(f"{self.model_class} loaded {k} grad_scaler state from checkpoint")
                else:
                    self.logger.warning(
                        f"{self.model_class} cannot reload {k} grad_scaler, key not exist in checkpoint"
                    )

    def _finalize_setup(self, checkpoint: Optional[Dict[str, Any]]) -> Tuple[int, bool]:
        """Finalize setup by applying distributed wrappers and moving states to device.

        This method performs the final setup steps:
            1. Determine epoch and is_resumed status
            2. Handle model EMA for inference if configured
            3. Move optimizer states to correct device
            4. Convert BatchNorm to SyncBatchNorm if configured
            5. Wrap model in DistributedDataParallel

        Note:
            Optimizer, lr_scheduler, and grad_scaler states are already loaded
            in their respective prepare methods. This method only handles device
            placement and DDP wrapping.

        Params:
            checkpoint: Optional checkpoint dictionary containing saved states
                (available on all ranks after broadcast).

        Returns:
            A tuple of (start_epoch, is_resumed):
                - start_epoch: The epoch to start/resume training from.
                - is_resumed: Whether training is resumed from a checkpoint.
        """
        epoch = 0
        is_resumed = False

        if checkpoint is not None:
            epoch = checkpoint.get("epoch", 0)
            is_resumed = True

            self.logger.info(f"{self.model_class} resuming from epoch {epoch}")

            # Use model EMA to initialize the model if needed
            if self.config[self.phase].get("use_model_ema", False):
                if "model_ema_state_dict" in checkpoint:
                    ema_key = self.config[self.phase].get("use_model_ema_key", "train")
                    if ema_key in checkpoint["model_ema_state_dict"]:
                        self.model.load_state_dict(checkpoint["model_ema_state_dict"][ema_key], strict=True)
                        self.logger.info(
                            f"{self.model_class} loaded weights from model EMA ({ema_key}) from checkpoint"
                        )

        # Move optimizer states to correct device if needed
        device = torch.device(utils.local_rank())
        if self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                for k, v in self.optimizer.items():
                    self._move_optimizer_to_device(v, device)
            else:
                self._move_optimizer_to_device(self.optimizer, device)

        # Convert BatchNorm to SyncBatchNorm if configured
        if self.config.convert_syncbn:
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap model in DistributedDataParallel
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[utils.local_rank()],
            output_device=utils.local_rank(),
            find_unused_parameters=self.config.globals.get("find_unused_parameters", False),
        )

        return epoch, is_resumed

    def _move_optimizer_to_device(self, optimizer: optim.Optimizer, device: torch.device) -> None:
        """Move optimizer states to the specified device.

        This is necessary because optimizer states loaded from a checkpoint
        may be on CPU, but they need to be on the same device as the model
        parameters for training.

        Params:
            optimizer: The optimizer whose states need to be moved.
            device: The target device.
        """
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    @property
    def _optimizer(self):
        """Get optimizer for the current phase.

        Returns:
            The optimizer for the current phase if using per-phase optimizers,
            the global optimizer otherwise, or None if not configured.
        """
        if isinstance(self.optimizer, dict):
            if self.phase in self.optimizer.keys():
                return self.optimizer[self.phase]
            else:
                return None
        else:
            return self.optimizer

    @property
    def _lr_scheduler(self):
        """Get learning rate scheduler for the current phase.

        Returns:
            The scheduler for the current phase if using per-phase schedulers,
            the global scheduler otherwise, or None if not configured.
        """
        if isinstance(self.lr_scheduler, dict):
            if self.phase in self.lr_scheduler.keys():
                return self.lr_scheduler[self.phase]
            else:
                return None
        else:
            return self.lr_scheduler

    @property
    def _grad_scaler(self):
        """Get gradient scaler for the current phase.

        Returns:
            The scaler for the current phase if using per-phase scalers,
            the global scaler otherwise, or None if not configured.
        """
        if isinstance(self.grad_scaler, dict):
            if self.phase in self.grad_scaler.keys():
                return self.grad_scaler[self.phase]
            else:
                return None
        else:
            return self.grad_scaler

    @property
    def _model_ema(self):
        """Get model EMA for the current phase.

        Returns:
            The ModelEma instance for the current phase if configured,
            None otherwise.
        """
        if isinstance(self.model_ema, dict):
            if self.phase in self.model_ema.keys():
                return self.model_ema[self.phase]
            else:
                return None
        else:
            return self.model_ema

    def check_model_synced(self, model=None) -> None:
        """Check whether the model parameters are synchronized across ranks.

        This is a debugging utility to verify that all ranks have identical
        model parameters after synchronization.

        Params:
            model: The model to check. If None, uses ``self.model``.

        Raises:
            AssertionError: If any parameters differ across ranks.
        """
        if model is None:
            model = self.model

        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        not_synced = []
        for name, param in model.named_parameters():
            param_list = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
            dist.all_gather(param_list, param)
            if not all(torch.allclose(p, param_list[0]) for p in param_list):
                not_synced.append(name)

        assert len(not_synced) == 0, (
            f"Model parameters are not synchronized across ranks. Unsynced parameters: {', '.join(not_synced)}"
        )

    def save(self, path: str, epoch: int, save_full_model: bool = False) -> None:
        """Save checkpoint including epoch, model, optimizer, and scheduler states.

        Params:
            path: The file path where the checkpoint will be saved.
            epoch: The current epoch number.
            save_full_model: If True, saves the complete model object instead
                of just the state_dict. This is useful for sparse/pruned models
                where the architecture may have changed.

        Note:
            Only rank 0 saves the checkpoint to avoid file conflicts.
            When ``save_full_model=True``, the checkpoint will contain the
            complete model object under the ``model`` key, which can be
            loaded directly without recreating the model architecture.
        """
        if utils.local_rank() == 0:
            state_dict = {"epoch": epoch}

            # Save model state dict
            if not save_full_model:
                if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                    state_dict["model_state_dict"] = self.model.module.state_dict()
                else:
                    state_dict["model_state_dict"] = self.model.state_dict()
            else:
                if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                    state_dict["model"] = copy.deepcopy(self.model.module).to("cpu")
                else:
                    state_dict["model"] = copy.deepcopy(self.model).to("cpu")

            # Save model ema state dict
            if self.model_ema:
                if not save_full_model:
                    state_dict["model_ema_state_dict"] = {k: v.ema.state_dict() for k, v in self.model_ema.items()}
                else:
                    state_dict["model_ema"] = {k: copy.deepcopy(v.ema).to("cpu") for k, v in self.model_ema.items()}

            # Save optimizer state dict
            if self.optimizer is not None:
                if isinstance(self.optimizer, dict):
                    state_dict["optimizer_state_dict"] = {k: v.state_dict() for k, v in self.optimizer.items()}
                else:
                    state_dict["optimizer_state_dict"] = self.optimizer.state_dict()

            # Save learning rate scheduler state dict
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, dict):
                    state_dict["lr_scheduler_state_dict"] = {k: v.state_dict() for k, v in self.lr_scheduler.items()}
                else:
                    state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

            # Save grad scaler state dict
            if self.grad_scaler:
                state_dict["grad_scaler_state_dict"] = {k: v.state_dict() for k, v in self.grad_scaler.items()}

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state_dict, path)

    def build_dataloader(
        self,
        phase: str,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        disable_pipeline: bool = False,
    ) -> DataLoader:
        """Build distributed Pytorch DataLoader or DALI Pipeloader.

        Params:
            phase: The phase for dataset configurations (e.g., ``train``, ``valid``).
            batch_size: The batch size per GPU. If None, uses
                ``config[phase].batch_size / world_size``.
            num_samples: The number of samples to randomly sample from the dataset.
                If None, uses all samples.
            disable_pipeline: If True, disables DALI pipeline even when available,
                falling back to PyTorch DataLoader.

        Returns:
            An iterable data loader (either PyTorch DataLoader or DALI Pipeloader).

        Raises:
            AssertionError: If the dataset is empty.
            NotImplementedError: If DALI pipeline is requested for an unsupported
                dataset type.
        """
        # Build transforms
        config = self.config[phase]
        transformer = None
        if "transforms" in config.dataset.keys():
            transformer = getattr(transforms, config.dataset.transforms.func)(
                **config.dataset.transforms.get("kwargs", {})
            )

        # Build dataset
        dataset = None
        if utils.local_rank() == 0:
            dataset_cls = self.config.dataset.cls
            root_path = self.config.dataset.root_path

            # Override dataset class if specified in phase config
            if "cls" in config.dataset.keys():
                dataset_cls = config.dataset.cls
                root_path = config.dataset.root_path

            dataset = getattr(datasets, dataset_cls)(
                root_path=root_path,
                transforms=transformer,
                **config.dataset.get("kwargs", {}),
            )
            assert len(dataset) > 0, "The dataset is empty."

        # Broadcast dataset to all ranks
        object_list = [dataset if utils.local_rank() == 0 else None]
        dist.broadcast_object_list(object_list, src=0)
        dataset = object_list[0]

        # Check if the dataset length is consistent across ranks
        if dist.is_initialized():
            len_dataset = len(dataset)
            len_dataset_tensor = torch.tensor(len_dataset, device=torch.device(utils.local_rank()))
            dist.all_reduce(len_dataset_tensor, op=dist.ReduceOp.MAX)
            assert len_dataset == len_dataset_tensor.item(), (
                f"Rank {utils.local_rank()} has mismatched dataset size: "
                f"{len_dataset} vs {len_dataset_tensor.item()} (MAX)"
            )

        # Build iterable dataloader
        num_workers = self.config.dataset.num_workers

        # Override num workers if specified in phase config
        if "num_workers" in config.dataset.keys():
            num_workers = config.dataset.num_workers

        if batch_size is None:
            batch_size = config.batch_size // dist.get_world_size()

        if "pipeline" in config.dataset.keys() and not disable_pipeline:
            if isinstance(dataset, ClassificationDataset):
                files = dataset.image_path
                if num_samples is not None:
                    indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]
                    files = [files[i] for i in indices]
                labels = [dataset.load_label(f) for f in files]

                pipeline_kwargs = config.dataset.pipeline.get("kwargs", {})
                if len(pipeline_kwargs) > 0:
                    pipeline_kwargs = OmegaConf.to_container(pipeline_kwargs)

                pipeline = getattr(dali_pipeline, config.dataset.pipeline.func)(
                    batch_size=batch_size,
                    num_threads=num_workers,
                    device_id=utils.local_rank(),
                    files=files,
                    labels=labels,
                    num_shards=dist.get_world_size(),
                    shard_id=utils.local_rank(),
                    random_seed=self.config.random_seed + utils.local_rank(),
                    **pipeline_kwargs,
                )
                pipeline.build()

                return dali_pipeline.ImagenetPipeloader(
                    dataset=dataset, pipeline=pipeline, reader_name=f"Reader{utils.local_rank()}"
                )

            elif isinstance(dataset, SegmentationDataset):
                indices = None
                if num_samples is not None:
                    indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]

                pipeline_kwargs = config.dataset.pipeline.get("kwargs", {})
                if len(pipeline_kwargs) > 0:
                    pipeline_kwargs = OmegaConf.to_container(pipeline_kwargs)

                pipeline = getattr(dali_pipeline, config.dataset.pipeline.func)(
                    dataset=dataset,
                    batch_size=batch_size,
                    batch_size_extsrc=batch_size,
                    sampling_indices=indices,
                    num_threads=num_workers,
                    device_id=utils.local_rank(),
                    num_shards=dist.get_world_size(),
                    shard_id=utils.local_rank(),
                    shuffle_seed=self.config.random_seed,
                    py_start_method="spawn",
                    **pipeline_kwargs,
                )
                pipeline.build()

                return dali_pipeline.SemSegPipeloader(dataset=dataset, pipeline=pipeline)

            else:
                raise NotImplementedError(f"DALI pipeline cannot be used for {dataset.__class__.__name__}")

        else:
            if num_samples is not None:
                indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]
                dataset = Subset(dataset, indices)

            return DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                sampler=DistributedSampler(dataset),
                pin_memory=True,
            )

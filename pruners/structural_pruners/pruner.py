# coding=utf-8

import gc
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

__all__ = ["ModuleSparsityPruner"]


@dataclass
class ModulePruningResult:
    """Container for module-specific pruning operation results."""

    success: bool
    module_original_params: int
    module_final_params: int
    module_target_sparsity: float
    module_achieved_sparsity: float
    model_original_params: int
    model_final_params: int
    model_achieved_sparsity: float
    channel_ratio_used: float
    search_iterations: int
    message: str = ""
    search_history: List[Dict] = field(default_factory=list)


class ModuleSparsityPruner:
    """
    Pruner that targets a specific module to achieve desired parameter sparsity.

    This pruner only prunes layers within the specified target module while
    keeping all other parts of the model unchanged. It uses binary search to
    find the channel pruning ratio that achieves the target parameter sparsity
    for the specified module.

    Key Features:
        - Only prunes the specified target module
        - Automatically ignores all layers outside the target module
        - Supports ignoring specific layers within the target module
        - Binary search based on target module's parameter sparsity
        - Preserves original model until final pruning

    Params:
        model: The complete model containing the target module.
        target_module: The specific module to prune (must be a submodule of model).
        example_inputs: Example input tensor for model tracing.
        importance: Importance estimator from torch-pruning or custom.
        pruning_ratio: Target parameter sparsity ratio for the target module (0.0 ~ 1.0).
        ignored_layers: Layers within target_module to ignore during pruning.
        global_pruning: Whether to use global pruning within target module. Default: True.
        channel_sparsity_range: Tuple of (min, max) channel sparsity to search. Default: (0.0, 0.9).
        max_iterations: Maximum iterations for binary search. Default: 20.
        tolerance: Acceptable tolerance for parameter sparsity. Default: 0.001.
        round_to: Round number of channels to multiple. Default: 1.
        isomorphic: Enable isomorphic pruning. Default: False. https://arxiv.org/abs/2407.04616

    Example:
        >>> model = SomeModelWithEncoder()
        >>> # Only prune the encoder module to 50% parameter sparsity
        >>> pruner = ModuleSparsityPruner(
        ...     model=model,
        ...     target_module=model.encoder,  # Only prune encoder
        ...     example_inputs=torch.randn(1, 3, 224, 224),
        ...     importance=tp.importance.MagnitudeImportance(p=2),
        ...     pruning_ratio=0.5,
        ...     ignored_layers=[model.encoder.final_layer],  # Ignore within encoder
        ... )
        >>> result = pruner.step()
        >>> print(f"Encoder sparsity: {result.module_achieved_sparsity:.2%}")
    """

    def __init__(
        self,
        model: nn.Module,
        target_module: nn.Module,
        example_inputs: torch.Tensor,
        importance: tp.importance.Importance,
        pruning_ratio: float = 0.5,
        ignored_layers: Optional[List[nn.Module]] = None,
        global_pruning: bool = True,
        channel_sparsity_range: Tuple[float, float] = (0.0, 1.0),
        max_iterations: int = 50,
        tolerance: float = 0.005,
        round_to: int = 1,
        isomorphic: bool = False,
        ignore_mlp_fc2: bool = False,
    ):
        # Validate inputs
        if not 0.0 < pruning_ratio < 1.0:
            raise ValueError(f"pruning_ratio must be in (0, 1), got {pruning_ratio}")

        if channel_sparsity_range[0] >= channel_sparsity_range[1]:
            raise ValueError("channel_sparsity_range[0] must be < channel_sparsity_range[1]")

        # Verify target_module is part of model
        if not self._is_submodule(model, target_module):
            raise ValueError("target_module must be a submodule of model")

        self.model = model
        self.target_module = target_module
        self.example_inputs = example_inputs
        self.importance = importance
        self.pruning_ratio = pruning_ratio
        self.ignored_layers_in_target = ignored_layers or []
        self.global_pruning = global_pruning
        self.channel_sparsity_range = channel_sparsity_range
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.round_to = round_to
        self.isomorphic = isomorphic
        self.ignore_mlp_fc2 = ignore_mlp_fc2

        # Store original model state
        self._original_model_state = deepcopy(model.state_dict())

        # Get target module name and build mappings
        self._target_module_name = self._get_module_name(model, target_module)
        self._ignored_layer_names_in_target = self._get_layer_names(model, self.ignored_layers_in_target)

        # Calculate original parameters
        self._original_model_params = self._count_params(model)
        self._original_module_params = self._count_params(target_module)

        # Build list of layers outside target module (to be ignored)
        self._outside_layer_names = self._get_layers_outside_module(model, target_module)

        # Search state
        self._search_history: List[Dict] = []

    @staticmethod
    def _is_submodule(model: nn.Module, target: nn.Module) -> bool:
        """Check if target is a submodule of model."""
        if model is target:
            return True
        for module in model.modules():
            if module is target:
                return True
        return False

    @staticmethod
    def _get_module_name(model: nn.Module, target: nn.Module) -> str:
        """Get the name of target module within model."""
        if model is target:
            return ""
        for name, module in model.named_modules():
            if module is target:
                return name
        raise ValueError("Target module not found in model")

    @staticmethod
    def _count_params(module: nn.Module) -> int:
        """Count total trainable parameters in module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    @staticmethod
    def _get_layer_names(model: nn.Module, layers: List[nn.Module]) -> List[str]:
        """Get names of specified layers."""
        layer_ids = {id(layer) for layer in layers}
        names = []
        for name, module in model.named_modules():
            if id(module) in layer_ids:
                names.append(name)
        return names

    @staticmethod
    def _collect_num_heads(model: nn.Module) -> dict[nn.Module, int]:
        """collect number of heads for modules with qkv attribute"""
        num_heads_dict = {}
        for name, m in model.named_modules():
            if hasattr(m, "qkv"):
                assert hasattr(m, "num_heads"), f"Module {name} has qkv but no head_dim attribute."
                num_heads = m.num_heads
                num_heads_dict[m.qkv] = num_heads

            if hasattr(m, "q") and hasattr(m, "k") and hasattr(m, "v"):
                assert hasattr(m, "num_heads"), f"Module {name} has q/k/v but no num_heads attribute."
                num_heads = m.num_heads
                num_heads_dict[m.q] = num_heads
                num_heads_dict[m.k] = num_heads
                num_heads_dict[m.v] = num_heads

        # print(num_heads_dict)
        return num_heads_dict

    def _get_layers_outside_module(self, model: nn.Module, target_module: nn.Module) -> List[str]:
        """Get names of all prunable layers outside the target module."""
        # Get all layer names inside target module
        target_layer_ids: Set[int] = {id(target_module)}
        for module in target_module.modules():
            target_layer_ids.add(id(module))

        # Collect layers outside target module
        outside_names = []
        prunable_types = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.MultiheadAttention,
        )

        for name, module in model.named_modules():
            if id(module) not in target_layer_ids:
                # Check if it's a prunable layer type
                if isinstance(module, prunable_types):
                    outside_names.append(name)

        return outside_names

    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """Get module by its name."""
        if name == "":
            return model
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _get_ignored_layers_from_model(self, model: nn.Module) -> List[nn.Module]:
        """
        Get all layers that should be ignored during pruning.
        This includes:
        1. All layers outside target module
        2. Specified ignored layers within target module
        """
        ignored = []

        # Add layers with .mlp.fc2 if specified
        if self.ignore_mlp_fc2:
            for n, m in model.named_modules():
                if hasattr(m, "mlp") and hasattr(m.mlp, "fc2"):
                    ignored.append(m.mlp.fc2)

        # Add layers outside target module
        for name in self._outside_layer_names:
            try:
                module = self._get_module_by_name(model, name)
                ignored.append(module)
            except AttributeError:
                # Layer may have been removed or renamed
                continue

        # Add specified ignored layers within target module
        for name in self._ignored_layer_names_in_target:
            try:
                module = self._get_module_by_name(model, name)
                ignored.append(module)
            except AttributeError:
                continue

        return ignored

    def _get_target_module_from_model(self, model: nn.Module) -> nn.Module:
        """Get target module reference from a model copy."""
        return self._get_module_by_name(model, self._target_module_name)

    def _create_fresh_model(self) -> nn.Module:
        """Create a fresh copy of the original model."""
        model_copy = deepcopy(self.model)
        model_copy.load_state_dict(deepcopy(self._original_model_state))
        for (n, p), (n_cpy, p_cpy) in zip(self.model.named_parameters(), model_copy.named_parameters()):
            assert n == n_cpy, "Parameter name mismatch in model copy"
            assert torch.equal(p, p_cpy), "Parameter value mismatch in model copy"
            if p.grad is not None:
                p_cpy.grad = torch.clone(p.grad).detach()
        return model_copy

    def _try_channel_sparsity(self, channel_sparsity: float) -> Tuple[float, int, int]:
        """
        Try pruning with given channel sparsity on a fresh model copy.

        Params:
            channel_sparsity: Channel pruning ratio to try (0.0 ~ 1.0).

        Returns:
            Tuple of (achieved_module_param_sparsity, module_final_params, model_final_params).
        """
        # Create fresh model copy
        model_copy = self._create_fresh_model()
        ignored_layers = self._get_ignored_layers_from_model(model_copy)
        target_module_copy = self._get_target_module_from_model(model_copy)

        # Create pruner
        pruner = tp.pruner.BasePruner(
            model=model_copy,
            example_inputs=self.example_inputs,
            importance=self.importance,
            pruning_ratio=channel_sparsity,
            ignored_layers=ignored_layers,
            global_pruning=self.global_pruning,
            round_to=self.round_to,
            num_heads=self._collect_num_heads(model_copy),
            isomorphic=self.isomorphic,
        )

        # Execute pruning
        pruner.step()

        # Calculate parameter statistics
        module_final_params = self._count_params(target_module_copy)
        model_final_params = self._count_params(model_copy)
        module_param_sparsity = 1.0 - (module_final_params / self._original_module_params)

        # Delete model copy to free memory
        del pruner
        del target_module_copy
        del model_copy

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return module_param_sparsity, module_final_params, model_final_params

    def search(self, initial, verbose: bool = True) -> Tuple[float, float, float]:
        """
        Binary search to find channel sparsity achieving target module param sparsity.

        Params:
            verbose: Whether to print search progress.

        Returns:
            Tuple of (best_channel_sparsity, achieved_module_param_sparsity).
        """
        low, high = self.channel_sparsity_range
        best_channel_sparsity = initial
        best_param_sparsity = 0.0
        best_error = float("inf")

        self._search_history.clear()

        if verbose:
            self._print_search_header()

        mid = initial
        for iteration in range(self.max_iterations):
            # Try this channel sparsity
            module_param_sparsity, module_params, model_params = self._try_channel_sparsity(mid)

            # Calculate error
            error = abs(module_param_sparsity - self.pruning_ratio)

            # Determine search direction
            if module_param_sparsity < self.pruning_ratio:
                direction = "↑ need more pruning"
                low = mid
                high = min(high, mid * 2.0)
            else:
                direction = "↓ too much pruning"
                high = mid
                low = max(low, mid / 2.0)

            if error < self.tolerance:
                direction = "✓ CONVERGED"

            # Record history
            record = {
                "iteration": iteration + 1,
                "low": low,
                "high": high,
                "channel_sparsity": mid,
                "module_param_sparsity": module_param_sparsity,
                "module_params": module_params,
                "model_params": model_params,
                "error": error,
                "direction": direction,
            }
            self._search_history.append(record)

            if verbose:
                self._print_search_iteration(record)

            # Update best result
            if error < best_error:
                best_channel_sparsity = mid
                best_param_sparsity = module_param_sparsity
                best_error = error

            # Early stopping
            if error < self.tolerance:
                break

            # Narrow search range
            if high - low < 1e-4:
                if module_param_sparsity < self.pruning_ratio:
                    high = mid * 2.0
                else:
                    low = mid / 2.0

            # Update mid for next iteration
            mid = (low + high) / 2.0

        if verbose:
            self._print_search_footer(best_channel_sparsity, best_param_sparsity)

        return best_channel_sparsity, best_param_sparsity, best_error

    def step(self, verbose: bool = True) -> ModulePruningResult:
        """
        Execute pruning to achieve target parameter sparsity for module.

        Params:
            verbose: Whether to print progress information.

        Returns:
            ModulePruningResult containing pruning statistics.
        """
        channel_sparsity, _, sparsity_error = self.search(verbose=verbose, initial=self.pruning_ratio)
        if sparsity_error > self.tolerance:
            warnings.warn(f"Sparsity error {sparsity_error:.4f} exceeds tolerance {self.tolerance:.4f}")

        if verbose:
            print(f"\nApplying pruning to target module '{self._target_module_name}'...")

        # Get ignored layers from original model
        ignored_layers = self._get_ignored_layers_from_model(self.model)

        # Create and execute pruner
        pruner = tp.pruner.BasePruner(
            model=self.model,
            example_inputs=self.example_inputs,
            importance=self.importance,
            pruning_ratio=channel_sparsity,
            ignored_layers=ignored_layers,
            global_pruning=self.global_pruning,
            round_to=self.round_to,
            num_heads=self._collect_num_heads(self.model),
            isomorphic=self.isomorphic,
        )
        pruner.step()

        # Calculate final statistics
        module_final_params = self._count_params(self.target_module)
        model_final_params = self._count_params(self.model)

        module_achieved_sparsity = 1.0 - (module_final_params / self._original_module_params)
        model_achieved_sparsity = 1.0 - (model_final_params / self._original_model_params)

        error = abs(module_achieved_sparsity - self.pruning_ratio)
        success = error < self.tolerance

        result = ModulePruningResult(
            success=success,
            module_original_params=self._original_module_params,
            module_final_params=module_final_params,
            module_target_sparsity=self.pruning_ratio,
            module_achieved_sparsity=module_achieved_sparsity,
            model_original_params=self._original_model_params,
            model_final_params=model_final_params,
            model_achieved_sparsity=model_achieved_sparsity,
            channel_ratio_used=channel_sparsity,
            search_iterations=len(self._search_history),
            message="Converged" if success else f"Best effort (error={error:.4f})",
            search_history=self._search_history.copy(),
        )

        if verbose:
            self._print_final_result(result)

        return result

    def estimate(self, target_sparsity: Optional[float] = None) -> float:
        """
        Estimate channel sparsity for target module parameter sparsity.

        Params:
            target_sparsity: Target parameter sparsity. Uses instance default if None.

        Returns:
            Estimated channel sparsity ratio.
        """
        original_target = self.pruning_ratio
        if target_sparsity is not None:
            self.pruning_ratio = target_sparsity

        channel_sparsity, _, sparsity_error = self.search(verbose=False)
        if sparsity_error > self.tolerance:
            warnings.warn(f"Sparsity error {sparsity_error:.4f} exceeds tolerance {self.tolerance:.4f}")

        self.pruning_ratio = original_target
        return channel_sparsity

    def get_search_history(self) -> List[Dict]:
        """Get the history of binary search iterations."""
        return self._search_history.copy()

    def reset_model(self) -> None:
        """Reset model to original state (undo pruning)."""
        self.model.load_state_dict(deepcopy(self._original_model_state))

    def get_target_module_info(self) -> Dict:
        """Get information about the target module."""
        return {
            "name": self._target_module_name,
            "original_params": self._original_module_params,
            "param_ratio_in_model": self._original_module_params / self._original_model_params,
            "num_ignored_layers_outside": len(self._outside_layer_names),
            "num_ignored_layers_inside": len(self._ignored_layer_names_in_target),
        }

    def _print_search_header(self) -> None:
        """Print search header."""
        print(f"\n{'='*90}")
        print("Binary Search for Target Module Parameter Sparsity")
        print(f"Target Module: '{self._target_module_name}'")
        print(
            f"Module Params: {self._original_module_params:,} / {self._original_model_params:,} "
            f"({self._original_module_params/self._original_model_params:.1%} of model)"
        )
        print(f"Target Sparsity: {self.pruning_ratio:.2%}")
        print(f"Channel Sparsity Range: [{self.channel_sparsity_range[0]:.2f}, {self.channel_sparsity_range[1]:.2f}]")
        print(f"{'='*90}")
        print(f"{'Iter':<5} {'Ch.Sparse':<12} {'Mod.Sparse':<14} " f"{'Error':<10} {'Mod.Params':<15} {'Direction'}")
        print(f"{'-'*90}")

    def _print_search_iteration(self, record: Dict) -> None:
        """Print single search iteration."""
        print(
            f"{record['iteration']:<5} "
            f"{record['channel_sparsity']:<12.4f} "
            f"{record['module_param_sparsity']:<14.4f} "
            f"{record['error']:<10.4f} "
            f"{record['module_params']:<15,} "
            f"{record['direction']}"
        )

    def _print_search_footer(self, best_channel: float, best_param: float) -> None:
        """Print search footer."""
        print(f"{'-'*90}")
        print(f"Best: channel_sparsity={best_channel:.4f} -> module_param_sparsity={best_param:.4f}")
        print(f"{'='*90}")

    def _print_final_result(self, result: ModulePruningResult) -> None:
        """Print final pruning result."""
        status = "✓ SUCCESS" if result.success else "⚠ BEST EFFORT"

        print(f"\n{'='*60}")
        print(f"Module Pruning Result: {status}")
        print(f"{'='*60}")
        print(f"\nTarget Module: '{self._target_module_name}'")
        print(f"{'-'*60}")
        print(f"{'Original Parameters:':<30} {result.module_original_params:>20,}")
        print(f"{'Final Parameters:':<30} {result.module_final_params:>20,}")
        print(f"{'Parameters Removed:':<30} {result.module_original_params - result.module_final_params:>20,}")
        print(f"{'Target Sparsity:':<30} {result.module_target_sparsity:>20.2%}")
        print(f"{'Achieved Sparsity:':<30} {result.module_achieved_sparsity:>20.2%}")

        print("\nOverall Model")
        print(f"{'-'*60}")
        print(f"{'Original Parameters:':<30} {result.model_original_params:>20,}")
        print(f"{'Final Parameters:':<30} {result.model_final_params:>20,}")
        print(f"{'Model Sparsity:':<30} {result.model_achieved_sparsity:>20.2%}")

        print("\nSearch Info")
        print(f"{'-'*60}")
        print(f"{'Channel Ratio Used:':<30} {result.channel_ratio_used:>20.4f}")
        print(f"{'Search Iterations:':<30} {result.search_iterations:>20}")
        print(f"{'='*60}\n")

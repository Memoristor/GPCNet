# coding=utf-8

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from prettytable import SINGLE_BORDER, PrettyTable

__all__ = ["LayerSparsityInfo", "SparsityAnalyzer", "MaskedLayerSparsityInfo", "MaskedSparsityAnalyzer"]


@dataclass
class LayerSparsityInfo:
    """Container for layer sparsity information."""

    name: str
    layer_type: str
    original_out_channels: int
    pruned_out_channels: int
    original_in_channels: int
    pruned_in_channels: int
    channel_sparsity: float
    original_params: int
    pruned_params: int
    param_sparsity: float


class SparsityAnalyzer:
    """
    Analyzer for comparing model sparsity before and after pruning.
    """

    def __init__(self):
        self._original_params: int = 0
        self._original_state: Dict[str, Dict] = {}
        self._model_name: str = ""

    def record_original(self, model: nn.Module, name: str = "model") -> None:
        """
        Record the original model structure before pruning.

        Params:
            model: Original model before pruning.
            name: Optional name for the model.
        """
        self._original_params = self._count_module_params(model)
        self._original_state.clear()
        self._model_name = name

        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self._original_state[module_name] = {
                    "type": "Conv2d",
                    "out_channels": module.out_channels,
                    "in_channels": module.in_channels,
                    "kernel_size": module.kernel_size,
                    "params": self._count_module_params(module),
                }
            elif isinstance(module, nn.Linear):
                self._original_state[module_name] = {
                    "type": "Linear",
                    "out_channels": module.out_features,
                    "in_channels": module.in_features,
                    "params": self._count_module_params(module),
                }
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                self._original_state[module_name] = {
                    "type": type(module).__name__,
                    "out_channels": module.num_features,
                    "in_channels": module.num_features,
                    "params": self._count_module_params(module),
                }
            elif isinstance(module, nn.LayerNorm):
                self._original_state[module_name] = {
                    "type": "LayerNorm",
                    "out_channels": module.normalized_shape[0] if module.normalized_shape else 0,
                    "in_channels": module.normalized_shape[0] if module.normalized_shape else 0,
                    "params": self._count_module_params(module),
                }

    @staticmethod
    def _count_module_params(module: nn.Module) -> int:
        """Count parameters in a single module."""
        return sum(p.numel() for p in module.parameters())

    def compute_sparsity(self, model: nn.Module) -> List[LayerSparsityInfo]:
        """
        Compute per-layer sparsity by comparing with original structure.

        Params:
            model: Pruned model to analyze.

        Returns:
            List of LayerSparsityInfo objects.

        """
        if not self._original_state:
            raise RuntimeError("Must call record_original() before compute_sparsity()")

        results = []

        for name, module in model.named_modules():
            if name not in self._original_state:
                continue

            orig = self._original_state[name]
            pruned_params = self._count_module_params(module)

            if isinstance(module, nn.Conv2d):
                pruned_out = module.out_channels
                pruned_in = module.in_channels
            elif isinstance(module, nn.Linear):
                pruned_out = module.out_features
                pruned_in = module.in_features
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                pruned_out = module.num_features
                pruned_in = module.num_features
            elif isinstance(module, nn.LayerNorm):
                pruned_out = module.normalized_shape[0] if module.normalized_shape else 0
                pruned_in = pruned_out
            else:
                continue

            orig_out = orig["out_channels"]
            orig_in = orig["in_channels"]
            orig_params = orig["params"]

            channel_sparsity = 1.0 - (pruned_out / orig_out) if orig_out > 0 else 0.0
            param_sparsity = 1.0 - (pruned_params / orig_params) if orig_params > 0 else 0.0

            results.append(
                LayerSparsityInfo(
                    name=name,
                    layer_type=orig["type"],
                    original_out_channels=orig_out,
                    pruned_out_channels=pruned_out,
                    original_in_channels=orig_in,
                    pruned_in_channels=pruned_in,
                    channel_sparsity=channel_sparsity,
                    original_params=orig_params,
                    pruned_params=pruned_params,
                    param_sparsity=param_sparsity,
                )
            )

        return results

    def print_report(
        self,
        results: List[LayerSparsityInfo],
        show_all: bool = True,
        min_sparsity: float = 0.0,
        max_name_length: Optional[int] = 80,
        title: Optional[str] = None,
    ) -> None:
        """
        Print a formatted sparsity report using PrettyTable.

        Params:
            results: List of LayerSparsityInfo from compute_sparsity().
            show_all: If True, show all layers including unpruned ones.
            min_sparsity: Minimum sparsity threshold to display.
            max_name_length: Maximum length for layer names (truncate if longer).
                            Set to None to show full names.
            title: Custom title for the report.
        """
        # Create main table
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)
        table.field_names = ["Layer Name", "Type", "Out Ch", "In Ch", "Ch Sparse", "Param Sparse"]

        # Align columns
        table.align["Layer Name"] = "l"
        table.align["Type"] = "l"
        table.align["Out Ch"] = "c"
        table.align["In Ch"] = "c"
        table.align["Ch Sparse"] = "r"
        table.align["Param Sparse"] = "r"

        # Add rows
        for info in results:
            if not show_all and info.channel_sparsity < min_sparsity:
                continue

            # Truncate name if needed
            name = self._truncate_name(info.name, max_name_length)
            out_ch = f"{info.pruned_out_channels}/{info.original_out_channels}"
            in_ch = f"{info.pruned_in_channels}/{info.original_in_channels}"
            ch_sparse = f"{info.channel_sparsity * 100:.2f}%"
            param_sparse = f"{info.param_sparsity * 100:.2f}%"

            table.add_row([name, info.layer_type, out_ch, in_ch, ch_sparse, param_sparse])

        # Print title
        report_title = title or f"Sparsity Report: {self._model_name}"
        print(f"\n{report_title}")
        print(table)

    def print_summary(self, model: nn.Module, hierarchy_depth: int = 2, max_name_length: Optional[int] = 50) -> None:
        """
        Print summary including hierarchical sparsity analysis.

        Args:
            model: Pruned model
            hierarchy_depth: Maximum depth for hierarchical analysis.
                            Set to 0 to disable hierarchical view.
            max_name_length: Maximum length for module names in hierarchy table.
        """
        # Create summary table
        summary_table = PrettyTable()
        summary_table.set_style(SINGLE_BORDER)
        summary_table.field_names = ["Metric", "Value"]
        summary_table.align["Metric"] = "l"
        summary_table.align["Value"] = "r"

        total_pruned_params = self._count_module_params(model)
        overall_sparsity = 1.0 - (total_pruned_params / self._original_params) if self._original_params > 0 else 0.0
        summary_table.add_row(["Overall Param Sparsity", f"{overall_sparsity * 100:.2f}%"])
        summary_table.add_row(["Total Original Params", f"{self._original_params:,}"])
        summary_table.add_row(["Total Pruned Params", f"{total_pruned_params:,}"])
        summary_table.add_row(["Parameters Removed", f"{self._original_params - total_pruned_params:,}"])

        print("\nSummary:")
        print(summary_table)

        # Create hierarchical sparsity table
        if hierarchy_depth > 0:
            hierarchy_table = PrettyTable()
            hierarchy_table.set_style(SINGLE_BORDER)
            hierarchy_table.field_names = [
                "Module",
                "Depth",
                "Original Params",
                "Pruned Params",
                "Sparsity",
            ]
            hierarchy_table.align["Module"] = "l"
            hierarchy_table.align["Depth"] = "c"
            hierarchy_table.align["Original Params"] = "r"
            hierarchy_table.align["Pruned Params"] = "r"
            hierarchy_table.align["Sparsity"] = "r"

            prefixes = self._get_hierarchical_prefixes(model, hierarchy_depth)

            for prefix in prefixes:
                orig_params, pruned_params, sparsity = self._compute_prefix_sparsity(model, prefix)
                if orig_params > 0:
                    if prefix == "":
                        module_name = self._model_name or "model"
                        depth = 0
                    else:
                        module_name = self._truncate_name(prefix, max_name_length)
                        depth = prefix.count(".") + 1

                    hierarchy_table.add_row(
                        [
                            module_name,
                            depth,
                            f"{orig_params:,}",
                            f"{pruned_params:,}",
                            f"{sparsity * 100:.2f}%",
                        ]
                    )

            print("\nHierarchical Sparsity:")
            print(hierarchy_table)

        print()

    def _get_hierarchical_prefixes(self, model: nn.Module, max_depth: int) -> List[str]:
        """Get unique module prefixes up to max_depth."""
        prefixes = set()
        prefixes.add("")  # Include the root module

        for name, _ in model.named_modules():
            if not name:
                continue
            parts = name.split(".")
            for i in range(1, min(len(parts), max_depth) + 1):
                prefix = ".".join(parts[:i])
                prefixes.add(prefix)

        return sorted(prefixes, key=lambda x: (x.count("."), x))

    def _compute_prefix_sparsity(self, model: nn.Module, prefix: str) -> tuple[int, int, float]:
        """Compute sparsity for all recorded layers under the given prefix."""
        original_params = 0
        pruned_params = 0

        for name, state in self._original_state.items():
            if prefix == "" or name == prefix or name.startswith(prefix + "."):
                original_params += state["params"]

        for name, module in model.named_modules():
            if name not in self._original_state:
                continue
            if prefix == "" or name == prefix or name.startswith(prefix + "."):
                pruned_params += self._count_module_params(module)

        sparsity = 1.0 - (pruned_params / original_params) if original_params > 0 else 0.0
        return original_params, pruned_params, sparsity

    def _truncate_name(self, name: str, max_length: Optional[int]) -> str:
        """Truncate layer name if too long."""
        if max_length is None or len(name) <= max_length:
            return name
        return "..." + name[-(max_length - 3) :]

    def get_summary(self, results: List[LayerSparsityInfo]) -> Dict:
        """Get a summary dictionary of sparsity statistics."""
        total_orig_params = sum(r.original_params for r in results)
        total_pruned_params = sum(r.pruned_params for r in results)

        channel_sparsities = [r.channel_sparsity for r in results if r.channel_sparsity > 0]
        param_sparsities = [r.param_sparsity for r in results if r.param_sparsity > 0]

        return {
            "model_name": self._model_name,
            "num_layers": len(results),
            "total_original_params": total_orig_params,
            "total_pruned_params": total_pruned_params,
            "overall_param_sparsity": 1.0 - (total_pruned_params / total_orig_params) if total_orig_params > 0 else 0.0,
            "num_pruned_layers": len(channel_sparsities),
            "avg_channel_sparsity": sum(channel_sparsities) / len(channel_sparsities) if channel_sparsities else 0.0,
            "max_channel_sparsity": max(channel_sparsities) if channel_sparsities else 0.0,
            "min_channel_sparsity": min(channel_sparsities) if channel_sparsities else 0.0,
            "avg_param_sparsity": sum(param_sparsities) / len(param_sparsities) if param_sparsities else 0.0,
        }

    def to_dataframe(self, results: List[LayerSparsityInfo]):
        """Convert results to pandas DataFrame."""
        data = [
            {
                "layer_name": r.name,
                "type": r.layer_type,
                "orig_out_ch": r.original_out_channels,
                "pruned_out_ch": r.pruned_out_channels,
                "orig_in_ch": r.original_in_channels,
                "pruned_in_ch": r.pruned_in_channels,
                "ch_sparsity": r.channel_sparsity,
                "orig_params": r.original_params,
                "pruned_params": r.pruned_params,
                "param_sparsity": r.param_sparsity,
            }
            for r in results
        ]

        return pd.DataFrame(data)


@dataclass
class MaskedLayerSparsityInfo:
    """Container for layer sparsity information."""

    name: str
    layer_type: str
    weight_shape: tuple
    nonzero_params: int
    total_params: int
    param_sparsity: float
    has_weight_mask: bool


class MaskedSparsityAnalyzer:
    """
    Analyzer for measuring parameter-level sparsity in neural networks.

    This analyzer counts zero-valued parameters to calculate sparsity,
    which is suitable for networks pruned with unstructured methods
    (like torch.nn.utils.prune.l1_unstructured or global_unstructured).
    """

    def __init__(self):
        self._original_params: int = 0
        self._original_state: Dict[str, Dict] = {}
        self._model_name: str = ""

    def record_original(self, model: nn.Module, name: str = "model") -> None:
        """
        Record the original model parameters before pruning.

        Params:
            model: Original model before pruning.
            name: Optional name for the model.
        """
        self._original_params = self._count_module_params(model)
        self._original_state.clear()
        self._model_name = name

        for module_name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight
                self._original_state[module_name] = {
                    "type": module.__class__.__name__,
                    "weight_shape": weight.shape,
                    "params": weight.numel(),
                    "nonzero_params": torch.count_nonzero(weight).item(),
                }

            # Record bias parameters separately if needed
            if hasattr(module, "bias") and module.bias is not None:
                bias_name = f"{module_name}.bias"
                bias = module.bias
                self._original_state[bias_name] = {
                    "type": f"{module.__class__.__name__} (bias)",
                    "weight_shape": bias.shape,
                    "params": bias.numel(),
                    "nonzero_params": torch.count_nonzero(bias).item(),
                }

    @staticmethod
    def _count_module_params(module: nn.Module) -> int:
        """Count parameters in a single module."""
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def _count_nonzero_params(tensor: torch.Tensor) -> int:
        """Count non-zero parameters in a tensor."""
        return torch.count_nonzero(tensor).item()

    def compute_sparsity(self, model: nn.Module) -> List[MaskedLayerSparsityInfo]:
        """
        Compute per-layer sparsity by counting zero-valued parameters.

        Params:
            model: Pruned model to analyze.

        Returns:
            List of MaskedLayerSparsityInfo objects.
        """
        if not self._original_state:
            raise RuntimeError("Must call record_original() before compute_sparsity()")

        results = []

        # Check for weights in modules
        for module_name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight

                # Check if this module has been pruned
                has_mask = hasattr(module, "weight_mask")

                nonzero_count = self._count_nonzero_params(weight)
                total_params = weight.numel()
                sparsity = 1.0 - (nonzero_count / total_params) if total_params > 0 else 0.0

                results.append(
                    MaskedLayerSparsityInfo(
                        name=module_name,
                        layer_type=module.__class__.__name__,
                        weight_shape=tuple(weight.shape),
                        nonzero_params=nonzero_count,
                        total_params=total_params,
                        param_sparsity=sparsity,
                        has_weight_mask=has_mask,
                    )
                )

            # Also check bias parameters
            if hasattr(module, "bias") and module.bias is not None:
                bias_name = f"{module_name}.bias"
                bias = module.bias

                has_mask = hasattr(module, "bias_mask")

                nonzero_count = self._count_nonzero_params(bias)
                total_params = bias.numel()
                sparsity = 1.0 - (nonzero_count / total_params) if total_params > 0 else 0.0

                results.append(
                    MaskedLayerSparsityInfo(
                        name=bias_name,
                        layer_type=f"{module.__class__.__name__} (bias)",
                        weight_shape=tuple(bias.shape),
                        nonzero_params=nonzero_count,
                        total_params=total_params,
                        param_sparsity=sparsity,
                        has_weight_mask=has_mask,
                    )
                )

        return results

    def print_report(
        self,
        results: List[MaskedLayerSparsityInfo],
        show_all: bool = True,
        min_sparsity: float = 0.0,
        max_name_length: Optional[int] = 80,
        title: Optional[str] = None,
    ) -> None:
        """
        Print a formatted sparsity report using PrettyTable.

        Params:
            results: List of MaskedLayerSparsityInfo from compute_sparsity().
            show_all: If True, show all layers including unpruned ones.
            min_sparsity: Minimum sparsity threshold to display.
            max_name_length: Maximum length for layer names (truncate if longer).
                            Set to None to show full names.
            title: Custom title for the report.
        """
        # Create main table
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)
        table.field_names = ["Layer Name", "Type", "Shape", "NonZero/Total", "Sparsity", "Pruned"]

        # Align columns
        table.align["Layer Name"] = "l"
        table.align["Type"] = "l"
        table.align["Shape"] = "l"
        table.align["NonZero/Total"] = "r"
        table.align["Sparsity"] = "r"
        table.align["Pruned"] = "c"

        # Add rows
        for info in results:
            if not show_all and info.param_sparsity < min_sparsity:
                continue

            # Truncate name if needed
            name = self._truncate_name(info.name, max_name_length)
            shape_str = str(info.weight_shape)
            nonzero_total = f"{info.nonzero_params:,}/{info.total_params:,}"
            param_sparse = f"{info.param_sparsity * 100:.2f}%"
            pruned = "✓" if info.has_weight_mask else ""

            table.add_row([name, info.layer_type, shape_str, nonzero_total, param_sparse, pruned])

        # Print title
        report_title = title or f"Sparsity Report: {self._model_name}"
        print(f"\n{report_title}")
        print(table)

    def print_summary(self, model: nn.Module, hierarchy_depth: int = 2, max_name_length: Optional[int] = 50) -> None:
        """
        Print summary including hierarchical sparsity analysis.

        Args:
            model: Pruned model
            hierarchy_depth: Maximum depth for hierarchical analysis.
                            Set to 0 to disable hierarchical view.
            max_name_length: Maximum length for module names in hierarchy table.
        """
        # Get total non-zero parameters
        total_nonzero_params = 0
        for name, param in model.named_parameters():
            total_nonzero_params += torch.count_nonzero(param).item()

        total_pruned_params = self._count_module_params(model)

        # Create summary table
        summary_table = PrettyTable()
        summary_table.set_style(SINGLE_BORDER)
        summary_table.field_names = ["Metric", "Value"]
        summary_table.align["Metric"] = "l"
        summary_table.align["Value"] = "r"

        overall_sparsity = 1.0 - (total_nonzero_params / total_pruned_params) if total_pruned_params > 0 else 0.0
        summary_table.add_row(["Overall Param Sparsity", f"{overall_sparsity * 100:.2f}%"])
        summary_table.add_row(["Total Parameters", f"{total_pruned_params:,}"])
        summary_table.add_row(["Non-Zero Parameters", f"{total_nonzero_params:,}"])
        summary_table.add_row(["Zero Parameters", f"{total_pruned_params - total_nonzero_params:,}"])

        print("\nSummary:")
        print(summary_table)

        # Create hierarchical sparsity table
        if hierarchy_depth > 0:
            hierarchy_table = PrettyTable()
            hierarchy_table.set_style(SINGLE_BORDER)
            hierarchy_table.field_names = [
                "Module",
                "Depth",
                "Total Params",
                "Non-Zero Params",
                "Sparsity",
            ]
            hierarchy_table.align["Module"] = "l"
            hierarchy_table.align["Depth"] = "c"
            hierarchy_table.align["Total Params"] = "r"
            hierarchy_table.align["Non-Zero Params"] = "r"
            hierarchy_table.align["Sparsity"] = "r"

            prefixes = self._get_hierarchical_prefixes(model, hierarchy_depth)

            for prefix in prefixes:
                total_params, nonzero_params, sparsity = self._compute_prefix_sparsity(model, prefix)
                if total_params > 0:
                    if prefix == "":
                        module_name = self._model_name or "model"
                        depth = 0
                    else:
                        module_name = self._truncate_name(prefix, max_name_length)
                        depth = prefix.count(".") + 1

                    hierarchy_table.add_row(
                        [
                            module_name,
                            depth,
                            f"{total_params:,}",
                            f"{nonzero_params:,}",
                            f"{sparsity * 100:.2f}%",
                        ]
                    )

            print("\nHierarchical Sparsity:")
            print(hierarchy_table)

        print()

    def _get_hierarchical_prefixes(self, model: nn.Module, max_depth: int) -> List[str]:
        """Get unique module prefixes up to max_depth."""
        prefixes = set()
        prefixes.add("")  # Include the root module

        for name, _ in model.named_modules():
            if not name:
                continue
            parts = name.split(".")
            for i in range(1, min(len(parts), max_depth) + 1):
                prefix = ".".join(parts[:i])
                prefixes.add(prefix)

        return sorted(prefixes, key=lambda x: (x.count("."), x))

    def _compute_prefix_sparsity(self, model: nn.Module, prefix: str) -> tuple[int, int, float]:
        """Compute sparsity for all parameters under the given prefix."""
        total_params = 0
        nonzero_params = 0

        # Count all parameters under this prefix
        for name, param in model.named_parameters():
            if prefix == "" or name.startswith(prefix):
                total_params += param.numel()
                nonzero_params += torch.count_nonzero(param).item()

        sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
        return total_params, nonzero_params, sparsity

    def _truncate_name(self, name: str, max_length: Optional[int]) -> str:
        """Truncate layer name if too long."""
        if max_length is None or len(name) <= max_length:
            return name
        return "..." + name[-(max_length - 3) :]

    def get_summary(self, results: List[MaskedLayerSparsityInfo]) -> Dict:
        """Get a summary dictionary of sparsity statistics."""
        total_params = sum(r.total_params for r in results)
        total_nonzero_params = sum(r.nonzero_params for r in results)

        # Get layers with sparsity > 0
        sparse_layers = [r for r in results if r.param_sparsity > 0]
        param_sparsities = [r.param_sparsity for r in sparse_layers]

        return {
            "model_name": self._model_name,
            "num_layers": len(results),
            "total_params": total_params,
            "nonzero_params": total_nonzero_params,
            "overall_param_sparsity": 1.0 - (total_nonzero_params / total_params) if total_params > 0 else 0.0,
            "num_sparse_layers": len(sparse_layers),
            "num_pruned_layers": sum(1 for r in results if r.has_weight_mask),
            "avg_param_sparsity": sum(param_sparsities) / len(param_sparsities) if param_sparsities else 0.0,
            "max_param_sparsity": max(param_sparsities) if param_sparsities else 0.0,
            "min_param_sparsity": min(param_sparsities) if param_sparsities else 0.0,
        }

    def to_dataframe(self, results: List[MaskedLayerSparsityInfo]):
        """Convert results to pandas DataFrame."""
        data = [
            {
                "layer_name": r.name,
                "type": r.layer_type,
                "weight_shape": str(r.weight_shape),
                "nonzero_params": r.nonzero_params,
                "total_params": r.total_params,
                "param_sparsity": r.param_sparsity,
                "has_mask": r.has_weight_mask,
            }
            for r in results
        ]

        return pd.DataFrame(data)

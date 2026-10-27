import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from svc import SpatialVariableConvolution


def test_param_variation():
    """Test SVC correctness under different dilation and stride settings without dilation_map"""
    print("\n" + "=" * 50)
    print("PARAMETER VARIATION TEST (WITHOUT DILATION MAP)")
    print("=" * 50)

    # Set parameters
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2

    # Parameters to test
    kernel_sizes = [3, 5]
    dilations = [1, 2, 3]
    strides = [1, 2]

    # Initialize results dictionary
    results = {}

    # Set random seed for reproducibility
    torch.manual_seed(42)

    for kernel_size in kernel_sizes:
        for dilation in dilations:
            for stride in strides:
                k2 = kernel_size * kernel_size
                channels_per_group = channels // groups
                padding = kernel_size // 2 * dilation  # Adjust padding for proper "same" padding with dilation

                # Calculate output dimensions
                height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

                print(f"\nTesting kernel={kernel_size}, dilation={dilation}, stride={stride}")
                print(f"Output shape: {height_out}x{width_out}")

                try:
                    # Create input tensors
                    x1 = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)

                    # Create weight tensor of appropriate size for the output dimensions
                    weight1 = torch.randn(
                        batch_size, groups * k2, height_out, width_out, device="cuda", requires_grad=True
                    )

                    # Clone tensors for SVC
                    x2 = x1.clone().detach().requires_grad_(True)
                    weight2 = weight1.clone().detach().requires_grad_(True)

                    param_key = (kernel_size, dilation, stride)

                    # PyTorch unfold + einsum reference
                    x_unfold = F.unfold(x1, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
                    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height_out, width_out)
                    weight_ref = weight1.view(batch_size, groups, k2, height_out, width_out)
                    out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
                    out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height_out, width_out)

                    # SVC implementation
                    svc = SpatialVariableConvolution(
                        kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups
                    ).cuda()
                    out2 = svc(x2, weight2)  # No dilation_map

                    # Compare shape and values
                    if out1.shape != out2.shape:
                        print(f"⚠️ Shape mismatch: unfold={out1.shape}, SVC={out2.shape}")
                        continue

                    # Compute forward differences
                    output_diff = (out1 - out2).abs()
                    max_output_diff = output_diff.max().item()
                    mean_output_diff = output_diff.mean().item()

                    # Compute backward pass - unfold reference
                    loss1 = out1.sum()
                    loss1.backward()
                    grad_x_ref = x1.grad.clone()
                    grad_weight_ref = weight1.grad.clone()

                    # Reset gradients
                    x1.grad.zero_()
                    weight1.grad.zero_()
                    x2.grad = None
                    weight2.grad = None

                    # Compute backward pass - SVC
                    loss2 = out2.sum()
                    loss2.backward()
                    grad_x_svc = x2.grad.clone()
                    grad_weight_svc = weight2.grad.clone()

                    # Compute gradient differences
                    grad_x_diff = (grad_x_ref - grad_x_svc).abs()
                    max_grad_x_diff = grad_x_diff.max().item()
                    mean_grad_x_diff = grad_x_diff.mean().item()

                    grad_weight_diff = (grad_weight_ref - grad_weight_svc).abs()
                    max_grad_weight_diff = grad_weight_diff.max().item()
                    mean_grad_weight_diff = grad_weight_diff.mean().item()

                    # Report results
                    print(f"Output diff - Max: {max_output_diff:.6e}, Mean: {mean_output_diff:.6e}")
                    print(f"Input grad diff - Max: {max_grad_x_diff:.6e}, Mean: {mean_grad_x_diff:.6e}")
                    print(f"Weight grad diff - Max: {max_grad_weight_diff:.6e}, Mean: {mean_grad_weight_diff:.6e}")

                    # Store results
                    results[param_key] = {
                        "max_output_diff": max_output_diff,
                        "mean_output_diff": mean_output_diff,
                        "max_grad_x_diff": max_grad_x_diff,
                        "mean_grad_x_diff": mean_grad_x_diff,
                        "max_grad_weight_diff": max_grad_weight_diff,
                        "mean_grad_weight_diff": mean_grad_weight_diff,
                    }

                    # Critical test: Check if differences are within tolerance
                    tolerance = 1e-4
                    if max_output_diff > tolerance or max_grad_x_diff > tolerance or max_grad_weight_diff > tolerance:
                        print(f"❌ Test failed: Differences exceed tolerance ({tolerance})")
                    else:
                        print(f"✅ Test passed: All differences within tolerance ({tolerance})")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

                finally:
                    # Clean up memory
                    torch.cuda.empty_cache()

    # Visualize results
    visualize_param_results(results, kernel_sizes, dilations, strides)
    return results


def visualize_param_results(results, kernel_sizes, dilations, strides):
    """Visualize test results across different parameter combinations"""
    if not results:
        print("No results to visualize")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    # Prepare data for plotting
    param_labels = [f"k{k}_d{d}_s{s}" for k in kernel_sizes for d in dilations for s in strides if (k, d, s) in results]

    output_diffs = [
        results[(k, d, s)]["max_output_diff"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]
    input_grad_diffs = [
        results[(k, d, s)]["max_grad_x_diff"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]
    weight_grad_diffs = [
        results[(k, d, s)]["max_grad_weight_diff"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]

    # Plot output differences
    axes[0].bar(param_labels, output_diffs)
    axes[0].set_ylabel("Maximum Absolute Difference")
    axes[0].set_title("Output Differences by Parameter Combination")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].grid(True, alpha=0.3, which="both")

    # Plot input gradient differences
    axes[1].bar(param_labels, input_grad_diffs)
    axes[1].set_ylabel("Maximum Absolute Difference")
    axes[1].set_title("Input Gradient Differences by Parameter Combination")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid(True, alpha=0.3, which="both")

    # Plot weight gradient differences
    axes[2].bar(param_labels, weight_grad_diffs)
    axes[2].set_ylabel("Maximum Absolute Difference")
    axes[2].set_title("Weight Gradient Differences by Parameter Combination")
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=90)
    axes[2].grid(True, alpha=0.3, which="both")

    # Add a horizontal reference line for typical tolerance
    for ax in axes:
        ax.axhline(1e-4, color="red", linestyle="--", alpha=0.7, label="Tolerance (1e-4)")
        ax.legend()

    plt.tight_layout()
    # plt.savefig("parameter_variation_results.png")
    # print("Results visualization saved as 'parameter_variation_results.png'")
    plt.show()


def test_performance_by_parameters():
    """Compare performance across different parameter settings"""
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON BY PARAMETERS")
    print("=" * 50)

    # Set parameters
    batch_size = 4
    channels = 32
    height = width = 64
    groups = 4

    # Parameters to test
    kernel_sizes = [3, 5]
    dilations = [1, 2]
    strides = [1, 2]

    # Results dictionary
    perf_results = {}

    # Set random seed
    torch.manual_seed(42)

    for kernel_size in kernel_sizes:
        for dilation in dilations:
            for stride in strides:
                k2 = kernel_size * kernel_size
                channels_per_group = channels // groups
                padding = kernel_size // 2 * dilation

                # Calculate output dimensions
                height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

                param_key = (kernel_size, dilation, stride)
                print(f"\nBenchmarking kernel={kernel_size}, dilation={dilation}, stride={stride}")

                try:
                    # Create input and weights
                    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
                    weight = torch.randn(
                        batch_size, groups * k2, height_out, width_out, device="cuda", requires_grad=True
                    )

                    # Create SVC module
                    svc = SpatialVariableConvolution(
                        kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups
                    ).cuda()

                    # Warmup
                    for _ in range(5):
                        out = svc(x, weight)
                        loss = out.sum()
                        loss.backward(retain_graph=True)
                        x.grad.zero_()
                        weight.grad.zero_()

                    # Setup timing
                    iterations = 20 if kernel_size * dilation <= 5 else 10  # Fewer iterations for larger configs
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    # Forward timing
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(iterations):
                        out = svc(x, weight)
                    end.record()
                    torch.cuda.synchronize()
                    forward_time = start.elapsed_time(end) / iterations

                    # Backward timing
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(iterations):
                        loss = out.sum()
                        loss.backward(retain_graph=True)
                        x.grad.zero_()
                        weight.grad.zero_()
                    end.record()
                    torch.cuda.synchronize()
                    backward_time = start.elapsed_time(end) / iterations

                    # Store results
                    perf_results[param_key] = {
                        "forward_time": forward_time,
                        "backward_time": backward_time,
                        "total_time": forward_time + backward_time,
                        "size": (height_out * width_out * channels * batch_size) / (1024 * 1024),  # Size in MB
                    }

                    print(
                        f"Forward: {forward_time:.3f} ms, Backward: {backward_time:.3f} ms, Total: {forward_time + backward_time:.3f} ms"
                    )

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

                finally:
                    # Clean up
                    torch.cuda.empty_cache()

    # Visualize performance results
    visualize_performance_results(perf_results, kernel_sizes, dilations, strides)
    return perf_results


def visualize_performance_results(results, kernel_sizes, dilations, strides):
    """Visualize performance across parameter combinations"""
    if not results:
        print("No performance results to visualize")
        return

    # Prepare data
    param_labels = [f"k{k}_d{d}_s{s}" for k in kernel_sizes for d in dilations for s in strides if (k, d, s) in results]
    forward_times = [
        results[(k, d, s)]["forward_time"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]
    backward_times = [
        results[(k, d, s)]["backward_time"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]
    total_times = [
        results[(k, d, s)]["total_time"]
        for k in kernel_sizes
        for d in dilations
        for s in strides
        if (k, d, s) in results
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))

    x = np.arange(len(param_labels))
    width = 0.25

    ax.bar(x - width, forward_times, width, label="Forward Time (ms)")
    ax.bar(x, backward_times, width, label="Backward Time (ms)")
    ax.bar(x + width, total_times, width, label="Total Time (ms)")

    ax.set_ylabel("Time (ms)")
    ax.set_title("SVC Performance by Parameter Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, rotation=90)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    # plt.savefig("performance_by_parameters.png")
    # print("Performance visualization saved as 'performance_by_parameters.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SVC PARAMETER VARIATION TESTS (WITHOUT DILATION MAP)")
    print("=" * 50)

    # Run correctness tests across different parameter combinations
    param_results = test_param_variation()

    # Run performance benchmarks across different parameter combinations
    perf_results = test_performance_by_parameters()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50)

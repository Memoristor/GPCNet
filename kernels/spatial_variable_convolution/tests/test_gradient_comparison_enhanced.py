import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from svc import SpatialVariableConvolution


def test_gradient_comparison():
    """比较 F.unfold + einsum 和 SVC 的梯度计算"""
    print("\n" + "=" * 50)
    print("GRADIENT COMPARISON TEST")
    print("=" * 50)

    # 设置参数
    batch_size = 2
    channels = 12
    height = 32
    width = 32
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)

    # 创建两组相同的输入和权重
    x1 = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight1 = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    x2 = x1.clone().detach().requires_grad_(True)
    weight2 = weight1.clone().detach().requires_grad_(True)

    # 确保输入完全相同
    assert torch.allclose(x1, x2), "Input tensors should be identical"
    assert torch.allclose(weight1, weight2), "Weight tensors should be identical"

    print("Computing gradients using F.unfold + einsum...")
    # 使用 F.unfold + einsum 方法
    x_unfold = F.unfold(x1, kernel_size=kernel_size, padding=kernel_size // 2)
    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
    weight_ref = weight1.view(batch_size, groups, k2, height, width)
    out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
    out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

    # 计算损失和反向传播
    loss1 = out1.sum()
    loss1.backward()

    print("Computing gradients using SVC...")
    # 使用 SVC 方法 (不使用 dilation_map)
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    out2 = svc(x2, weight2)  # 没有 dilation_map

    # 计算损失和反向传播
    loss2 = out2.sum()
    loss2.backward()

    # 首先比较输出
    output_diff = (out1 - out2).abs()
    max_out_diff = output_diff.max().item()
    mean_out_diff = output_diff.mean().item()
    print("\nOutput Comparison:")
    print(f"  Max absolute difference: {max_out_diff:.6e}")
    print(f"  Mean absolute difference: {mean_out_diff:.6e}")

    # 比较输入梯度
    input_diff = (x1.grad - x2.grad).abs()
    max_input_diff = input_diff.max().item()
    mean_input_diff = input_diff.mean().item()
    rel_input_diff = (input_diff / (x1.grad.abs() + 1e-7)).mean().item()

    print("\nInput Gradient Comparison:")
    print(f"  Max absolute difference: {max_input_diff:.6e}")
    print(f"  Mean absolute difference: {mean_input_diff:.6e}")
    print(f"  Mean relative difference: {rel_input_diff:.6e}")

    # 比较权重梯度
    weight_diff = (weight1.grad - weight2.grad).abs()
    max_weight_diff = weight_diff.max().item()
    mean_weight_diff = weight_diff.mean().item()
    rel_weight_diff = (weight_diff / (weight1.grad.abs() + 1e-7)).mean().item()

    print("\nWeight Gradient Comparison:")
    print(f"  Max absolute difference: {max_weight_diff:.6e}")
    print(f"  Mean absolute difference: {mean_weight_diff:.6e}")
    print(f"  Mean relative difference: {rel_weight_diff:.6e}")

    # 检查梯度差异是否在可接受范围内
    tolerance = 1e-4
    assert max_input_diff < tolerance, f"Input gradient max difference too large: {max_input_diff}"
    assert max_weight_diff < tolerance, f"Weight gradient max difference too large: {max_weight_diff}"

    print("\nTest passed! Gradients from SVC match F.unfold + einsum method.")

    # 绘制梯度差异分布图
    visualize_gradient_differences(x1.grad, x2.grad, weight1.grad, weight2.grad)

    return x1, x2, weight1, weight2


def visualize_gradient_differences(grad1_input, grad2_input, grad1_weight, grad2_weight):
    """可视化梯度差异分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Input梯度差异
    input_diff = (grad1_input - grad2_input).abs().cpu().numpy().flatten()
    ax1.hist(input_diff, bins=50, alpha=0.7)
    ax1.set_title("Input Gradient Absolute Differences")
    ax1.set_xlabel("Absolute Difference")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    # Weight梯度差异
    weight_diff = (grad1_weight - grad2_weight).abs().cpu().numpy().flatten()
    ax2.hist(weight_diff, bins=50, alpha=0.7)
    ax2.set_title("Weight Gradient Absolute Differences")
    ax2.set_xlabel("Absolute Difference")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("gradient_differences.png")
    # print("Gradient visualization saved as 'gradient_differences.png'")
    # plt.close()
    plt.show()


def test_param_variation():
    """Test SVC correctness under different dilation and stride settings"""
    print("\n" + "=" * 50)
    print("PARAMETER VARIATION TEST")
    print("=" * 50)

    # Set parameters
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2

    # Parameters to test
    dilations = [1, 2, 3, 5]
    strides = [1, 2]
    kernel_sizes = [3, 5]

    # Initialize results dictionary
    results = {}

    # Set random seed for reproducibility
    torch.manual_seed(42)

    for kernel_size in kernel_sizes:
        for dilation in dilations:
            for stride in strides:
                k2 = kernel_size * kernel_size
                channels_per_group = channels // groups
                padding = kernel_size // 2

                # Calculate output dimensions
                height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

                # Create input tensors
                x1 = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)

                # Create weight tensor of appropriate size
                weight1 = torch.randn(batch_size, groups * k2, height_out, width_out, device="cuda", requires_grad=True)

                # Clone tensors for SVC
                x2 = x1.clone().detach().requires_grad_(True)
                weight2 = weight1.clone().detach().requires_grad_(True)

                param_key = (kernel_size, dilation, stride)
                print(f"\nTesting kernel={kernel_size}, dilation={dilation}, stride={stride}")
                print(f"Output shape: {height_out}x{width_out}")

                try:
                    # PyTorch unfold + einsum reference
                    x_unfold = F.unfold(x1, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
                    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height_out, width_out)
                    weight_ref = weight1.view(batch_size, groups, k2, height_out, width_out)
                    out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
                    out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height_out, width_out)

                    # SVC implementation
                    svc = SpatialVariableConvolution(
                        kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, stride=stride
                    ).cuda()
                    out2 = svc(x2, weight2)

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


def measure_memory_usage():
    """比较两种方法的内存使用"""
    print("\n" + "=" * 50)
    print("MEMORY USAGE COMPARISON")
    print("=" * 50)

    # 设置较大的参数以便明显观察内存差异
    batch_size = 8
    channels = 64
    height = width = 128
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 创建输入和权重
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda")

    # 预热GPU
    for _ in range(3):
        torch.cuda.empty_cache()
        # 使用 F.unfold + einsum
        x_unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

        # 清理内存
        del x_unfold, weight_ref, out1
        torch.cuda.empty_cache()

        # 使用 SVC
        svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
        out2 = svc(x, weight)

        del out2
        torch.cuda.empty_cache()

    # 测量 F.unfold + einsum 的内存使用
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()

    # 运行 unfold + einsum
    x_unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
    weight_ref = weight.view(batch_size, groups, k2, height, width)
    out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
    out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

    # 计算峰值内存
    peak_mem_unfold = torch.cuda.max_memory_allocated() - start_mem

    # 清理内存
    del x_unfold, weight_ref, out1
    torch.cuda.empty_cache()

    # 测量 SVC 的内存使用
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()

    # 运行 SVC
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    out2 = svc(x, weight)

    # 计算峰值内存
    peak_mem_svc = torch.cuda.max_memory_allocated() - start_mem

    # 输出内存使用比较
    print(f"F.unfold + einsum peak memory: {peak_mem_unfold / (1024**2):.2f} MB")
    print(f"SVC peak memory:               {peak_mem_svc / (1024**2):.2f} MB")
    print(
        f"Memory saved:                  {(peak_mem_unfold - peak_mem_svc) / (1024**2):.2f} MB ({(1 - peak_mem_svc/peak_mem_unfold) * 100:.2f}%)"
    )

    return peak_mem_unfold, peak_mem_svc


def benchmark_performance():
    """比较两种方法的性能"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)

    # 设置参数
    batch_size = 8
    channels = 64
    height = width = 128
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 创建输入和权重
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    # 预热GPU缓存
    print("Warming up...")
    for _ in range(10):
        # F.unfold + einsum
        x_unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)
        loss1 = out1.sum()
        loss1.backward()

        # SVC
        svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
        out2 = svc(x, weight)
        loss2 = out2.sum()
        loss2.backward()

        # 重置梯度
        x.grad.zero_()
        weight.grad.zero_()

    # 设置CUDA事件用于精确计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 测量 unfold + einsum 前向传播性能
    print("Measuring F.unfold + einsum forward pass...")
    iterations = 50
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        x_unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)
    end.record()
    torch.cuda.synchronize()
    unfold_forward_time = start.elapsed_time(end) / iterations

    # 测量 unfold + einsum 反向传播性能
    print("Measuring F.unfold + einsum backward pass...")
    start.record()
    for _ in range(iterations):
        loss1 = out1.sum()
        loss1.backward(retain_graph=True)
        x.grad.zero_()
        weight.grad.zero_()
    end.record()
    torch.cuda.synchronize()
    unfold_backward_time = start.elapsed_time(end) / iterations

    # 测量 SVC 前向传播性能
    print("Measuring SVC forward pass...")
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()

    start.record()
    for _ in range(iterations):
        out2 = svc(x, weight)
    end.record()
    torch.cuda.synchronize()
    svc_forward_time = start.elapsed_time(end) / iterations

    # 测量 SVC 反向传播性能
    print("Measuring SVC backward pass...")
    start.record()
    for _ in range(iterations):
        loss2 = out2.sum()
        loss2.backward(retain_graph=True)
        x.grad.zero_()
        weight.grad.zero_()
    end.record()
    torch.cuda.synchronize()
    svc_backward_time = start.elapsed_time(end) / iterations

    # 输出性能结果
    print("\nPerformance Results (average time per iteration):")
    print(f"{'Operation':<25} {'F.unfold + einsum':<20} {'SVC':<20} {'Speedup':<10}")
    print("-" * 75)
    print(
        f"{'Forward pass':<25} {unfold_forward_time:.3f} ms {svc_forward_time:.3f} ms {unfold_forward_time/svc_forward_time:.2f}x"
    )
    print(
        f"{'Backward pass':<25} {unfold_backward_time:.3f} ms {svc_backward_time:.3f} ms {unfold_backward_time/svc_backward_time:.2f}x"
    )
    print(
        f"{'Total (forward + backward)':<25} {(unfold_forward_time + unfold_backward_time):.3f} ms {(svc_forward_time + svc_backward_time):.3f} ms {(unfold_forward_time + unfold_backward_time)/(svc_forward_time + svc_backward_time):.2f}x"
    )

    return {
        "unfold_forward": unfold_forward_time,
        "unfold_backward": unfold_backward_time,
        "svc_forward": svc_forward_time,
        "svc_backward": svc_backward_time,
    }


if __name__ == "__main__":
    # Run basic gradient comparison test
    print("Running basic gradient comparison test...")
    # with torch.autocast(device_type="cuda", dtype=torch.float32):
    x1, x2, weight1, weight2 = test_gradient_comparison()

    # Run parameter variation test
    print("Running parameter variation test...")
    # with torch.autocast(device_type="cuda", dtype=torch.float32):
    param_results = test_param_variation()

    # Run memory usage test
    print("Running memory usage test...")
    peak_mem_unfold, peak_mem_svc = measure_memory_usage()

    # Run performance benchmark
    print("Running performance benchmark...")
    perf_results = benchmark_performance()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

import matplotlib.pyplot as plt
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold
from svc import SpatialVariableConvolution


def test_gradient_comparison_with_dilation():
    """
    比较 dynamic_dilation_unfold + einsum 和 SVC (with dilation_map) 的梯度计算
    """
    print("\n" + "=" * 50)
    print("GRADIENT COMPARISON TEST WITH DILATION MAP")
    print("=" * 50)

    # 设置参数
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)

    # 创建输入和权重
    x1 = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight1 = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    # 创建非均匀的 dilation_map (更真实的测试场景)
    dilation_map1 = torch.ones(batch_size, groups, height, width, device="cuda") * 1.5
    # 在不同位置添加一些变化
    dilation_map1[:, :, height // 4 : height // 2, width // 4 : width // 2] = 2.0
    dilation_map1[:, :, height // 2 : 3 * height // 4, width // 2 : 3 * width // 4] = 0.8
    # 使用 torch.full 直接创建叶子张量
    dilation_map1 = torch.nn.Parameter(dilation_map1)

    # 克隆对应的张量用于 SVC 计算
    x2 = x1.clone().detach().requires_grad_(True)
    weight2 = weight1.clone().detach().requires_grad_(True)
    dilation_map2 = torch.nn.Parameter(dilation_map1.clone().detach())

    # 确保输入完全相同
    assert torch.allclose(x1, x2), "Input tensors should be identical"
    assert torch.allclose(weight1, weight2), "Weight tensors should be identical"
    assert torch.allclose(dilation_map1, dilation_map2), "Dilation map tensors should be identical"

    print("Computing gradients using dynamic_dilation_unfold + einsum...")
    # 使用 dynamic_dilation_unfold + einsum 方法
    x_unfold = dynamic_dilation_unfold(
        x1.contiguous(),
        kernel_size=kernel_size,
        dilation_map=dilation_map1,
        padding=kernel_size // 2,
        groups=groups,
    )
    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
    weight_ref = weight1.view(batch_size, groups, k2, height, width)
    out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
    out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

    # 计算损失和反向传播
    loss1 = out1.sum()
    loss1.backward()

    print("Computing gradients using SVC with dilation_map...")
    # 使用 SVC 方法 (使用 dilation_map)
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    out2 = svc(x2, weight2, dilation_map2)

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

    # 比较 dilation_map 梯度
    dilation_diff = (dilation_map1.grad - dilation_map2.grad).abs()
    max_dilation_diff = dilation_diff.max().item()
    mean_dilation_diff = dilation_diff.mean().item()
    rel_dilation_diff = (dilation_diff / (dilation_map1.grad.abs() + 1e-7)).mean().item()

    print("\nDilation Map Gradient Comparison:")
    print(f"  Max absolute difference: {max_dilation_diff:.6e}")
    print(f"  Mean absolute difference: {mean_dilation_diff:.6e}")
    print(f"  Mean relative difference: {rel_dilation_diff:.6e}")

    # 检查梯度差异是否在可接受范围内
    tolerance = 1e-4
    assert max_input_diff < tolerance, f"Input gradient max difference too large: {max_input_diff}"
    assert max_weight_diff < tolerance, f"Weight gradient max difference too large: {max_weight_diff}"
    assert max_dilation_diff < tolerance, f"Dilation map gradient max difference too large: {max_dilation_diff}"

    print("\nTest passed! Gradients from SVC match dynamic_dilation_unfold + einsum method.")

    # 绘制梯度差异分布图
    visualize_gradient_differences_with_dilation(
        x1.grad, x2.grad, weight1.grad, weight2.grad, dilation_map1.grad, dilation_map2.grad
    )

    return x1, x2, weight1, weight2, dilation_map1, dilation_map2


def visualize_gradient_differences_with_dilation(
    grad1_input, grad2_input, grad1_weight, grad2_weight, grad1_dilation, grad2_dilation
):
    """可视化带 dilation_map 的梯度差异分布"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

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

    # Dilation map 梯度差异
    dilation_diff = (grad1_dilation - grad2_dilation).abs().cpu().numpy().flatten()
    ax3.hist(dilation_diff, bins=50, alpha=0.7)
    ax3.set_title("Dilation Map Gradient Absolute Differences")
    ax3.set_xlabel("Absolute Difference")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("gradient_differences_with_dilation.png")
    # print("Gradient visualization saved as 'gradient_differences_with_dilation.png'")
    # plt.close()
    plt.show()


def measure_memory_usage_with_dilation():
    """比较带 dilation_map 时两种方法的内存使用"""
    print("\n" + "=" * 50)
    print("MEMORY USAGE COMPARISON WITH DILATION MAP")
    print("=" * 50)

    # 设置较大的参数以便明显观察内存差异
    batch_size = 8
    channels = 64
    height = width = 128
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 创建输入、权重和 dilation_map
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda")
    dilation_map = torch.ones(batch_size, groups, height, width, device="cuda") * 1.5

    # 预热GPU
    for _ in range(3):
        torch.cuda.empty_cache()
        # 使用 dynamic_dilation_unfold + einsum
        x_unfold = dynamic_dilation_unfold(
            x.contiguous(),
            kernel_size=kernel_size,
            dilation_map=dilation_map,
            padding=kernel_size // 2,
            groups=groups,
        )
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

        # 清理内存
        del x_unfold, weight_ref, out1
        torch.cuda.empty_cache()

        # 使用 SVC
        svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
        out2 = svc(x, weight, dilation_map)

        del out2
        torch.cuda.empty_cache()

    # 测量 dynamic_dilation_unfold + einsum 的内存使用
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()

    # 运行 dynamic_dilation_unfold + einsum
    x_unfold = dynamic_dilation_unfold(
        x.contiguous(),
        kernel_size=kernel_size,
        dilation_map=dilation_map,
        padding=kernel_size // 2,
        groups=groups,
    )
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
    out2 = svc(x, weight, dilation_map)

    # 计算峰值内存
    peak_mem_svc = torch.cuda.max_memory_allocated() - start_mem

    # 输出内存使用比较
    print(f"dynamic_dilation_unfold + einsum peak memory: {peak_mem_unfold / (1024**2):.2f} MB")
    print(f"SVC with dilation_map peak memory:             {peak_mem_svc / (1024**2):.2f} MB")
    print(
        f"Memory saved:                                  {(peak_mem_unfold - peak_mem_svc) / (1024**2):.2f} MB ({(1 - peak_mem_svc/peak_mem_unfold) * 100:.2f}%)"
    )

    return peak_mem_unfold, peak_mem_svc


def benchmark_performance_with_dilation():
    """比较带 dilation_map 时两种方法的性能"""
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK WITH DILATION MAP")
    print("=" * 50)

    # 设置参数
    batch_size = 8
    channels = 64
    height = width = 128
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 创建输入、权重和 dilation_map
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)
    dilation_map = torch.nn.Parameter(torch.ones(batch_size, groups, height, width, device="cuda") * 1.5)

    # 预热GPU缓存
    print("Warming up...")
    for _ in range(10):
        # dynamic_dilation_unfold + einsum
        x_unfold = dynamic_dilation_unfold(
            x.contiguous(),
            kernel_size=kernel_size,
            dilation_map=dilation_map,
            padding=kernel_size // 2,
            groups=groups,
        )
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)
        loss1 = out1.sum()
        loss1.backward()

        # SVC
        svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
        out2 = svc(x, weight, dilation_map)
        loss2 = out2.sum()
        loss2.backward()

        # 重置梯度
        x.grad.zero_()
        weight.grad.zero_()
        dilation_map.grad.zero_()

    # 设置CUDA事件用于精确计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 测量 dynamic_dilation_unfold + einsum 前向传播性能
    print("Measuring dynamic_dilation_unfold + einsum forward pass...")
    iterations = 50
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        x_unfold = dynamic_dilation_unfold(
            x.contiguous(),
            kernel_size=kernel_size,
            dilation_map=dilation_map,
            padding=kernel_size // 2,
            groups=groups,
        )
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)
    end.record()
    torch.cuda.synchronize()
    unfold_forward_time = start.elapsed_time(end) / iterations

    # 测量 dynamic_dilation_unfold + einsum 反向传播性能
    print("Measuring dynamic_dilation_unfold + einsum backward pass...")
    start.record()
    for _ in range(iterations):
        loss1 = out1.sum()
        loss1.backward(retain_graph=True)
        x.grad.zero_()
        weight.grad.zero_()
        dilation_map.grad.zero_()
    end.record()
    torch.cuda.synchronize()
    unfold_backward_time = start.elapsed_time(end) / iterations

    # 测量 SVC 前向传播性能
    print("Measuring SVC with dilation_map forward pass...")
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()

    start.record()
    for _ in range(iterations):
        out2 = svc(x, weight, dilation_map)
    end.record()
    torch.cuda.synchronize()
    svc_forward_time = start.elapsed_time(end) / iterations

    # 测量 SVC 反向传播性能
    print("Measuring SVC with dilation_map backward pass...")
    start.record()
    for _ in range(iterations):
        loss2 = out2.sum()
        loss2.backward(retain_graph=True)
        x.grad.zero_()
        weight.grad.zero_()
        dilation_map.grad.zero_()
    end.record()
    torch.cuda.synchronize()
    svc_backward_time = start.elapsed_time(end) / iterations

    # 输出性能结果
    print("\nPerformance Results (average time per iteration):")
    print(f"{'Operation':<25} {'dynamic_dilation_unfold':<24} {'SVC+dilation_map':<20} {'Speedup':<10}")
    print("-" * 80)
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


def test_comparison_with_varying_dilation_map():
    """
    对比在不同 dilation_map 数值下的梯度差异
    """
    print("\n" + "=" * 50)
    print("GRADIENT COMPARISON WITH VARYING DILATION VALUES")
    print("=" * 50)

    # 设置参数
    batch_size = 2
    channels = 6
    height = 24
    width = 24
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # 设置随机种子
    torch.manual_seed(42)

    # 创建输入和权重 (一次性创建，多次使用)
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    # 测试不同的 dilation 值
    dilation_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = []

    for dil_val in dilation_values:
        print(f"\nTesting with dilation value: {dil_val}")

        # 创建固定值的 dilation_map
        dilation_map1 = torch.nn.Parameter(torch.ones(batch_size, groups, height, width, device="cuda") * dil_val)
        dilation_map2 = torch.nn.Parameter(dilation_map1.clone().detach())

        # 重置输入和权重的梯度
        if x.grad is not None:
            x.grad.zero_()
            weight.grad.zero_()

        x1 = x.clone().detach().requires_grad_(True)
        weight1 = weight.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)
        weight2 = weight.clone().detach().requires_grad_(True)

        # 使用 dynamic_dilation_unfold + einsum
        x_unfold = dynamic_dilation_unfold(
            x1.contiguous(),
            kernel_size=kernel_size,
            dilation_map=dilation_map1,
            padding=kernel_size // 2,
            groups=groups,
        )
        x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
        weight_ref = weight1.view(batch_size, groups, k2, height, width)
        out1 = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
        out1 = out1.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

        # 计算损失和反向传播
        loss1 = out1.sum()
        loss1.backward()

        # 使用 SVC
        svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
        out2 = svc(x2, weight2, dilation_map2)

        # 计算损失和反向传播
        loss2 = out2.sum()
        loss2.backward()

        # 比较结果
        output_diff = (out1 - out2).abs()
        max_out_diff = output_diff.max().item()
        mean_out_diff = output_diff.mean().item()

        input_diff = (x1.grad - x2.grad).abs()
        max_input_diff = input_diff.max().item()
        mean_input_diff = input_diff.mean().item()

        weight_diff = (weight1.grad - weight2.grad).abs()
        max_weight_diff = weight_diff.max().item()
        mean_weight_diff = weight_diff.mean().item()

        dilation_diff = (dilation_map1.grad - dilation_map2.grad).abs()
        max_dilation_diff = dilation_diff.max().item()
        mean_dilation_diff = dilation_diff.mean().item()

        # 记录结果
        results.append(
            {
                "dilation_value": dil_val,
                "max_out_diff": max_out_diff,
                "mean_out_diff": mean_out_diff,
                "max_input_diff": max_input_diff,
                "mean_input_diff": mean_input_diff,
                "max_weight_diff": max_weight_diff,
                "mean_weight_diff": mean_weight_diff,
                "max_dilation_diff": max_dilation_diff,
                "mean_dilation_diff": mean_dilation_diff,
            }
        )

        print(f"Output diff - Max: {max_out_diff:.6e}, Mean: {mean_out_diff:.6e}")
        print(f"Input grad diff - Max: {max_input_diff:.6e}, Mean: {mean_input_diff:.6e}")
        print(f"Weight grad diff - Max: {max_weight_diff:.6e}, Mean: {mean_weight_diff:.6e}")
        print(f"Dilation grad diff - Max: {max_dilation_diff:.6e}, Mean: {mean_dilation_diff:.6e}")

    # 可视化结果随 dilation 值变化的趋势
    plt.figure(figsize=(12, 8))

    # 提取结果
    dil_values = [r["dilation_value"] for r in results]
    max_out_diffs = [r["max_out_diff"] for r in results]
    max_input_diffs = [r["max_input_diff"] for r in results]
    max_weight_diffs = [r["max_weight_diff"] for r in results]
    max_dilation_diffs = [r["max_dilation_diff"] for r in results]

    # 绘图
    plt.plot(dil_values, max_out_diffs, "o-", label="Output Difference")
    plt.plot(dil_values, max_input_diffs, "o-", label="Input Gradient Difference")
    plt.plot(dil_values, max_weight_diffs, "o-", label="Weight Gradient Difference")
    plt.plot(dil_values, max_dilation_diffs, "o-", label="Dilation Gradient Difference")

    plt.xlabel("Dilation Value")
    plt.ylabel("Maximum Absolute Difference")
    plt.title("Maximum Differences vs Dilation Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")

    plt.tight_layout()
    # plt.savefig("dilation_value_comparison.png")
    # print("Dilation value comparison saved as 'dilation_value_comparison.png'")
    plt.show()

    return results


if __name__ == "__main__":
    from dynamic_dilation_unfold import dynamic_dilation_unfold

    print("\n" + "=" * 50)
    print("DILATION MAP MODE TESTS")
    print("=" * 50)

    # 1. 测试带 dilation_map 的梯度计算一致性
    x1_dil, x2_dil, w1_dil, w2_dil, dmap1, dmap2 = test_gradient_comparison_with_dilation()

    # 2. 测量带 dilation_map 的内存使用
    peak_mem_unfold_dil, peak_mem_svc_dil = measure_memory_usage_with_dilation()

    # 3. 基准测试带 dilation_map 的性能
    perf_results_dil = benchmark_performance_with_dilation()

    # 4. 不同 dilation 值下的对比
    dilation_comparison_results = test_comparison_with_varying_dilation_map()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

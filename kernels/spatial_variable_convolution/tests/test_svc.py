import torch
import torch.nn.functional as F
from svc import SpatialVariableConvolution


def test_spatial_variable_conv_basic():
    # Basic test with standard convolution (without dilation map)
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size

    # Create input and weights
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda")

    # Create module
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()

    # Forward pass (without dilation map - uses integer indexing)
    output = svc(x, weight)

    # Check output shape
    assert output.shape == (batch_size, channels, height, width)

    # Check output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))

    print("Basic test passed")


def test_against_unfold_reference():
    # Compare against the unfold+einsum reference implementation
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups

    # Create input and weights
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight_flat = torch.randn(batch_size, groups * k2, height, width, device="cuda")

    # Reference implementation
    x_unfold = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2)
    x_unfold = x_unfold.view(batch_size, groups, channels_per_group, k2, height, width)
    weight_ref = weight_flat.view(batch_size, groups, k2, height, width)
    x_out_ref = torch.einsum("bgckhw,bgkhw->bgchw", x_unfold, weight_ref)
    x_out_ref = x_out_ref.flatten(1, 2).contiguous().view(batch_size, channels, height, width)

    # Our implementation without dilation map (integer indexing)
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    x_out = svc(x, weight_flat)

    # Check output
    torch.testing.assert_close(x_out, x_out_ref, rtol=1e-4, atol=1e-4)

    print("Reference comparison test passed")


def test_with_dilation_map():
    # Test with dynamic dilation map (uses bilinear interpolation)
    batch_size = 2
    channels = 6
    height = 32
    width = 32
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size

    # Create input, weights, and dilation map
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda")
    dilation_map = torch.ones(batch_size, groups, height, width, device="cuda") * 1.5

    # Create module
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()

    # Forward pass with dilation map (uses bilinear interpolation)
    output = svc(x, weight, dilation_map)

    # Check output shape
    assert output.shape == (batch_size, channels, height, width)

    # Check output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))

    print("Dilation map test passed")


def test_backward_without_dilation_map():
    # Test backward pass with direct integer indexing (no dilation map)
    batch_size = 2
    channels = 6
    height = 16
    width = 16
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size

    # Create tensors with gradients enabled
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    # Forward pass without dilation map
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    output = svc(x, weight)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()

    print("Backward pass (without dilation map) test passed")


def test_backward_with_dilation_map():
    # Test backward pass with bilinear interpolation (with dilation map)
    batch_size = 2
    channels = 6
    height = 16
    width = 16
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size

    # Create tensors with gradients enabled
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda", requires_grad=True)

    # Create dilation map and retain its gradient
    dilation_map = torch.ones(batch_size, groups, height, width, device="cuda") * 1.5
    dilation_map.requires_grad_(True)
    dilation_map.retain_grad()  # This line ensures the gradient is retained

    # Forward pass with dilation map
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()
    output = svc(x, weight, dilation_map)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert dilation_map.grad is not None and torch.isfinite(dilation_map.grad).all()

    print("Backward pass (with dilation map) test passed")


def test_performance_comparison():
    # Compare performance between with/without dilation map
    batch_size = 8
    channels = 64
    height = width = 128
    groups = 4
    kernel_size = 3
    k2 = kernel_size * kernel_size

    # Create input and weights
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    weight = torch.randn(batch_size, groups * k2, height, width, device="cuda")
    dilation_map = torch.ones(batch_size, groups, height, width, device="cuda")

    # Create module
    svc = SpatialVariableConvolution(kernel_size=kernel_size, padding=kernel_size // 2, groups=groups).cuda()

    # Warmup
    for _ in range(5):
        svc(x, weight)
        svc(x, weight, dilation_map)

    # Benchmark without dilation map
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        svc(x, weight)
    end.record()
    torch.cuda.synchronize()
    time_without_dilation = start.elapsed_time(end)

    # Benchmark with dilation map
    start.record()
    for _ in range(100):
        svc(x, weight, dilation_map)
    end.record()
    torch.cuda.synchronize()
    time_with_dilation = start.elapsed_time(end)

    print(f"Time without dilation map: {time_without_dilation:.2f}ms")
    print(f"Time with dilation map: {time_with_dilation:.2f}ms")
    print(f"Speedup factor: {time_with_dilation/time_without_dilation:.2f}x")

    # The version without dilation map should be faster
    assert time_without_dilation < time_with_dilation

    print("Performance comparison test passed")


if __name__ == "__main__":
    print("Running tests...")
    test_spatial_variable_conv_basic()
    test_against_unfold_reference()
    test_with_dilation_map()
    test_backward_without_dilation_map()
    test_backward_with_dilation_map()
    test_performance_comparison()
    print("All tests passed!")

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    from . import _ext as _backend
except ImportError:
    raise ImportError("Could not import CUDA extension. Please build the extension first.")


class SpatialVariableConvolutionFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, dilation_map=None, padding=0, stride=1, dilation=1, groups=1):
        """
        Args:
            input: (B, C, H, W)
            weight: (B, groups*K, H_out, W_out) where K is the number of kernels per group
            kernel_size: int or tuple - size of the convolutional kernel
            dilation_map: Optional (B, groups, H_out, W_out) - dilation values for each position
            padding: int or tuple - zero padding added to both sides of the input
            stride: int or tuple - stride of the convolution
            dilation: int or tuple - base dilation value
            groups: int - number of groups for grouped convolution

        Returns:
            output: (B, C_out, H_out, W_out)
        """
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        batch_size = input.size(0)
        in_channels = input.size(1)
        height = input.size(2)
        width = input.size(3)

        # Calculate output dimensions with stride
        height_out = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_out = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Check inputs
        assert groups > 0, "Number of groups should be positive"
        assert in_channels % groups == 0, f"Number of channels ({in_channels}) should be divisible by groups ({groups})"
        channels_per_group = in_channels // groups

        k2 = kernel_h * kernel_w

        # Check weight dimensions
        assert weight.dim() == 4, "weight should be a 4D tensor"
        assert weight.size(0) == batch_size, f"Batch size mismatch: weight ({weight.size(0)}) vs input ({batch_size})"

        weight_groups = weight.size(1) // k2
        assert (
            weight_groups == groups
        ), f"Number of weight groups ({weight_groups}) must match the specified groups ({groups})"

        # Flag to determine if we're using a dilation map
        use_dilation_map = dilation_map is not None

        if use_dilation_map:
            # Process with dilation map as before
            assert dilation_map.dim() == 4, "dilation_map should be a 4D tensor"
            assert dilation_map.size(0) == batch_size, "Batch size mismatch in dilation_map"
            assert dilation_map.size(1) == groups, "Number of groups mismatch in dilation_map"
            assert dilation_map.size(2) == height_out, "Height mismatch in dilation_map"
            assert dilation_map.size(3) == width_out, "Width mismatch in dilation_map"
            dilation_map_flat = dilation_map.contiguous().view(batch_size, groups, -1)
        else:
            # Create a dummy empty tensor - the kernel won't use this when use_dilation_map is False
            # This avoids creating a large tensor of ones
            dilation_map_flat = torch.empty(1, 1, 1, device=input.device, dtype=input.dtype)

        # Reshape weight for the CUDA kernel
        weight_flat = weight.view(batch_size, groups, k2, -1)

        # Call the CUDA extension with the use_dilation_map flag and stride parameters
        output = _backend.forward(
            input.contiguous(),
            weight_flat.contiguous(),
            dilation_map_flat.contiguous(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            channels_per_group,
            height_out,
            width_out,
            use_dilation_map,
        )

        ctx.save_for_backward(input, weight_flat, dilation_map_flat)
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.groups = groups
        ctx.channels_per_group = channels_per_group
        ctx.height_out = height_out
        ctx.width_out = width_out
        ctx.use_dilation_map = use_dilation_map

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight_flat, dilation_map_flat = ctx.saved_tensors

        grad_input, grad_weight, grad_dilation_map = _backend.backward(
            grad_output.contiguous(),
            input,
            weight_flat,
            dilation_map_flat,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.groups,
            ctx.channels_per_group,
            ctx.height_out,
            ctx.width_out,
            ctx.use_dilation_map,
        )

        # Reshape grad_weight to match the original weight shape
        batch_size = grad_weight.size(0)
        grad_weight = grad_weight.view(
            batch_size, ctx.groups * ctx.kernel_h * ctx.kernel_w, ctx.height_out, ctx.width_out
        )

        if ctx.use_dilation_map:
            # Reshape grad_dilation_map to match the original dilation_map shape
            grad_dilation_map = grad_dilation_map.view(batch_size, ctx.groups, ctx.height_out, ctx.width_out)
            return grad_input, grad_weight, None, grad_dilation_map, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None, None


def spatial_variable_convolution(
    input, weight, kernel_size, dilation_map=None, padding=0, stride=1, dilation=1, groups=1
):
    """
    Spatial Variable Convolution

    Args:
        input: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (B, groups*K, H_out, W_out)
        kernel_size: Size of the convolution kernel (int or tuple)
        dilation_map: Optional tensor of shape (B, groups, H_out, W_out) for variable dilation
        padding: Zero-padding added to both sides of the input (int or tuple)
        stride: Stride of the convolution (int or tuple)
        dilation: Base dilation factor (int or tuple)
        groups: Number of blocked connections from input channels to output channels

    Returns:
        output: Tensor of shape (B, C_out, H_out, W_out)

    Examples:
        >>> # Standard convolution (faster, uses integer indexing)
        >>> input = torch.randn(2, 6, 32, 32).cuda()
        >>> weight = torch.randn(2, 18, 30, 30).cuda()  # 2 groups, kernel size 3x3
        >>> output = spatial_variable_convolution(input, weight, kernel_size=3, groups=2)
        >>>
        >>> # With dynamic dilation (uses bilinear interpolation)
        >>> dilation_map = torch.ones(2, 2, 30, 30).cuda() * 1.5
        >>> output = spatial_variable_convolution(input, weight, kernel_size=3, dilation_map=dilation_map, groups=2)
    """
    return SpatialVariableConvolutionFunction.apply(
        input, weight, kernel_size, dilation_map, padding, stride, dilation, groups
    )

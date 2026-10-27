import torch
import torch.nn as nn
from .functions import spatial_variable_convolution


class SpatialVariableConvolution(nn.Module):
    """
    PyTorch module for Spatial Variable Convolution
    
    Args:
        kernel_size: Size of the convolution kernel
        padding: Zero-padding added to both sides of the input
        stride: Stride of the convolution
        dilation: Base dilation factor
        groups: Number of blocked connections from input channels to output channels
    
    Example:
        >>> svc = SpatialVariableConvolution(kernel_size=3, padding=1, stride=2, groups=2).cuda()
        >>> input = torch.randn(2, 6, 32, 32).cuda()
        >>> weight = torch.randn(2, 18, 16, 16).cuda()  # 2 groups, 3x3 kernel, output 16x16
        >>> output = svc(input, weight)  # Standard convolution (faster)
        >>> # With dilation map (using bilinear interpolation)
        >>> dilation_map = torch.ones(2, 2, 16, 16).cuda() * 1.5
        >>> output = svc(input, weight, dilation_map)
    """
    
    def __init__(self, kernel_size, padding=0, stride=1, dilation=1, groups=1):
        super(SpatialVariableConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, input, weight, dilation_map=None):
        """
        Args:
            input: Input tensor of shape (B, C, H, W)
            weight: Weight tensor of shape (B, groups*K, H_out, W_out)
            dilation_map: Optional tensor of shape (B, groups, H_out, W_out) for variable dilation
        
        Returns:
            output: Tensor of shape (B, C_out, H_out, W_out)
        """
        return spatial_variable_convolution(
            input, weight, self.kernel_size, dilation_map,
            self.padding, self.stride, self.dilation, self.groups
        )
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, padding={self.padding}, ' \
               f'stride={self.stride}, dilation={self.dilation}, groups={self.groups}'
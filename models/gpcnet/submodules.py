# coding=utf-8

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from svc import spatial_variable_convolution as sv_conv


class ConvBNSiLU(nn.Sequential):
    """Convolution + BatchNorm + SiLU Module.

    Params:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the convolution kernel
        stride: convolution stride
        padding: convolution padding
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        bn = nn.BatchNorm2d(out_channels)
        act = nn.SiLU(inplace=True)
        super().__init__(conv, bn, act)


class ConvUpsample(nn.Sequential):
    """Convolution + Upsample Module.

    Params:
        in_channels: number of input channels
        out_channels: number of output channels
        upsample_scale: upsampling factor
    """

    def __init__(self, in_channels, out_channels=None, upsample_scale=2):
        out_channels = in_channels if out_channels is None else out_channels
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        upsample = nn.Upsample(scale_factor=upsample_scale, mode="bilinear", align_corners=False)
        super().__init__(conv, upsample)


class SegmentationHead(nn.Sequential):
    """Segmentation Head Module.

    Params:
        in_channels: number of input channels
        out_channels: number of output channels
        upsample_scale: upsampling factor
    """

    def __init__(self, in_channels, out_channels, upsample_scale=4):
        conv3 = ConvBNSiLU(in_channels, in_channels, kernel_size=3)
        drop = nn.Dropout2d(p=0.1)
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        upsample = nn.Upsample(scale_factor=upsample_scale, mode="bilinear", align_corners=False)
        super().__init__(conv3, drop, conv1, upsample)


class SpatialVariableLowPassKernel(nn.Module):
    """Spatial Variable Low-Pass Kernel Module.

    Params:
        channels: number of channels of the input feature
        kernel_size: size of the low-pass kernel
        groups: number of filter groups
        spatial_reduction: feature spatial reduction ratio for weight generation
        channel_reduction: feature channel reduction ratio for weight generation
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        spatial_reduction: int = 2,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.spatial_reduction = spatial_reduction

        self.s2 = spatial_reduction * spatial_reduction
        self.k2 = kernel_size * kernel_size

        in_chans = channels * self.s2
        hidden_chans = in_chans // channel_reduction
        self.weights_gen = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.Conv2d(hidden_chans, self.s2 * groups * self.k2, kernel_size=1),
        )

    def generate_weights(self, x):
        b, c, H, W = x.size()
        assert H % self.spatial_reduction == 0 and W % self.spatial_reduction == 0, (
            "Input feature size must be divisible by spatial_reduction."
        )
        h = H // self.spatial_reduction
        w = W // self.spatial_reduction
        x = F.pixel_unshuffle(x, self.spatial_reduction)
        weights = self.weights_gen(x)
        weights = weights.view(b, self.s2, self.groups, self.k2, h, w)
        weights = torch.softmax(weights, dim=3)
        weights = weights.view(b * self.s2, self.groups * self.k2, h, w)
        return weights

    def forward(self, x, weight):
        """Apply spatial-variant low-pass filter"""
        b, c, h, w = x.size()
        assert c % self.groups == 0, "Input channels must be divisible by number of groups."
        assert weight.size() == (b * self.s2, self.groups * self.k2, h, w), "Weight size mismatch."
        x = x.unsqueeze(1).repeat(1, self.s2, 1, 1, 1).view(b * self.s2, c, h, w)
        x_out = sv_conv(
            x.contiguous(),
            weight.to(x),
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
        )
        x_out = x_out.view(b, self.s2, -1, h, w).transpose(1, 2).contiguous().view(b, -1, h, w)
        x_out = F.pixel_shuffle(x_out, self.spatial_reduction)
        return x_out


class SpatialVariableHighPassKernel(nn.Module):
    """Spatial Variable High-Pass Kernel Module.

    Params:
        channels: number of channels of the input feature
        kernel_size: size of the high-pass kernel
        groups: number of filter groups
        channel_reduction: feature channel reduction ratio for weight generation
    """

    def __init__(self, channels: int, kernel_size: int, groups: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.k2 = kernel_size * kernel_size
        self.weights_gen = nn.Conv2d(channels, groups * self.k2, kernel_size=3, padding=1)

        identity_weight = torch.zeros(1, 1, self.k2, 1, 1)
        identity_weight[:, :, self.k2 // 2, :, :] = 1.0
        self.register_buffer("identity_weight", identity_weight)

    def generate_weights(self, x):
        b, c, h, w = x.size()
        assert c % self.groups == 0, "Input channels must be divisible by number of groups."
        weights = self.weights_gen(x)
        weights = weights.view(b, self.groups, self.k2, h, w)
        weights = torch.softmax(weights, dim=2)
        assert self.identity_weight.size() == (1, 1, self.k2, 1, 1), "Identity weight size mismatch."
        assert len(self.identity_weight.size()) == len(weights.size()), "Identity weight dimension mismatch."
        weights = self.identity_weight - weights
        weights = weights.view(b, self.groups * self.k2, h, w)
        return weights

    def forward(self, x, weight):
        """Apply spatial-variant high-pass filter"""
        b, c, H, W = x.size()
        assert c % self.groups == 0, "Input channels must be divisible by number of groups."
        assert weight.size() == (b, self.groups * self.k2, H, W), "Weight size mismatch."
        x_out = sv_conv(
            x.contiguous(),
            weight.to(x),
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.groups,
        )
        x_out = x + x_out
        return x_out


class SpatialVariableDynamicDilationKernel(nn.Module):
    """Spatial Variable Dynamic Dilation Kernel Module.

    Params:
        channels: number of channels of the input feature
        kernel_size: size of the dilation kernel
        groups: number of filter groups
        stride: convolution stride
        min_dilation: minimum dilation value for stability (default: -1, meaning no minimum constraint)
        max_dilation: maximum dilation value for stability (optional)
    """

    def __init__(self, channels: int, kernel_size: int, groups: int, min_dilation=-1, max_dilation=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_dilation = min_dilation
        self.max_dilation = max_dilation

        self.k2 = self.kernel_size * self.kernel_size

        self.weights_gen = nn.Conv2d(channels, groups * self.k2, kernel_size=3, padding=1)
        self.dilations_gen = nn.Conv2d(channels, groups, kernel_size=3, padding=1)

    def generate_weights_dilations(self, x):
        b, c, h, w = x.size()
        assert c % self.groups == 0, "Input channels must be divisible by number of groups."
        weights = self.weights_gen(x)
        dilation_map = self.dilations_gen(x)
        if not hasattr(self, "max_dilation"):
            self.max_dilation = None
        if not hasattr(self, "min_dilation"):
            self.min_dilation = 1.0
        if self.max_dilation is not None:
            dilation_map = F.hardtanh(F.relu(dilation_map) + 1.0, min_val=self.min_dilation, max_val=self.max_dilation)
        else:
            dilation_map = F.hardtanh(F.relu(dilation_map) + 1.0, min_val=self.min_dilation)
        return weights, dilation_map

    def forward(self, x, weight, dilation_map):
        """Apply spatial-variant high-pass filter"""
        b, c, H, W = x.size()
        assert c % self.groups == 0, "Input channels must be divisible by number of groups."
        assert weight.size() == (b, self.groups * self.k2, H, W), "Weight size mismatch."
        assert dilation_map.size() == (b, self.groups, H, W), "Dilation map size mismatch."
        x_out = sv_conv(
            x.contiguous(),
            weight.to(x),
            kernel_size=self.kernel_size,
            dilation_map=dilation_map.to(x),
            padding=self.kernel_size // 2,
            groups=self.groups,
        )
        return x_out


class SpatialAdaptiveFusion(nn.Module):
    """Spatial Adaptive Fusion Module.

    Params:
        channels: number of channels of the input features
        kernel_size: size of the spatial-variable kernels
        groups: number of filter groups
        spatial_reduction: feature stride between high-level and low-level features
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        spatial_reduction: int,
        min_dilation=-1,
        max_dilation=None,
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.spatial_reduction = spatial_reduction
        self.min_dilation = min_dilation
        self.max_dilation = max_dilation

        self.ll_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.hl_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.init_lpk = SpatialVariableLowPassKernel(channels, kernel_size, groups, spatial_reduction)
        self.init_hpk = SpatialVariableHighPassKernel(channels, kernel_size, groups)
        self.init_fuse_proj = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)

        self.ddk = SpatialVariableDynamicDilationKernel(
            channels,
            kernel_size,
            groups,
            min_dilation=self.min_dilation,
            max_dilation=self.max_dilation,
        )

        self.final_lpk = SpatialVariableLowPassKernel(channels, kernel_size, groups, spatial_reduction)
        self.final_hpk = SpatialVariableHighPassKernel(channels, kernel_size, groups)
        self.final_fuse_proj = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, high_level, low_level):
        b, c, h, w = high_level.size()
        b, c, H, W = low_level.size()
        assert H % h == 0 and H // h == self.spatial_reduction and W % w == 0 and W // w == self.spatial_reduction, (
            "Low-level feature size must be spatial_reduction times high-level feature size."
        )

        high_level = self.hl_proj(high_level)
        low_level = self.ll_proj(low_level)

        high_level_out = self.init_lpk(high_level, self.init_lpk.generate_weights(low_level))
        low_level_out = self.init_hpk(low_level, self.init_hpk.generate_weights(low_level))

        init_fuse = torch.cat([high_level_out, low_level_out], dim=1)
        init_fuse = self.init_fuse_proj(init_fuse)

        dd_weights, dd_dilations = self.ddk.generate_weights_dilations(init_fuse)
        init_fuse = self.ddk(init_fuse, dd_weights, dd_dilations)

        high_level_out = self.final_lpk(high_level, self.final_lpk.generate_weights(init_fuse))
        low_level_out = self.final_hpk(low_level, self.final_hpk.generate_weights(init_fuse))

        final_fuse = torch.cat([high_level_out, low_level_out], dim=1)
        final_fuse = self.final_fuse_proj(final_fuse)

        return init_fuse + final_fuse


class ProgressFeatureFusion(nn.Module):
    """Progressly Feature Fusion Decoder for Semantic Segmentation.

    Params:
        num_classes: Number of output segmentation classes.
        image_size: Tuple specifying the input image size (height, width).
        feature_dims: List of input feature dimensions from the encoder at different scales.
        decoder_dims: Dimension of the decoder feature maps.
        kernel_size: Size of the spatial-variable kernels.
        num_heads: Number of attention heads.
        groups: Number of filter groups.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int] = (512, 512),
        feature_dims: List[int] = [48, 96, 240, 384],
        decoder_dims: int = 128,
        kernel_size: int = 3,
        min_dilation: int = -1,
        max_dilation: Optional[int] = None,
        groups: int = 4,
        aux_heads: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.feature_dims = feature_dims
        self.decoder_dims = decoder_dims
        self.kernel_size = kernel_size
        self.groups = groups
        self.aux_heads = aux_heads

        self.feats_convs = nn.ModuleList()
        self.feats_fusion = nn.ModuleList()
        for i, f in enumerate(feature_dims):
            self.feats_convs.append(ConvBNSiLU(f, decoder_dims, kernel_size=1))
            if i < len(feature_dims) - 1:
                self.feats_fusion.append(
                    SpatialAdaptiveFusion(
                        decoder_dims,
                        kernel_size,
                        groups,
                        spatial_reduction=2,
                        min_dilation=min_dilation,
                        max_dilation=max_dilation,
                    )
                )

        self.seg_head = SegmentationHead(decoder_dims, num_classes, upsample_scale=4)

        if aux_heads:
            self.aux_head1 = SegmentationHead(decoder_dims, num_classes, upsample_scale=4)
            self.aux_head2 = SegmentationHead(decoder_dims, num_classes, upsample_scale=4)

    def forward(self, feats):
        feats = [conv(feat) for conv, feat in zip(self.feats_convs, feats)]
        for i in range(len(feats) - 2, -1, -1):
            feats[i] = self.feats_fusion[i](feats[i + 1], feats[i])

        if self.aux_heads:
            seg = self.seg_head(feats[-4])
            aux1 = self.aux_head1(feats[-3])
            aux2 = self.aux_head2(feats[-2])
            return seg, aux1, aux2
        else:
            seg = self.seg_head(feats[-4])
            return seg

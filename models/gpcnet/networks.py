# coding=utf-8


from typing import List, Tuple

import einops
import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

from ..mix_transformers import Attention as MitAttention
from ..mix_transformers import MixVisionTransformer
from .submodules import ProgressFeatureFusion

__all__ = [
    "GPCNet",
    "GPCNetMitB0",
    "GPCNetMitB1",
    "GPCNetMitB2",
    "GPCNetResNet",
    "GPCNetResNet18",
    "GPCNetResNet50",
    "GPCNetResNext50",
    "GPCNetSwinT",
    "GPCNetPVTv2B2",
    "GPCNetTwinsSVTSmall",
]


class GPCNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder: nn.Module,
        enc_chans: List[int],
        dec_chans: int,
        image_size: Tuple[int, int],
        num_grids: Tuple[int, int] = (4, 4),
        enable_intx: List[bool] = [True, True, True, True],
        kernel_size: int = 3,
        groups: int = 4,
        aux_heads: bool = False,
    ):
        super(GPCNet, self).__init__()

        self.encoder = encoder
        self.image_size = image_size
        self.num_grids = num_grids
        self.enable_intx = enable_intx

        self.semantic_decoder = ProgressFeatureFusion(
            num_classes=num_classes,
            image_size=image_size,
            feature_dims=enc_chans,
            decoder_dims=dec_chans,
            kernel_size=kernel_size,
            groups=groups,
            aux_heads=aux_heads,
        )

        self.hooks = []
        self.setup_hooks()

    def forward_encoder(self, x: torch.Tensor):
        """Extract features from the semantic encoder."""
        raise NotImplementedError("please implement this method in the subclass.")

    def forward_encoder_last_feat(self, x: torch.Tensor):
        """Extract only the last feature from the semantic encoder."""
        feats = self.forward_encoder(x)
        return feats[-1]

    @staticmethod
    def split_x(x: torch.Tensor, num_grids: Tuple[int, int]):
        """Split input tensor x into grids for efficient processing."""
        nh, nw = num_grids
        x = einops.rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)
        return x

    @staticmethod
    def merge_x(x: torch.Tensor, num_grids: Tuple[int, int]):
        """Merge grids back into the whole scene."""
        nh, nw = num_grids
        x = einops.rearrange(x, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)
        return x

    def forward(self, x):
        x = self.split_x(x, self.num_grids)
        feats = self.forward_encoder(x)
        feats = [self.merge_x(feat, self.num_grids) for feat in feats]
        logits = self.semantic_decoder(feats)
        return logits

    def hook_modify_input(self, module, input):
        x = input[0]
        x = self.merge_x(x, self.num_grids)
        return (x,)

    def hook_modify_output(self, module, input, output):
        x = output
        x = self.split_x(x, self.num_grids)
        return x

    def setup_hooks(self):
        """Setup hooks for the encoder to extract features."""
        self.remove_hooks()

        backbone = self.encoder
        if isinstance(backbone, MixVisionTransformer):
            attn_layers = [backbone.block1[-1], backbone.block2[-1], backbone.block3[-1], backbone.block4[-1]]
        else:
            print(f"Warning: Hooks are not set up for this encoder type {type(backbone)}")

        for layer, enable in zip(attn_layers, self.enable_intx):
            if enable:
                self.hooks.append(layer.register_forward_pre_hook(self.hook_modify_input))
                self.hooks.append(layer.register_forward_hook(self.hook_modify_output))

    def remove_hooks(self):
        """Remove hooks from the encoder."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GPCNetSMP(GPCNet):
    def __init__(self, backbone_name: str, image_size: Tuple[int, int], *args, **kwargs):
        encoder = smp.encoders.get_encoder(backbone_name, weights="imagenet")
        encoder_channels = encoder.out_channels[-4:]

        if isinstance(encoder, MixVisionTransformer):
            for m in encoder.modules():
                if isinstance(m, MitAttention):
                    q_dim = m.q.out_features
                    q_bias = m.q.bias is not None
                    m.k = nn.Linear(q_dim, q_dim, bias=q_bias)
                    m.v = nn.Linear(q_dim, q_dim, bias=q_bias)
                    m.k.weight.data.copy_(m.kv.weight.data[:q_dim, :])
                    m.v.weight.data.copy_(m.kv.weight.data[q_dim:, :])
                    if q_bias:
                        m.k.bias.data.copy_(m.kv.bias.data[:q_dim])
                        m.v.bias.data.copy_(m.kv.bias.data[q_dim:])
                    del m.kv

        super(GPCNetSMP, self).__init__(
            encoder=encoder,
            enc_chans=encoder_channels,
            image_size=image_size,
            *args,
            **kwargs,
        )

    def forward_encoder(self, x: torch.Tensor):
        """Extract features from the semantic encoder."""
        return self.encoder(x)[-4:]


class GPCNetMitB0(GPCNetSMP):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetMitB0, self).__init__(
            backbone_name="mit-b0",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetMitB1(GPCNetSMP):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetMitB1, self).__init__(
            backbone_name="mit-b1",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetMitB2(GPCNetSMP):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetMitB2, self).__init__(
            backbone_name="mit-b2",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetResNet(GPCNet):
    """S2Seg with ResNet as the backbone encoder."""

    def __init__(self, backbone_name, image_size: Tuple[int, int], *args, **kwargs):
        encoder = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=True,
        )
        encoder_channels = encoder.feature_info.channels()
        super(GPCNetResNet, self).__init__(
            encoder=encoder,
            enc_chans=encoder_channels,
            image_size=image_size,
            *args,
            **kwargs,
        )

    def forward_encoder(self, x: torch.Tensor):
        """Extract features from the semantic encoder."""
        return self.encoder(x)


class GPCNetResNet18(GPCNetResNet):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetResNet18, self).__init__(
            backbone_name="swsl_resnet18",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetResNet50(GPCNetResNet):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetResNet50, self).__init__(
            backbone_name="swsl_resnet50",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetResNext50(GPCNetResNet):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetResNext50, self).__init__(
            backbone_name="resnext50_32x4d.fb_swsl_ig1b_ft_in1k",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetSwinTrans(GPCNet):
    def __init__(self, backbone_name, image_size: Tuple[int, int], *args, **kwargs):
        encoder = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=True,
            img_size=image_size,
        )
        encoder_channels = encoder.feature_info.channels()
        super(GPCNetSwinTrans, self).__init__(
            encoder=encoder,
            enc_chans=encoder_channels,
            image_size=image_size,
            num_grids=(1, 1),
            enable_intx=[False, False, False, False],
            *args,
            **kwargs,
        )

    def forward_encoder(self, x: torch.Tensor):
        feats = self.encoder(x)

        fixed_feats = []
        for feat, ch in zip(feats, self.semantic_decoder.feature_dims):
            if feat.ndim != 4:
                raise RuntimeError(f"Expected 4D feature map, got {feat.shape}")

            if feat.shape[1] != ch and feat.shape[-1] == ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            fixed_feats.append(feat)

        return fixed_feats

    def forward(self, x: torch.Tensor):
        feats = self.forward_encoder(x)
        return self.semantic_decoder(feats)

    def setup_hooks(self):
        pass


class GPCNetSwinT(GPCNetSwinTrans):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetSwinT, self).__init__(
            backbone_name="swin_tiny_patch4_window7_224.ms_in1k",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetPVTv2(GPCNet):
    def __init__(self, backbone_name, image_size: Tuple[int, int], *args, **kwargs):
        encoder = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=True,
        )
        encoder_channels = encoder.feature_info.channels()
        super(GPCNetPVTv2, self).__init__(
            encoder=encoder,
            enc_chans=encoder_channels,
            image_size=image_size,
            num_grids=(1, 1),
            enable_intx=[False, False, False, False],
            *args,
            **kwargs,
        )

    def forward_encoder(self, x: torch.Tensor):
        feats = self.encoder(x)

        fixed_feats = []
        for feat, ch in zip(feats, self.semantic_decoder.feature_dims):
            if feat.ndim != 4:
                raise RuntimeError(f"Expected 4D feature map, got {feat.shape}")

            if feat.shape[1] != ch and feat.shape[-1] == ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            fixed_feats.append(feat)

        return fixed_feats

    def forward(self, x: torch.Tensor):
        feats = self.forward_encoder(x)
        return self.semantic_decoder(feats)

    def setup_hooks(self):
        pass


class GPCNetPVTv2B2(GPCNetPVTv2):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetPVTv2B2, self).__init__(
            backbone_name="pvt_v2_b2.in1k",
            image_size=image_size,
            *args,
            **kwargs,
        )


class GPCNetTwinsSVTSmall(GPCNetPVTv2):
    def __init__(self, image_size: Tuple[int, int], *args, **kwargs):
        super(GPCNetTwinsSVTSmall, self).__init__(
            backbone_name="twins_svt_small.in1k",
            image_size=image_size,
            *args,
            **kwargs,
        )

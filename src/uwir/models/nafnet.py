"""
nafnet.py
---------
Nonlinear Activation Free Network (NAFNet) adapted for lightweight
underwater image restoration.

Reference:
    Chen et al. "Simple Baselines for Image Restoration" (ECCV 2022)
    https://arxiv.org/abs/2204.04663

Key innovations:
  1. Complete removal of traditional nonlinear activations (GELU, ReLU, Sigmoid).
  2. SimpleGate: splits feature channels and performs element-wise multiplication (x1 * x2).
  3. Simplified Channel Attention (SCA): global average pooling followed by 1x1 conv.
  4. LayerNorm2d: channels-first layer normalization for spatial feature maps.
  5. Residual learning: predicts RGB residual on top of degraded input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=(0, 2, 3)),
            grad_output.sum(dim=(0, 2, 3)),
            None,
        )


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm for (N, C, H, W) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Element-wise multiplication of split feature channels: x1 * x2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Channel attention using global pooling and a single 1x1 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.conv(self.pool(x))
        return x * scale


class NAFBlock(nn.Module):
    """
    Core Nonlinear Activation Free Block.

    Args:
        c (int): Number of feature channels.
        dw_expand (int): Expansion factor for depthwise convolution (default: 2).
        ffn_expand (int): Expansion factor for feed-forward network (default: 2).
        drop_out_rate (float): Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(c)
        self.conv3 = nn.Conv2d(c, c, 1)

        ffn_channels = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(c, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        # LayerScale parameters (initialized to zeros for exact residual identity)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = y + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


def _pad_to_multiple(inputs: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = inputs.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if not (pad_h or pad_w):
        return inputs, 0, 0
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(inputs, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w


class NAFNet(nn.Module):
    """
    Lightweight NAFNet U-Net architecture for underwater image restoration.

    Args:
        in_channels   (int): Input channels (3 for RGB, 4/5 for physics-augmented).
        out_channels  (int): Output channels (3 for RGB).
        width         (int): Base feature channels.
        middle_blk_num(int): Number of NAFBlocks at the bottleneck.
        enc_blk_nums  (tuple[int, ...]): Number of NAFBlocks at each encoder level.
        dec_blk_nums  (tuple[int, ...]): Number of NAFBlocks at each decoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 16,
        middle_blk_num: int = 2,
        enc_blk_nums: tuple[int, ...] = (1, 1, 1),
        dec_blk_nums: tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.outro = nn.Conv2d(width, out_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        # Encoders and downsampling
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan = chan * 2

        # Bottleneck
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # Decoders and upsampling
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        # Zero-initialize the outro head to start as identity residual
        nn.init.zeros_(self.outro.weight)
        nn.init.zeros_(self.outro.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        multiple = 2 ** len(self.downs)
        inputs, pad_h, pad_w = _pad_to_multiple(inputs, multiple=multiple)

        x = self.intro(inputs)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip  # Additive skip connection (standard NAFNet design)
            x = decoder(x)

        residual = self.outro(x)
        output = torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)

        if pad_h:
            output = output[..., :-pad_h, :]
        if pad_w:
            output = output[..., :, :-pad_w]
        return output


def build_nafnettiny(in_channels: int = 3) -> NAFNet:
    """
    Construct NAFNet-Tiny (~104k params):
    Width = 16, Stages = (16, 32, 64), Blocks = 1-1-2-1-1.
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=3,
        width=16,
        middle_blk_num=2,
        enc_blk_nums=(1, 1),
        dec_blk_nums=(1, 1),
    )


def build_nafnetmicro(in_channels: int = 3) -> NAFNet:
    """
    Construct NAFNet-Micro (~42k params):
    Width = 12, Stages = (12, 24, 48), Blocks = 1-1-1-1-1.
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=3,
        width=12,
        middle_blk_num=1,
        enc_blk_nums=(1, 1),
        dec_blk_nums=(1, 1),
    )


def build_nafnet(in_channels: int = 3) -> NAFNet:
    """
    Construct 4-level NAFNet (~288k params):
    Width = 16, Stages = (16, 32, 64, 128), Blocks = 1-1-1-1-1-1-1.
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1),
        dec_blk_nums=(1, 1, 1),
    )


# Backward-compatible alias
build_nafnet_tiny = build_nafnettiny
build_nafnet_micro = build_nafnetmicro


__all__ = [
    "LayerNorm2d",
    "NAFBlock",
    "NAFNet",
    "SimpleGate",
    "SimplifiedChannelAttention",
    "build_nafnet",
    "build_nafnet_micro",
    "build_nafnet_tiny",
    "build_nafnetmicro",
    "build_nafnettiny",
]


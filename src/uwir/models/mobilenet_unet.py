"""MobileNet-style U-Net built from inverted residual convolution blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvBlock(nn.Module):
    """Pointwise expansion, depthwise convolution, and pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 2):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels
        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(inplace=True),
            )
            if expand_ratio != 1
            else nn.Identity()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.project(self.depthwise(self.expand(x)))
        return x + output if self.use_residual else output


class MobileDoubleConv(nn.Module):
    """Two MobileNet-style inverted residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            MBConvBlock(in_channels, out_channels, expand_ratio),
            MBConvBlock(out_channels, out_channels, expand_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            MobileDoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, previous_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        upsampled_channels = previous_channels // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(previous_channels, upsampled_channels, kernel_size=1, bias=False),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(
                upsampled_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.conv = MobileDoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        height_difference = skip.size(2) - x.size(2)
        width_difference = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [
                width_difference // 2,
                width_difference - width_difference // 2,
                height_difference // 2,
                height_difference - height_difference // 2,
            ],
        )
        return self.conv(self.compress(torch.cat((skip, x), dim=1)))


class MobileNetUNet(nn.Module):
    """The 3.704M-parameter MBConv U-Net used by the MobileNet ablation."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 192, 256),
    ):
        super().__init__()
        f = features
        self.enc1 = MobileDoubleConv(in_channels, f[0])
        self.enc2 = _Down(f[0], f[1])
        self.enc3 = _Down(f[1], f[2])
        self.enc4 = _Down(f[2], f[3])
        self.bottleneck = _Down(f[3], f[3] * 2)
        self.dec4 = _Up(f[3] * 2, f[3], f[3])
        self.dec3 = _Up(f[3], f[2], f[2])
        self.dec2 = _Up(f[2], f[1], f[1])
        self.dec1 = _Up(f[1], f[0], f[0])
        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder1 = self.enc1(x)
        encoder2 = self.enc2(encoder1)
        encoder3 = self.enc3(encoder2)
        encoder4 = self.enc4(encoder3)
        bottleneck = self.bottleneck(encoder4)
        decoder4 = self.dec4(bottleneck, encoder4)
        decoder3 = self.dec3(decoder4, encoder3)
        decoder2 = self.dec2(decoder3, encoder2)
        decoder1 = self.dec1(decoder2, encoder1)
        return self.head(decoder1)

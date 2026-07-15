"""MBConv-based U-Net for underwater image restoration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 2):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = in_ch == out_ch

        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            )
            if expand_ratio != 1
            else nn.Identity()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_res:
            return x + out
        return out


class MobileDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            MBConvBlock(in_ch, out_ch, expand_ratio),
            MBConvBlock(out_ch, out_ch, expand_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            MobileDoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(prev_ch, prev_ch // 2, kernel_size=1, bias=False),
            )
            combined_ch = (prev_ch // 2) + skip_ch
        else:
            self.up = nn.ConvTranspose2d(prev_ch, prev_ch // 2, kernel_size=2, stride=2)
            combined_ch = (prev_ch // 2) + skip_ch

        self.compress = nn.Sequential(
            nn.Conv2d(combined_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

        self.conv = MobileDoubleConv(out_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        d_y = x2.size(2) - x1.size(2)
        d_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.compress(x)
        return self.conv(x)


class MBConvUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, int, int, int] = (64, 128, 192, 256),
        bilinear: bool = True,
    ):
        super().__init__()
        f = features

        self.enc1 = MobileDoubleConv(in_channels, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])

        self.bottleneck = Down(f[3], f[3] * 2)

        self.dec4 = Up(f[3] * 2, f[3], f[3], bilinear)
        self.dec3 = Up(f[3], f[2], f[2], bilinear)
        self.dec2 = Up(f[2], f[1], f[1], bilinear)
        self.dec1 = Up(f[1], f[0], f[0], bilinear)

        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)

        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)


# Historical name used by net/mbconv_unet_defined.py.
UNet5ch = MBConvUNet

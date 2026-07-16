"""U-Net++ with nested dense skip connections and deep supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv


class UNetPlusPlus5ch(nn.Module):
    """4-level U-Net++ for 3/4/5-channel underwater image restoration inputs."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 192, 256),
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        f0, f1, f2, f3 = features
        f4 = f3 * 2

        self.conv0_0 = DoubleConv(in_channels, f0)
        self.conv1_0 = DoubleConv(f0, f1)
        self.conv2_0 = DoubleConv(f1, f2)
        self.conv3_0 = DoubleConv(f2, f3)
        self.conv4_0 = DoubleConv(f3, f4)

        self.conv0_1 = DoubleConv(f0 + f1, f0)
        self.conv1_1 = DoubleConv(f1 + f2, f1)
        self.conv2_1 = DoubleConv(f2 + f3, f2)
        self.conv3_1 = DoubleConv(f3 + f4, f3)

        self.conv0_2 = DoubleConv(f0 * 2 + f1, f0)
        self.conv1_2 = DoubleConv(f1 * 2 + f2, f1)
        self.conv2_2 = DoubleConv(f2 * 2 + f3, f2)

        self.conv0_3 = DoubleConv(f0 * 3 + f1, f0)
        self.conv1_3 = DoubleConv(f1 * 3 + f2, f1)

        self.conv0_4 = DoubleConv(f0 * 4 + f1, f0)

        self.final1 = nn.Sequential(nn.Conv2d(f0, out_channels, kernel_size=1), nn.Sigmoid())
        self.final2 = nn.Sequential(nn.Conv2d(f0, out_channels, kernel_size=1), nn.Sigmoid())
        self.final3 = nn.Sequential(nn.Conv2d(f0, out_channels, kernel_size=1), nn.Sigmoid())
        self.final4 = nn.Sequential(nn.Conv2d(f0, out_channels, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1)
        )

        outputs = [
            self.final1(x0_1),
            self.final2(x0_2),
            self.final3(x0_3),
            self.final4(x0_4),
        ]
        if self.deep_supervision and self.training:
            return outputs
        return outputs[-1]


class UNetPlusPlusLarge5ch(UNetPlusPlus5ch):
    """U-Net++ variant with the original large channel schedule."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        deep_supervision: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=(64, 128, 256, 512),
            deep_supervision=deep_supervision,
        )

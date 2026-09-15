"""
PLite-Net: Physics-Guided LiteEnhanceNet
Inherits the ultra-lightweight backbone of LiteEnhanceNet (~15.2k parameters).
Integrates:
  1. Transmission map estimation head t(x)
  2. Background / water light estimation head B
  3. Forward Optical Degradation Re-synthesis (Koschmieder optical model):
     I_redeg = J * t + B * (1 - t)
"""

import torch
import torch.nn as nn
from .lite_enhancenet import ConvBlock1, ConvBlock2, ConvBlock3, ConvBlock4


class PLiteNet(nn.Module):
    """
    Physical-Guided LiteEnhanceNet (~15.2k parameters).
    During training: returns (J, I_redeg, t, B) if requested, or J by default.
    During inference: returns J directly with zero overhead.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.input = nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1 = ConvBlock1(16, 32)
        self.block2 = ConvBlock2(32, 64)
        self.block3 = ConvBlock3(64, 32)
        self.block4 = ConvBlock4(80, 32)
        
        # Head 1: Radiance residual delta J
        self.head_j = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Head 2: Transmission map t(x) (1-ch, in (0, 1])
        self.head_t = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Head 3: Background light B (3-ch scalar per image)
        self.head_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, x, return_physics=False):
        inp_rgb = x[:, :3, :, :] if x.shape[1] > 3 else x
        f0 = self.input(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f_cat = torch.cat([f0, f1, f3], dim=1)
        f4 = self.block4(f_cat)
        
        delta = self.head_j(f4)
        j = torch.clamp(inp_rgb + delta, 0.0, 1.0)
        
        if return_physics or self.training:
            t = torch.clamp(self.head_t(f4), 0.05, 1.0)
            b = self.head_b(f4).view(-1, 3, 1, 1)
            i_redeg = j * t + b * (1.0 - t)
            if return_physics:
                return j, i_redeg, t, b
        
        return j

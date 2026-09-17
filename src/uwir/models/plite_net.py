"""
PLite-Net: Physics-Guided LiteEnhanceNet with Optical Re-degradation Branch
Total Parameters: 18,444
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from .lite_enhancenet import LiteEnhanceNet


class PLiteNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PLiteNet, self).__init__()
        self.enhancer = LiteEnhanceNet(in_channels=in_channels)
        
        # Physical estimation branch for t(x) (1 channel) and B(x) (3 channels)
        self.phy_branch = nn.Sequential(
            nn.Conv2d(in_channels, 72, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 4, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x, return_physics=False):
        """
        Args:
            x: Input degraded underwater image [B, 3, H, W] in range [0, 1]
            return_physics: If True, always returns tuple (J, I_redeg, t, B)
        Returns:
            In training mode: (J, I_redeg, t, B)
            In eval mode (default): J
        """
        # 1. Enhanced image J
        j = self.enhancer(x)
        
        # 2. Physics priors: transmission t and background light B
        phy = torch.sigmoid(self.phy_branch(x))
        t = phy[:, 0:1, :, :]  # [B, 1, H, W]
        b = phy[:, 1:4, :, :]  # [B, 3, H, W]
        
        # 3. Optical re-degradation formula: I_redeg = J * t + B * (1 - t)
        redeg = torch.clamp(j * t + b * (1.0 - t), 0.0, 1.0)
        
        if self.training or return_physics:
            return j, redeg, t, b
        return j

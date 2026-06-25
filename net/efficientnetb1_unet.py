"""
efficientnetb1_unet.py
----------------------
EfficientNet-B1 encoder + U-Net decoder for underwater image restoration.

EfficientNet-B1 shares EfficientNet-B0's width multiplier (1.0); only the depth
multiplier differs (more MBConv sub-blocks per stage). The top-level feature
list still has 9 entries with identical per-stage channel widths, so the same
stage slicing and decoder configuration as ``EfficientNetB0UNet`` apply.

Architecture
------------
EfficientNet-B1 feature stages (stride-based skip points):

  stage0   features[0:2]    16 ch,  H/2
  stage1   features[2:3]    24 ch,  H/4
  stage2   features[3:4]    40 ch,  H/8
  stage3   features[4:6]   112 ch,  H/16
  stage4   features[6:9]  1280 ch,  H/32  ← bottleneck

Decoder:

  dec4  (1280, skip=112) → 256,  H/16
  dec3  ( 256,  skip=40) → 128,  H/8
  dec2  ( 128,  skip=24) →  64,  H/4
  dec1  (  64,  skip=16) →  32,  H/2
  dec0  (  32,  skip= 0) →  32,  H
  head  Conv1×1 → Sigmoid → (B, 3, H, W)

Extra channels for physics ablation
------------------------------------
  in_channels = 3  → RGB only
  in_channels = 4  → RGB + t(x)  OR  RGB + B
  in_channels = 5  → RGB + t(x) + B

The first Conv2d inside features[0] is re-initialised for in_channels ≠ 3.
"""

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

try:
    from net.blocks import DecoderBlock
except ModuleNotFoundError:                       # run directly: `python net/efficientnetb1_unet.py`
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from net.blocks import DecoderBlock


class EfficientNetB1UNet(nn.Module):
    """
    EfficientNet-B1 encoder + lightweight U-Net decoder.

    Args:
        in_channels  (int):  Input channels – 3, 4, or 5.
        out_channels (int):  Output channels. Default: 3.
        pretrained   (bool): Load ImageNet-pretrained EfficientNet-B1 weights.
                             Default: True.

    Example::

        model = EfficientNetB1UNet(in_channels=5)
        x = torch.randn(2, 5, 256, 256)
        y = model(x)   # (2, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels:  int  = 5,
        out_channels: int  = 3,
        pretrained:   bool = True,
    ):
        super().__init__()

        weights  = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b1(weights=weights)
        features = list(backbone.features)

        # ------------------------------------------------------------------
        # Patch first conv when in_channels ≠ 3
        # features[0] is Conv2dNormActivation (a Sequential);
        # features[0][0] is the raw Conv2d.
        # ------------------------------------------------------------------
        if in_channels != 3:
            orig_conv = features[0][0]
            new_conv  = nn.Conv2d(
                in_channels, orig_conv.out_channels,
                kernel_size = orig_conv.kernel_size,
                stride      = orig_conv.stride,
                padding     = orig_conv.padding,
                bias        = False,
            )
            if pretrained:
                with torch.no_grad():
                    n = min(in_channels, 3)
                    new_conv.weight[:, :n] = orig_conv.weight[:, :n]
            features[0][0] = new_conv

        # ------------------------------------------------------------------
        # Encoder stages
        # ------------------------------------------------------------------
        self.stage0 = nn.Sequential(*features[0:2])    # 16 ch,   H/2
        self.stage1 = nn.Sequential(*features[2:3])    # 24 ch,   H/4
        self.stage2 = nn.Sequential(*features[3:4])    # 40 ch,   H/8
        self.stage3 = nn.Sequential(*features[4:6])    # 112 ch,  H/16
        self.stage4 = nn.Sequential(*features[6:9])    # 1280 ch, H/32

        # ------------------------------------------------------------------
        # Decoder  (in_ch, skip_ch, out_ch)
        # ------------------------------------------------------------------
        self.dec4 = DecoderBlock(1280, 112, 256)
        self.dec3 = DecoderBlock(256,   40, 128)
        self.dec2 = DecoderBlock(128,   24,  64)
        self.dec1 = DecoderBlock( 64,   16,  32)
        self.dec0 = DecoderBlock( 32,    0,  32)   # no skip at full resolution

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (N, C_in, H, W)

        Returns:
            Tensor: (N, 3, H, W) restored RGB in [0, 1].
        """
        s0 = self.stage0(x)       # (N,   16, H/2,  W/2 )
        s1 = self.stage1(s0)      # (N,   24, H/4,  W/4 )
        s2 = self.stage2(s1)      # (N,   40, H/8,  W/8 )
        s3 = self.stage3(s2)      # (N,  112, H/16, W/16)
        s4 = self.stage4(s3)      # (N, 1280, H/32, W/32)

        d4 = self.dec4(s4, s3)    # (N, 256, H/16, W/16)
        d3 = self.dec3(d4, s2)    # (N, 128, H/8,  W/8 )
        d2 = self.dec2(d3, s1)    # (N,  64, H/4,  W/4 )
        d1 = self.dec1(d2, s0)    # (N,  32, H/2,  W/2 )
        d0 = self.dec0(d1)        # (N,  32, H,    W   )

        return self.head(d0)      # (N,   3, H,    W   )


# ---------------------------------------------------------------------------
# Smoke-test: `python net/efficientnetb1_unet.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print(f"  EfficientNetB1UNet smoke-test  –  device: {device}")
    print("=" * 55)

    for in_channels in (3, 4, 5):
        model = EfficientNetB1UNet(in_channels=in_channels, pretrained=False).to(device)
        model.eval()

        x = torch.randn(2, in_channels, 256, 256, device=device)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 3, 256, 256), f"Bad output shape: {tuple(y.shape)}"
        assert torch.all((y >= 0) & (y <= 1)), "Output not in [0, 1]"

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  [ok] in_channels={in_channels}  "
            f"{tuple(x.shape)} -> {tuple(y.shape)}  "
            f"params={n_params / 1e6:.2f}M"
        )

    print("\n" + "=" * 55)
    print("  All checks passed [ok]")
    print("=" * 55)

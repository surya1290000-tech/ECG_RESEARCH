"""
XResNet1D — 1D Bag-of-Tricks ResNet Architecture for ECG Classification
========================================================================

Implements the XResNet architectural improvements (He et al., 2019 "Bag of Tricks")
adapted to 1-dimensional 12-lead ECG signals:

1. ResNet-B (Stem Tweak):
   Replaces the monolithic large 7x1 conv stem with a 3-layer cascade of 1D convs
   (12 -> 32 -> 32 -> 64) with BatchNorm and ReLU, providing smoother gradient flow
   and better feature hierarchy from raw 12-lead waveforms.

2. ResNet-C (Downsampling Conv Placement):
   In downsampling blocks, stride is placed in the main convolutional filter
   rather than any preparatory 1x1 compression.

3. ResNet-D (Anti-Aliased Downsampling Shortcut):
   In downsampling residual shortcuts (stride > 1), uses AvgPool1d(kernel_size=2, stride=stride)
   BEFORE the 1x1 conv projection. This prevents information loss from subsampling
   by averaging adjacent signal samples prior to channel projection.

Input  : (batch, 12, 1000)  — 12-lead ECG, 10 s @ 100 Hz
Output : (batch, 5)          — Multi-label logits for NORM, STTC, CD, MI, HYP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# Standard class mapping
CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class ConvStem1D(nn.Module):
    """
    ResNet-B 3-stage 1D convolutional stem.
    Replaces a single 7x1 conv with three consecutive 5x1/3x1 convs.
    """
    def __init__(self, in_channels: int = 12, out_channels: int = 64):
        super().__init__()
        mid_channels = out_channels // 2  # 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class XResBlock1D(nn.Module):
    """
    1-D XResNet Residual Block implementing ResNet-D shortcut tweak.
    
    When downsampling (stride > 1), an AvgPool1d is applied before the 1x1 conv
    projection to avoid aliasing and preserve temporal context.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()
        padding = kernel_size // 2

        # Main branch
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # ResNet-D Shortcut branch
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = []
            if stride > 1:
                # Average pooling before projection (ResNet-D)
                shortcut_layers.append(
                    nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)
                )
            shortcut_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels),
            ])
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class XResNet1D(nn.Module):
    """
    XResNet1D Architecture for 12-lead ECG Multi-Label Classification.

    Backbone features:
      - 3-stage ConvStem (ResNet-B)
      - 4 Residual Stages with ResNet-D downsampling shortcuts
      - Global Average Pooling
      - 2-layer MLP Classifier with Dropout(0.3)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 12,
        stem_channels: int = 64,
        stage_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if stage_channels is None:
            stage_channels = [64, 128, 256, 512]

        self.stem = ConvStem1D(in_channels=in_channels, out_channels=stem_channels)

        # 4 Residual Blocks (matching Model A depth for direct architectural comparison)
        self.block1 = XResBlock1D(stage_channels[0], stage_channels[0], stride=1)
        self.block2 = XResBlock1D(stage_channels[0], stage_channels[1], stride=2)
        self.block3 = XResBlock1D(stage_channels[1], stage_channels[2], stride=2)
        self.block4 = XResBlock1D(stage_channels[2], stage_channels[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(stage_channels[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : (batch, 12, 1000)
        Output: (batch, num_classes)
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

    def get_feature_extractor(self) -> nn.Sequential:
        return nn.Sequential(
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.pool,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_xresnet1d(num_classes: int = NUM_CLASSES) -> XResNet1D:
    return XResNet1D(num_classes=num_classes)


if __name__ == "__main__":
    model = build_xresnet1d()
    x = torch.randn(2, 12, 1000)
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters  : {count_parameters(model):,}")

"""
InceptionTime1D — Multi-Scale Temporal Convolutional Architecture for 12-Lead ECG
==================================================================================

Implements 1D Inception network (Ismail Fawaz et al., 2020) adapted for 12-lead ECG
multi-label diagnostic classification on the official PTB-XL benchmark:

Architecture Overview:
  - Input: (batch, 12, 1000) — 12-lead ECG, 10 s @ 100 Hz
  - 6 Stacked Inception Blocks with multi-scale receptive fields:
      Branch A: kernel = 9   (90 ms temporal span at 100 Hz)
      Branch B: kernel = 19  (190 ms temporal span at 100 Hz)
      Branch C: kernel = 39  (390 ms temporal span at 100 Hz)
      Branch D: MaxPool (3x1) + 1x1 Conv projection
  - Concatenation of all 4 branches
  - BatchNorm1d
  - Residual shortcut connection + ReLU
  - Hierarchical temporal downsampling (strides [2, 2, 2, 1, 1, 1])
  - Global Average Pooling (AdaptiveAvgPool1d(1)) -> 512 features
  - Classification Head:
      Linear(512 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 5)

Target Parameter Budget: 3.8M–4.3M parameters.
Exact Parameter Count  : 3,886,149 parameters (Model A: 3,919,493; Delta = -0.85%).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# Diagnostic Superclasses
CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class InceptionBlock1D(nn.Module):
    """
    Single 1D Inception Block with bottleneck projection, 3 multi-scale
    temporal convolution branches (k=9, 19, 39), a max-pooling branch,
    concatenation, batch normalization, and a residual shortcut connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int = 64,
        kernel_sizes: Optional[List[int]] = None,
        stride: int = 1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]

        branch_channels = out_channels // 4  # Equal division among 4 branches

        # 1. Bottleneck 1x1 Conv projection
        self.bottleneck = nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )

        # 2. Parallel multi-scale Conv1D branches
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                bottleneck_channels,
                branch_channels,
                kernel_size=k,
                stride=stride,
                padding=k // 2,
                bias=False,
            )
            for k in kernel_sizes
        ])

        # 3. MaxPool branch with 1x1 Conv projection
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.pool_conv = nn.Conv1d(
            in_channels,
            branch_channels,
            kernel_size=1,
            bias=False,
        )

        # 4. BatchNorm and Activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 5. Residual connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)

        # Bottleneck projection
        b = self.bottleneck(x)

        # Convolve 3 parallel branches
        branch_outs = [conv(b) for conv in self.conv_branches]

        # Maxpool branch
        pool_out = self.pool_conv(self.maxpool(x))
        branch_outs.append(pool_out)

        # Concatenate 4 branches along channel axis
        out = torch.cat(branch_outs, dim=1)
        out = self.bn(out)

        # Residual addition followed by non-linear activation
        out = self.relu(out + res)
        return out


class InceptionTime1D(nn.Module):
    """
    6-Block InceptionTime1D Architecture with 3.89M parameters, matching
    the parameter budget of Model A (ECGResNet GAP, 3.92M parameters).
    """
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 6 Inception Blocks with progressive channels and temporal downsampling:
        # (in_channels, out_channels, bottleneck_channels, stride)
        # Block 1: 12 -> 256, bottle=48, stride=2 (L: 1000 -> 500)
        # Block 2: 256 -> 256, bottle=48, stride=2 (L: 500 -> 250)
        # Block 3: 256 -> 384, bottle=76, stride=2 (L: 250 -> 125)
        # Block 4: 384 -> 384, bottle=76, stride=1 (L: 125 -> 125)
        # Block 5: 384 -> 512, bottle=100, stride=1 (L: 125 -> 125)
        # Block 6: 512 -> 512, bottle=100, stride=1 (L: 125 -> 125)
        block_configs = [
            (in_channels, 256, 48, 2),
            (256, 256, 48, 2),
            (256, 384, 76, 2),
            (384, 384, 76, 1),
            (384, 512, 100, 1),
            (512, 512, 100, 1),
        ]

        blocks = []
        for c_in, c_out, c_bottle, s in block_configs:
            blocks.append(
                InceptionBlock1D(
                    in_channels=c_in,
                    out_channels=c_out,
                    bottleneck_channels=c_bottle,
                    kernel_sizes=[9, 19, 39],
                    stride=s,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head: Linear(512 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 5)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input  : (batch, 12, 1000)
        Output : (batch, 5) — unnormalized logits
        """
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_inceptiontime1d(num_classes: int = NUM_CLASSES) -> InceptionTime1D:
    return InceptionTime1D(num_classes=num_classes)


if __name__ == "__main__":
    m = build_inceptiontime1d()
    p = count_parameters(m)
    x = torch.randn(2, 12, 1000)
    out = m(x)
    print(f"InceptionTime1D Total Parameters: {p:,}")
    print(f"Forward Output Shape: {out.shape}")

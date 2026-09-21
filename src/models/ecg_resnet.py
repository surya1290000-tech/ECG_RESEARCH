"""
ECGResNet — Canonical Architecture (Version B)
===============================================

Source: arth.ipynb, execution_count 15 (lines 8415–8549).
This is the architecture that was used to produce checkpoints/best_ecg_model.pth.
It was verified via strict=True load with 0 missing / 0 unexpected keys.

DO NOT modify this file without updating checkpoints/ accordingly,
as any structural change will break checkpoint compatibility.

Architecture summary
--------------------
Input  : (batch, 12, 1000)   — 12-lead ECG, 10 s @ 100 Hz
Output : (batch, 5)           — logits for 5 diagnostic superclasses

Confirmed checkpoint key shapes (from Colab execution_count 20):
    stem.0.weight          → (64, 12, 7)
    block1.conv1.weight    → (64, 64, 7)
    block2.conv1.weight    → (128, 64, 7)
    block2.shortcut.0.weight → (128, 64, 1)
    block3.conv1.weight    → (256, 128, 7)
    block3.shortcut.0.weight → (256, 128, 1)
    block4.conv1.weight    → (512, 256, 7)
    block4.shortcut.0.weight → (512, 256, 1)
    classifier.0.weight    → (128, 512)
    classifier.3.weight    → (5, 128)

Class ordering used during training (MUST NOT change for checkpoint compatibility):
    CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
    CLASS_TO_ID = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Class label constants — must match the training-time mapping
# ---------------------------------------------------------------------------

CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]

CLASS_TO_ID = {
    "NORM": 0,
    "STTC": 1,
    "CD": 2,
    "MI": 3,
    "HYP": 4,
}

ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

NUM_CLASSES = len(CLASS_NAMES)  # 5


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    1-D Residual block for ECG signals.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int  — applied to conv1 and the shortcut projection.
                          stride=2 halves the time dimension.

    When in_channels != out_channels or stride != 1, a projection shortcut
    (1x1 conv + BN) is used.  Otherwise the shortcut is an identity
    (empty Sequential, adds no parameters).

    Note: bias=False on all convolutions because BatchNorm subsumes the bias.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut projection (only when shape changes)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            # Identity shortcut — no parameters added
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = F.relu(x)

        return x


# ---------------------------------------------------------------------------
# ECGResNet
# ---------------------------------------------------------------------------

class ECGResNet(nn.Module):
    """
    ECGResNet — 12-lead ECG classifier.

    Parameters
    ----------
    num_classes : int  — default 5 (NORM, MI, STTC, CD, HYP)

    Data flow
    ---------
    Input  : (B, 12, 1000)
    stem   : (B, 64, 1000)   — Conv7 stride=1, BN, ReLU
    block1 : (B, 64, 1000)   — stride=1, no shape change
    block2 : (B, 128, 500)   — stride=2
    block3 : (B, 256, 250)   — stride=2
    block4 : (B, 512, 125)   — stride=2
    pool   : (B, 512, 1)     — AdaptiveAvgPool1d
    flatten: (B, 512)
    head   : (B, 5)          — Linear->ReLU->Dropout->Linear
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Stem: initial feature extraction, no downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(
                12,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Residual blocks — each doubles channels, blocks 2-4 halve time dim
        self.block1 = ResidualBlock(64, 64, stride=1)
        self.block2 = ResidualBlock(64, 128, stride=2)
        self.block3 = ResidualBlock(128, 256, stride=2)
        self.block4 = ResidualBlock(256, 512, stride=2)

        # Global average pooling -> fixed-size vector regardless of input length
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, 12, 1000)

        Returns
        -------
        torch.Tensor  shape (batch, num_classes)  — raw logits
        """
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)        # (batch, 512, 1)
        x = x.squeeze(-1)       # (batch, 512)

        x = self.classifier(x)  # (batch, num_classes)

        return x

    def get_feature_extractor(self) -> nn.Sequential:
        """
        Return the feature extraction part (everything before the classifier head).
        Useful for representation analysis.
        """
        return nn.Sequential(
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.pool,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_classes: int = NUM_CLASSES) -> ECGResNet:
    """Convenience factory — returns an ECGResNet instance in eval mode."""
    model = ECGResNet(num_classes=num_classes)
    return model

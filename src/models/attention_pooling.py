"""
ECGResNetAttention — Model B with Learned Temporal Attention Pooling
====================================================================

Replaces AdaptiveAvgPool1d(1) in ECGResNet (Version B) with a minimal,
learnable 1D temporal attention pooling module.

Architecture Overview:
  Input  : (batch, 12, 1000)
  stem   : (batch, 64, 1000)
  block1 : (batch, 64, 1000)
  block2 : (batch, 128, 500)
  block3 : (batch, 256, 250)
  block4 : (batch, 512, 125)
  pool   : (batch, 512)       — TemporalAttentionPooling
  classifier : (batch, 5)     — Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 5)

Temporal Attention Mechanism:
  1. Conv1d(512 -> 1, kernel_size=1) computes raw attention logits per temporal frame t in [0..124].
  2. Softmax across temporal dimension normalizes weights sum(alpha_t) = 1.
  3. Weighted aggregation: z = sum_t (alpha_t * h_t).
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ecg_resnet import (
    ResidualBlock,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
)


class TemporalAttentionPooling(nn.Module):
    """
    Minimal Learnable 1D Temporal Attention Pooling Module.

    Parameters
    ----------
    in_channels : int
        Number of feature channels per temporal frame (default 512).
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()
        # 1x1 convolution mapping 512 channels to 1 scalar attention logit per frame
        self.attn_conv = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape (batch_size, in_channels, length)
            e.g., (B, 512, 125).

        return_attn_weights : bool
            If True, also returns normalized attention weights of shape (batch_size, length).

        Returns
        -------
        pooled : torch.Tensor
            Aggregated representation of shape (batch_size, in_channels) e.g., (B, 512).
        weights : torch.Tensor (optional)
            Normalized attention weights of shape (batch_size, length) e.g., (B, 125).
        """
        # 1. Scalar logit per temporal position: (B, 1, L)
        logits = self.attn_conv(x)

        # 2. Softmax normalization over temporal dimension L: (B, 1, L)
        weights = F.softmax(logits, dim=-1)

        # 3. Weighted aggregation along temporal dimension L: (B, C)
        pooled = torch.sum(x * weights, dim=-1)

        if return_attn_weights:
            return pooled, weights.squeeze(1)
        return pooled


class ECGResNetAttention(nn.Module):
    """
    ECGResNet with Learned Temporal Attention Pooling (Model B).

    Identical to ECGResNet Version B in stem, blocks 1-4, and classifier head,
    differing ONLY in replacing AdaptiveAvgPool1d(1) with TemporalAttentionPooling.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Stem: 12 leads -> 64 channels, kernel size 7
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

        # Residual blocks
        self.block1 = ResidualBlock(64, 64, stride=1)
        self.block2 = ResidualBlock(64, 128, stride=2)
        self.block3 = ResidualBlock(128, 256, stride=2)
        self.block4 = ResidualBlock(256, 512, stride=2)

        # Model B intervention: Learned Temporal Attention Pooling
        self.pool = TemporalAttentionPooling(in_channels=512)

        # Classification head (identical to Model A)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, 12, 1000)
        return_attn : bool, if True returns (logits, attn_weights)

        Returns
        -------
        logits : torch.Tensor of shape (batch, num_classes)
        attn_weights : torch.Tensor of shape (batch, 125) [optional]
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if return_attn:
            pooled, weights = self.pool(x, return_attn_weights=True)
            logits = self.classifier(pooled)
            return logits, weights

        pooled = self.pool(x)
        logits = self.classifier(pooled)
        return logits


def build_attention_model(num_classes: int = NUM_CLASSES) -> ECGResNetAttention:
    """Factory helper — instantiates ECGResNetAttention."""
    return ECGResNetAttention(num_classes=num_classes)

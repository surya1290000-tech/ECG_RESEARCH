"""
Phase 5 Loss Functions for Class Imbalance in 12-Lead ECG Classification
========================================================================

Implements three loss formulations designed to handle acute superclass imbalance:
  1. Inverse Class Frequency Weighted Cross-Entropy (Weighted-CE)
  2. Multi-Class Focal Loss (Focal-Loss, Lin et al.)
  3. Class-Balanced Effective Sample Loss (CB-Loss, Cui et al., CVPR 2019)
"""

from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Inverse Class Frequency Weighting
# =============================================================================

def compute_inverse_frequency_weights(class_counts: Union[List[int], np.ndarray]) -> torch.Tensor:
    """
    Compute inverse frequency weights:
        w_i = N / (K * n_i)
    normalized such that the mean weight is 1.0.
    """
    counts = np.array(class_counts, dtype=np.float32)
    total_samples = np.sum(counts)
    num_classes = len(counts)
    
    weights = total_samples / (num_classes * counts)
    # Normalize so sum(weights) == num_classes
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_cross_entropy(class_counts: Union[List[int], np.ndarray], device: torch.device) -> nn.CrossEntropyLoss:
    """Build nn.CrossEntropyLoss initialized with normalized inverse frequency weights."""
    weights = compute_inverse_frequency_weights(class_counts).to(device)
    return nn.CrossEntropyLoss(weight=weights)


# =============================================================================
# 2. Multi-Class Focal Loss
# =============================================================================

class FocalLoss1D(nn.Module):
    """
    Multi-Class Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    gamma : float
        Focusing parameter that down-weights easy examples (default 2.0).
    alpha : Optional[torch.Tensor]
        Class weighting factor of shape (num_classes,). If None, uniform weighting is used.
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        targets: (B,) containing integer class indices in [0..C-1]
        """
        log_probs = F.log_softmax(logits, dim=-1)           # (B, C)
        probs = torch.exp(log_probs)                       # (B, C)

        # Gather target probabilities: (B, 1)
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(dim=1, index=targets).squeeze(-1)  # (B,)
        pt = probs.gather(dim=1, index=targets).squeeze(-1)          # (B,)

        focal_weight = torch.pow(1.0 - pt, self.gamma)      # (B,)

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(dim=0, index=targets.squeeze(-1)) # (B,)
            loss = -at * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# 3. Class-Balanced Loss (Cui et al., CVPR 2019)
# =============================================================================

def compute_class_balanced_weights(
    class_counts: Union[List[int], np.ndarray], beta: float = 0.9999
) -> torch.Tensor:
    """
    Compute Class-Balanced weights:
        E_n = (1 - beta^n) / (1 - beta)
        w_i = 1 / E_n = (1 - beta) / (1 - beta^n_i)
    normalized such that mean(w) == 1.0.
    """
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al., CVPR 2019) wrapping CrossEntropy with effective-sample weights.
    """

    def __init__(
        self,
        class_counts: Union[List[int], np.ndarray],
        beta: float = 0.9999,
        loss_type: str = "cross_entropy",
        gamma: float = 2.0,
    ):
        super().__init__()
        self.weights = compute_class_balanced_weights(class_counts, beta=beta)
        self.loss_type = loss_type
        self.gamma = gamma

        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        elif loss_type == "focal":
            self.criterion = FocalLoss1D(gamma=gamma, alpha=self.weights)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
                self.criterion.weight = self.weights
            elif hasattr(self.criterion, "alpha") and self.criterion.alpha is not None:
                self.criterion.alpha = self.weights

        return self.criterion(logits, targets)

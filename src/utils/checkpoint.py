"""
Checkpoint loading utility.
===========================

Handles loading best_ecg_model.pth — which is a raw state_dict (OrderedDict).
No optimizer state, no epoch metadata.

Usage
-----
    from src.utils.checkpoint import load_checkpoint
    model = load_checkpoint("checkpoints/best_ecg_model.pth")
"""

import collections
import os
from pathlib import Path
from typing import Optional

import torch

from src.models.ecg_resnet import ECGResNet, NUM_CLASSES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> ECGResNet:
    """
    Instantiate ECGResNet (Version B) and load weights from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the .pth file.  Must be a raw state_dict (OrderedDict).
    num_classes : int
        Must match the training-time value (default 5).
    device : torch.device or None
        Target device.  Defaults to CPU.
    strict : bool
        Passed directly to load_state_dict.  Must be True for verified
        checkpoint compatibility.

    Returns
    -------
    ECGResNet
        Model in evaluation mode on the specified device.

    Raises
    ------
    FileNotFoundError
        If checkpoint_path does not exist.
    RuntimeError
        If load_state_dict fails (missing / unexpected keys when strict=True).
    ValueError
        If the checkpoint format is unexpected.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}\n"
            "Expected: checkpoints/best_ecg_model.pth"
        )

    if device is None:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Load raw file
    # ------------------------------------------------------------------
    raw = torch.load(str(path), map_location=device)

    # ------------------------------------------------------------------
    # Resolve state_dict
    # The checkpoint is a raw OrderedDict (state_dict only).
    # Guard against wrapped formats just in case.
    # ------------------------------------------------------------------
    if isinstance(raw, collections.OrderedDict):
        state_dict = raw
    elif isinstance(raw, dict):
        if "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        else:
            # Assume it's a flat dict acting as state_dict
            state_dict = raw
    else:
        raise ValueError(
            f"Unexpected checkpoint type: {type(raw)}. "
            "Expected collections.OrderedDict or dict."
        )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = ECGResNet(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    incompatible = model.load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_missing = len(incompatible.missing_keys)
    n_unexpected = len(incompatible.unexpected_keys)

    print("=" * 60)
    print("CHECKPOINT LOAD REPORT")
    print("=" * 60)
    print(f"  Path           : {path.resolve()}")
    print(f"  File size      : {path.stat().st_size / 1e6:.2f} MB")
    print(f"  Device         : {device}")
    print(f"  strict         : {strict}")
    print(f"  Missing keys   : {n_missing}")
    print(f"  Unexpected keys: {n_unexpected}")
    print(f"  State dict keys: {len(state_dict)}")
    print("=" * 60)

    if n_missing > 0:
        print("  MISSING KEYS:")
        for k in incompatible.missing_keys:
            print(f"    - {k}")

    if n_unexpected > 0:
        print("  UNEXPECTED KEYS:")
        for k in incompatible.unexpected_keys:
            print(f"    - {k}")

    # Set to evaluation mode
    model.eval()
    model = model.to(device)

    return model


def verify_checkpoint_shapes(checkpoint_path: str | Path) -> dict:
    """
    Load raw state_dict and print every key with its tensor shape.
    Does not instantiate the model — useful for low-level inspection.

    Returns
    -------
    dict mapping key -> tuple(shape)
    """
    path = Path(checkpoint_path)
    raw = torch.load(str(path), map_location="cpu")

    if isinstance(raw, (dict, collections.OrderedDict)):
        state_dict = raw.get("state_dict", raw.get("model_state_dict", raw))
    else:
        raise ValueError(f"Cannot parse checkpoint of type {type(raw)}")

    shapes = {}
    print("=" * 60)
    print("CHECKPOINT KEY SHAPES")
    print("=" * 60)
    for key, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        shapes[key] = shape
        print(f"  {key:<45s} {shape}")
    print(f"\n  Total keys: {len(shapes)}")
    print("=" * 60)

    return shapes

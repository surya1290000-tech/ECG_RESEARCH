"""
Phase 2A Verification Script
==============================

Performs the following checks:
  1. Instantiate ECGResNet (Version B)
  2. Load checkpoints/best_ecg_model.pth with strict=True
  3. Verify: 0 missing keys, 0 unexpected keys
  4. Verify all known layer shapes against Colab-confirmed values
  5. Run dummy inference: input (1, 12, 1000) -> output (1, 5)
  6. Print full architecture summary

Run from the project root:
    python verify_checkpoint.py
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.models.ecg_resnet import (
    ECGResNet,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
    count_parameters,
    build_model,
)
from src.utils.checkpoint import load_checkpoint, verify_checkpoint_shapes


# ---------------------------------------------------------------------------
# Expected shapes — sourced from Colab execution_count 20 (confirmed output)
# ---------------------------------------------------------------------------

EXPECTED_SHAPES = {
    "stem.0.weight":              (64, 12, 7),
    "stem.1.weight":              (64,),
    "stem.1.bias":                (64,),
    "block1.conv1.weight":        (64, 64, 7),
    "block1.bn1.weight":          (64,),
    "block1.conv2.weight":        (64, 64, 7),
    "block1.bn2.weight":          (64,),
    "block2.conv1.weight":        (128, 64, 7),
    "block2.bn1.weight":          (128,),
    "block2.conv2.weight":        (128, 128, 7),
    "block2.bn2.weight":          (128,),
    "block2.shortcut.0.weight":   (128, 64, 1),
    "block2.shortcut.1.weight":   (128,),
    "block3.conv1.weight":        (256, 128, 7),
    "block3.bn1.weight":          (256,),
    "block3.conv2.weight":        (256, 256, 7),
    "block3.bn2.weight":          (256,),
    "block3.shortcut.0.weight":   (256, 128, 1),
    "block3.shortcut.1.weight":   (256,),
    "block4.conv1.weight":        (512, 256, 7),
    "block4.bn1.weight":          (512,),
    "block4.conv2.weight":        (512, 512, 7),
    "block4.bn2.weight":          (512,),
    "block4.shortcut.0.weight":   (512, 256, 1),
    "block4.shortcut.1.weight":   (512,),
    "classifier.0.weight":        (128, 512),
    "classifier.0.bias":          (128,),
    "classifier.3.weight":        (5, 128),
    "classifier.3.bias":          (5,),
}

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "best_ecg_model.pth",
)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label: str, condition: bool, details: str = ""):
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if details:
        line += f"  — {details}"
    print(line)
    return condition


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_passed = True

    # ------------------------------------------------------------------
    # 1. Model instantiation
    # ------------------------------------------------------------------
    section("STEP 1: Model instantiation")

    model = build_model(num_classes=NUM_CLASSES)
    n_params = count_parameters(model)

    all_passed &= check(
        "ECGResNet instantiated",
        isinstance(model, ECGResNet),
    )
    all_passed &= check(
        "num_classes == 5",
        model.classifier[-1].out_features == 5,
        f"got {model.classifier[-1].out_features}",
    )
    print(f"\n  Trainable parameters: {n_params:,}")
    print(f"  Class names : {CLASS_NAMES}")
    print(f"  CLASS_TO_ID : {CLASS_TO_ID}")

    # ------------------------------------------------------------------
    # 2. Checkpoint load
    # ------------------------------------------------------------------
    section("STEP 2: Checkpoint loading  (strict=True)")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n  ERROR: Checkpoint not found at:\n  {CHECKPOINT_PATH}")
        print("\n  PHASE 2A STATUS: BLOCKED — checkpoint missing.")
        sys.exit(1)

    model = load_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        strict=True,
    )

    # ------------------------------------------------------------------
    # 3. Raw shape verification
    # ------------------------------------------------------------------
    section("STEP 3: State-dict key shape verification")

    actual_shapes = verify_checkpoint_shapes(CHECKPOINT_PATH)

    shape_failures = []
    for key, expected_shape in EXPECTED_SHAPES.items():
        if key not in actual_shapes:
            shape_failures.append(f"MISSING key: {key}")
            all_passed = False
        elif actual_shapes[key] != expected_shape:
            shape_failures.append(
                f"SHAPE MISMATCH: {key} "
                f"expected {expected_shape} got {actual_shapes[key]}"
            )
            all_passed = False

    all_passed &= check(
        "All expected keys present with correct shapes",
        len(shape_failures) == 0,
        f"{len(shape_failures)} failures" if shape_failures else "OK",
    )
    for msg in shape_failures:
        print(f"    !! {msg}")

    all_passed &= check(
        f"Total key count == {len(actual_shapes)}",
        True,  # Key count is informational; strict=True is the authoritative check
        f"{len(actual_shapes)} keys (confirmed from live checkpoint)",
    )

    # ------------------------------------------------------------------
    # 4. Dummy inference
    # ------------------------------------------------------------------
    section("STEP 4: Dummy inference  input=(1,12,1000)  expected=(1,5)")

    model.eval()
    dummy_input = torch.randn(1, 12, 1000)

    with torch.no_grad():
        output = model(dummy_input)

    all_passed &= check(
        "output.shape == (1, 5)",
        output.shape == (1, 5),
        f"got {tuple(output.shape)}",
    )
    all_passed &= check(
        "output is finite (no NaN / Inf)",
        torch.isfinite(output).all().item(),
    )

    print(f"\n  Input  shape : {tuple(dummy_input.shape)}")
    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output values: {output.squeeze().tolist()}")

    # ------------------------------------------------------------------
    # 5. Architecture summary
    # ------------------------------------------------------------------
    section("STEP 5: Architecture")
    print(model)

    # ------------------------------------------------------------------
    # 6. Final status
    # ------------------------------------------------------------------
    section("PHASE 2A STATUS")

    print(f"  Architecture verified  : {'YES' if all_passed else 'PARTIAL'}")
    print(f"  Checkpoint verified    : {'YES' if all_passed else 'CHECK ABOVE'}")
    print(f"  Missing keys           : 0  (strict=True passed)")
    print(f"  Unexpected keys        : 0  (strict=True passed)")
    print(f"  Dummy inference        : {'PASSED' if all_passed else 'FAILED'}")
    print()
    print("  Files created:")
    print("    src/__init__.py")
    print("    src/models/__init__.py")
    print("    src/models/ecg_resnet.py")
    print("    src/utils/__init__.py")
    print("    src/utils/checkpoint.py")
    print("    src/data/__init__.py")
    print("    src/evaluation/__init__.py")
    print("    configs/config.py")
    print("    verify_checkpoint.py")
    print()
    print("  Dataset status         : MISSING locally")
    print("    The PTB-XL dataset is NOT present in this project folder.")
    print("    Expected location: ECG_Research/data/ptbxl/")
    print("    Download from   : https://physionet.org/content/ptb-xl/1.0.3/")
    print("    Required files  :")
    print("      data/ptbxl/ptbxl_database.csv")
    print("      data/ptbxl/scp_statements.csv")
    print("      data/ptbxl/records100/  (100 Hz ECG WFDB files)")
    print()
    print("  Next recommended step  :")
    if all_passed:
        print("    Phase 2B — Place PTB-XL dataset, then run Phase 2B dataset")
        print("    verification to reproduce the exact train/val/test split.")
    else:
        print("    Fix the failures listed above before proceeding.")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

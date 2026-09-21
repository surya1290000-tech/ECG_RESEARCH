"""
Baseline Evaluation Runner.
===========================

Evaluates checkpoints/best_ecg_model.pth on the PTB-XL test set.
Computes:
  - Test Loss (CrossEntropyLoss)
  - Accuracy
  - Weighted Precision, Recall, F1
  - Macro Precision, Recall, F1
  - Per-class metrics (Precision, Recall, F1, Support)
  - Confusion Matrix
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.models.ecg_resnet import (
    ECGResNet,
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
)
from src.utils.checkpoint import load_checkpoint
from src.data.dataset import (
    load_ptbxl_metadata,
    create_patient_level_splits,
    PTBXLECGDataset,
)
from configs.config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    BATCH_SIZE,
)


def evaluate_baseline(
    checkpoint_path: Path = CHECKPOINT_PATH,
    data_dir: Path = DATA_DIR,
    device: torch.device = None,
) -> dict:
    """
    Run clean baseline evaluation of the saved model on the test split.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHASE 2B: BASELINE EVALUATION")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Checkpoint path : {checkpoint_path}")
    print(f"Data directory  : {data_dir}")

    # 1. Load Model
    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=NUM_CLASSES,
        device=device,
        strict=True,
    )
    model.eval()

    # 2. Load Metadata and Split
    df, scp_df = load_ptbxl_metadata(data_dir=data_dir)
    train_df, val_df, test_df = create_patient_level_splits(df, random_state=42)

    print("\n--- SPLIT SUMMARY & PATIENT VERIFICATION ---")
    print(f"Train records     : {len(train_df)} | Unique patients: {train_df['patient_id'].nunique()}")
    print(f"Validation records: {len(val_df)} | Unique patients: {val_df['patient_id'].nunique()}")
    print(f"Test records      : {len(test_df)} | Unique patients: {test_df['patient_id'].nunique()}")

    train_p = set(train_df["patient_id"])
    val_p = set(val_df["patient_id"])
    test_p = set(test_df["patient_id"])
    assert len(train_p & val_p) == 0, "Patient leakage train-val"
    assert len(train_p & test_p) == 0, "Patient leakage train-test"
    assert len(val_p & test_p) == 0, "Patient leakage val-test"
    print("[PASS] Zero patient leakage verified across all splits!")

    # 3. Create Test DataLoader
    test_dataset = PTBXLECGDataset(test_df, data_dir=data_dir, apply_preprocessing=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Evaluation Loop
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(ecgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_loss = total_loss / len(test_dataset)

    # 5. Metrics Computation
    acc = accuracy_score(all_targets, all_preds)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    weighted_prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    weighted_rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 70)
    print("BASELINE EVALUATION RESULTS (TEST SET)")
    print("=" * 70)
    print(f"Test Samples    : {len(test_dataset)}")
    print(f"Test Loss       : {avg_loss:.4f}")
    print(f"Test Accuracy   : {acc * 100:.2f}% ({acc:.4f})")
    print(f"Macro Precision : {macro_prec:.4f}")
    print(f"Macro Recall    : {macro_rec:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_prec:.4f}")
    print(f"Weighted Recall   : {weighted_rec:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    print("\n--- PER-CLASS METRICS ---")
    for cls_name in CLASS_NAMES:
        cls_metrics = report[cls_name]
        print(
            f"  {cls_name:<6s} | Precision: {cls_metrics['precision']:.4f} | "
            f"Recall: {cls_metrics['recall']:.4f} | F1: {cls_metrics['f1-score']:.4f} | "
            f"Support: {int(cls_metrics['support'])}"
        )

    print("\n--- CONFUSION MATRIX ---")
    header_title = "True \\ Pred"
    print(f"{header_title:<12s} " + " ".join([f"{name:>7s}" for name in CLASS_NAMES]))
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = " ".join([f"{cm[i, j]:>7d}" for j in range(NUM_CLASSES)])
        print(f"{row_name:<12s} {row_str}")

    results = {
        "test_loss": avg_loss,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "per_class": {cls: report[cls] for cls in CLASS_NAMES},
        "confusion_matrix": cm,
        "num_test_samples": len(test_dataset),
    }

    return results


if __name__ == "__main__":
    evaluate_baseline()

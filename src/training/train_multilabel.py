"""
Phase 6 — Multi-Label PTB-XL Training Pipeline (5 Diagnostic Superclasses)
========================================================================

Trains and compares:
  - Model A-ML: ECGResNet (Global Average Pooling) + BCEWithLogitsLoss
  - Model B-ML: ECGResNetAttention (Temporal Attention Pooling) + BCEWithLogitsLoss

Protocol & Hyperparameters:
  - Loss: nn.BCEWithLogitsLoss()
  - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
  - Batch size: 16
  - Epochs: 20
  - Model Selection Metric: Validation Macro AUROC
  - Split: 13,673 train / 3,407 val / 4,308 test (0 patient leakage)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import ECGResNet, NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
from src.models.attention_pooling import ECGResNetAttention, build_attention_model
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_patient_level_multilabel_splits,
    PTBXLMultiLabelECGDataset,
)
from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    SPLIT_RANDOM_STATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

PHASE6_DIR = RESULTS_DIR / "phase6"
PHASE6_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_threads)
num_workers = min(4, os.cpu_count() or 2)


# =============================================================================
# Multi-Label Metric Computation Helper
# =============================================================================

def compute_multilabel_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard multi-label clinical benchmark metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    # 1. Macro AUROC
    try:
        macro_auroc = float(roc_auc_score(y_true, y_probs, average="macro"))
    except Exception:
        macro_auroc = 0.0

    # 2. Macro Average Precision (PR-AUC)
    try:
        macro_ap = float(average_precision_score(y_true, y_probs, average="macro"))
    except Exception:
        macro_ap = 0.0

    # 3. Macro F1 & Weighted F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 4. Subset Exact Match Accuracy
    subset_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))

    # 5. Hamming Loss (fraction of wrong labels)
    hamming_loss = float(np.mean(y_pred != y_true))

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming_loss,
    }


# =============================================================================
# Training & Validation Loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for 1 epoch using multi-label BCE."""
    model.train()
    running_loss = 0.0
    total = 0

    for batch in dataloader:
        ecgs = batch["ecg"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ecgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * ecgs.size(0)
        total += labels.size(0)

    return running_loss / total


def evaluate_multilabel_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate multi-label performance on dataloader."""
    model.eval()
    running_loss = 0.0
    all_logits, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ecgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * ecgs.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    total = len(dataloader.dataset)
    avg_loss = running_loss / total
    logits_tensor = torch.cat(all_logits, dim=0)
    targets_np = torch.cat(all_targets, dim=0).numpy()
    probs_np = torch.sigmoid(logits_tensor).numpy()

    metrics = compute_multilabel_metrics(targets_np, probs_np, threshold=0.5)

    return avg_loss, metrics, targets_np, probs_np, logits_tensor.numpy()


# =============================================================================
# Model Training Runner
# =============================================================================

def train_multilabel_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ckpt_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Train a single multi-label model for 20 epochs."""
    print("\n" + "=" * 80)
    print(f"TRAINING MULTI-LABEL: {model_name}")
    print("=" * 80)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_auroc = 0.0
    best_val_ap = 0.0
    best_val_macro_f1 = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_multilabel_dataset(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        val_auroc = val_metrics["macro_auroc"]
        is_best = val_auroc > best_val_macro_auroc
        if is_best:
            best_val_macro_auroc = val_auroc
            best_val_ap = val_metrics["macro_ap"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            marker = " [* BEST AUROC]"
        else:
            marker = ""

        history.append({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_ap": val_metrics["macro_ap"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_subset_acc": val_metrics["subset_accuracy"],
            "epoch_time_sec": elapsed,
            "is_best": is_best,
        })

        print(
            f"[{model_name}] Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro AUROC: {val_auroc:.4f} | Val Macro AP: {val_metrics['macro_ap']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | Time: {elapsed:.1f}s{marker}"
        )

    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": float(best_val_macro_auroc),
        "best_val_macro_ap": float(best_val_ap),
        "best_val_macro_f1": float(best_val_macro_f1),
        "checkpoint_path": str(ckpt_path),
    }
    print(f"Finished {model_name}. Best Epoch: {best_epoch} (Val Macro AUROC: {best_val_macro_auroc:.4f})")
    return summary, history


# =============================================================================
# Main Pipeline Runner
# =============================================================================

def run_multilabel_training():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 6: MULTI-LABEL PTB-XL BENCHMARK TRAINING PIPELINE")
    print("=" * 80)
    print(f"Device: {device} | Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")

    # 1. Load Data & Create Datasets
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_patient_level_multilabel_splits(df, random_state=SPLIT_RANDOM_STATE)

    print(f"Dataset Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} (0 patient leakage)")

    train_dataset = PTBXLMultiLabelECGDataset(train_df, data_dir=DATA_DIR, apply_preprocessing=True)
    val_dataset = PTBXLMultiLabelECGDataset(val_df, data_dir=DATA_DIR, apply_preprocessing=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()

    # 2. Train Model A-ML (GAP Multi-Label)
    ckpt_a_ml = CHECKPOINT_DIR / "model_a_multilabel_best.pth"
    model_a_ml = ECGResNet(num_classes=NUM_CLASSES).to(device)
    summary_a, hist_a = train_multilabel_model(
        "Model A-ML (GAP Baseline)", model_a_ml, train_loader, val_loader, criterion, device, ckpt_a_ml
    )

    # 3. Train Model B-ML (Temporal Attention Multi-Label)
    ckpt_b_ml = CHECKPOINT_DIR / "model_b_multilabel_best.pth"
    model_b_ml = build_attention_model(num_classes=NUM_CLASSES).to(device)
    summary_b, hist_b = train_multilabel_model(
        "Model B-ML (Temporal Attention)", model_b_ml, train_loader, val_loader, criterion, device, ckpt_b_ml
    )

    # 4. Save Training History
    all_histories = hist_a + hist_b
    hist_df = pd.DataFrame(all_histories)
    hist_path = PHASE6_DIR / "multilabel_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"\nSaved training histories to {hist_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 MULTI-LABEL TRAINING COMPLETE: BOTH CHECKPOINTS SAVED")
    print("=" * 80)


if __name__ == "__main__":
    run_multilabel_training()

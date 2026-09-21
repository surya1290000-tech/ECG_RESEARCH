"""
Phase 7.2 — XResNet1D Official PTB-XL Benchmark Training & Evaluation (Fold 10 Protocol)
========================================================================================

Authoritative training and evaluation script for XResNet1D on the official
PTB-XL strat_fold benchmark split:
  - Folds 1-8: Training (17,084 records, 14,823 unique patients)
  - Fold 9   : Validation (2,146 records, 1,917 unique patients)
  - Fold 10  : Frozen Test (2,158 records, 1,877 unique patients)

Experimental Controls:
  - Model: XResNet1D (ResNet-B stem, ResNet-D downsampling shortcuts, GAP + Dropout(0.3))
  - Loss: BCEWithLogitsLoss()
  - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
  - Batch Size: 16 | Epochs: 20 | Seed: 42
  - Model Selection: Best Validation Macro AUROC on Fold 9
  - Decision Thresholds: Tuned strictly on Fold 9 (Validation Split)
  - Frozen Test Evaluation: Fold 10 (2,158 records)
  - Output Checkpoint: checkpoints/model_xresnet_fold10_best.pth
  - Output Directory: results/phase7/benchmark_fold10/model_xresnet/
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xresnet1d import XResNet1D, count_parameters, NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_ptbxl_fold10_splits,
    load_preprocessed_split_arrays,
    TensorMultiLabelDataset,
)
from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

PHASE7_XRESNET_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_xresnet"
PHASE7_MODEL_A_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_a"
PHASE7_XRESNET_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CKPT_PATH = CHECKPOINT_DIR / "model_xresnet_fold10_best.pth"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Validation Threshold Search Helper
# =============================================================================

def find_optimal_thresholds(val_targets: np.ndarray, val_probs: np.ndarray) -> np.ndarray:
    """Find per-class optimal decision thresholds that maximize F1 on Validation Set."""
    thresholds = np.zeros(NUM_CLASSES, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)

    for ci in range(NUM_CLASSES):
        y_true = val_targets[:, ci]
        y_prob = val_probs[:, ci]
        best_th = 0.5
        best_f1 = 0.0

        for th in grid:
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        thresholds[ci] = best_th

    return thresholds


# =============================================================================
# Multi-Label Metrics Helper
# =============================================================================

def compute_detailed_multilabel_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray
) -> Dict[str, Any]:
    y_pred = np.zeros_like(y_true, dtype=int)
    for ci in range(NUM_CLASSES):
        y_pred[:, ci] = (y_probs[:, ci] >= thresholds[ci]).astype(int)

    # 1. Macro AUROC
    per_class_auroc = {}
    for ci, cname in enumerate(CLASS_NAMES):
        try:
            per_class_auroc[cname] = float(roc_auc_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_auroc[cname] = 0.0
    macro_auroc = float(np.mean(list(per_class_auroc.values())))

    # 2. Macro Average Precision (PR-AUC)
    per_class_ap = {}
    for ci, cname in enumerate(CLASS_NAMES):
        try:
            per_class_ap[cname] = float(average_precision_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_ap[cname] = 0.0
    macro_ap = float(np.mean(list(per_class_ap.values())))

    # 3. Macro & Weighted F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 4. Per-Class Precision, Recall, F1, Support
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_detail = {}
    for ci, cname in enumerate(CLASS_NAMES):
        per_class_detail[cname] = {
            "threshold": float(thresholds[ci]),
            "auroc": float(per_class_auroc[cname]),
            "ap": float(per_class_ap[cname]),
            "precision": float(p_arr[ci]),
            "recall": float(r_arr[ci]),
            "f1": float(f1_arr[ci]),
            "support": int(sup_arr[ci]),
        }

    # 5. Subset Exact Match Accuracy & Hamming Loss
    subset_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_loss = float(np.mean(y_pred != y_true))

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming_loss,
        "per_class": per_class_detail,
    }


# =============================================================================
# Bootstrap Confidence Intervals Helper
# =============================================================================

def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n--- COMPUTING 95% BOOTSTRAP CONFIDENCE INTERVALS (1,000 RESAMPLES) ---")
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_auroc, boot_ap, boot_f1, boot_wf1, boot_sub, boot_ham = [], [], [], [], [], []
    boot_class_auroc = {c: [] for c in CLASS_NAMES}
    boot_class_ap = {c: [] for c in CLASS_NAMES}
    boot_class_f1 = {c: [] for c in CLASS_NAMES}

    for b_idx in range(n_bootstraps):
        if (b_idx + 1) % 250 == 0:
            print(f"  Bootstrap resample {b_idx + 1}/{n_bootstraps}")
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_probs[idx]

        y_bin = np.zeros_like(yt, dtype=int)
        for ci in range(NUM_CLASSES):
            y_bin[:, ci] = (yp[:, ci] >= thresholds[ci]).astype(int)

        try:
            c_aucs = [float(roc_auc_score(yt[:, ci], yp[:, ci])) for ci in range(NUM_CLASSES)]
            boot_auroc.append(float(np.mean(c_aucs)))
            for ci, cname in enumerate(CLASS_NAMES):
                boot_class_auroc[cname].append(c_aucs[ci])
        except Exception:
            pass

        try:
            c_aps = [float(average_precision_score(yt[:, ci], yp[:, ci])) for ci in range(NUM_CLASSES)]
            boot_ap.append(float(np.mean(c_aps)))
            for ci, cname in enumerate(CLASS_NAMES):
                boot_class_ap[cname].append(c_aps[ci])
        except Exception:
            pass

        boot_f1.append(float(f1_score(yt, y_bin, average="macro", zero_division=0)))
        boot_wf1.append(float(f1_score(yt, y_bin, average="weighted", zero_division=0)))
        boot_sub.append(float(np.mean(np.all(y_bin == yt, axis=1))))
        boot_ham.append(float(np.mean(y_bin != yt)))
        for ci, cname in enumerate(CLASS_NAMES):
            boot_class_f1[cname].append(float(f1_score(yt[:, ci], y_bin[:, ci], zero_division=0)))

    def summarize(arr: List[float]) -> Dict[str, float]:
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "std": float(np.std(arr_np)),
            "ci_2.5": float(np.percentile(arr_np, 2.5)),
            "ci_97.5": float(np.percentile(arr_np, 97.5)),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "macro_auroc": summarize(boot_auroc),
        "macro_ap": summarize(boot_ap),
        "macro_f1": summarize(boot_f1),
        "weighted_f1": summarize(boot_wf1),
        "subset_accuracy": summarize(boot_sub),
        "hamming_loss": summarize(boot_ham),
        "per_class_auroc": {c: summarize(boot_class_auroc[c]) for c in CLASS_NAMES},
        "per_class_ap": {c: summarize(boot_class_ap[c]) for c in CLASS_NAMES},
        "per_class_f1": {c: summarize(boot_class_f1[c]) for c in CLASS_NAMES},
    }


# =============================================================================
# ROC Curves Plotting Helper
# =============================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Path,
    macro_auroc: float,
) -> None:
    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ci, (cname, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, ci], y_probs[:, ci])
        class_auc = roc_auc_score(y_true[:, ci], y_probs[:, ci])
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2.5,
            label=f"{cname} (AUROC = {class_auc:.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance (AUROC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title(
        f"Phase 7.2: XResNet1D Fold-10 Benchmark ROC Curves\nMacro AUROC = {macro_auroc:.4f}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved ROC curves plot to: {save_path}")


# =============================================================================
# Paired Comparison with Model A Helper
# =============================================================================

def compute_paired_comparison_with_model_a(
    test_Y: np.ndarray,
    xresnet_probs: np.ndarray,
    xresnet_metrics: Dict[str, Any],
    model_a_preds_path: Path,
    model_a_metrics_path: Path,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n--- STEP 8: PERFORMING PAIRED STATISTICAL COMPARISON WITH MODEL A ---")
    
    assert model_a_preds_path.exists(), f"Model A predictions missing at {model_a_preds_path}"
    assert model_a_metrics_path.exists(), f"Model A metrics missing at {model_a_metrics_path}"

    model_a_df = pd.read_csv(model_a_preds_path)
    with open(model_a_metrics_path, "r") as f:
        model_a_metrics_data = json.load(f)

    # Extract Model A probabilities
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    model_a_probs = model_a_df[prob_cols].values

    # Model A metrics (validation tuned)
    ma_metrics = model_a_metrics_data["test_metrics_validation_tuned_thresholds"]

    # Metric Deltas (XResNet - Model A)
    macro_auroc_delta = xresnet_metrics["macro_auroc"] - ma_metrics["macro_auroc"]
    macro_auroc_rel_pct = (macro_auroc_delta / ma_metrics["macro_auroc"]) * 100.0
    macro_ap_delta = xresnet_metrics["macro_ap"] - ma_metrics["macro_ap"]
    macro_f1_delta = xresnet_metrics["macro_f1"] - ma_metrics["macro_f1"]
    subset_acc_delta = (xresnet_metrics["subset_accuracy"] - ma_metrics["subset_accuracy"]) * 100.0
    hamming_loss_delta = xresnet_metrics["hamming_loss"] - ma_metrics["hamming_loss"]

    # Per-class AUROC deltas
    per_class_deltas = {}
    for cname in CLASS_NAMES:
        x_auc = xresnet_metrics["per_class"][cname]["auroc"]
        a_auc = ma_metrics["per_class"][cname]["auroc"]
        per_class_deltas[cname] = {
            "xresnet_auroc": x_auc,
            "model_a_auroc": a_auc,
            "delta_auroc": x_auc - a_auc,
            "rel_pct_change": ((x_auc - a_auc) / a_auc) * 100.0,
        }

    # Paired sample log-loss comparison (per-record cross entropy)
    eps = 1e-12
    x_probs_clipped = np.clip(xresnet_probs, eps, 1.0 - eps)
    a_probs_clipped = np.clip(model_a_probs, eps, 1.0 - eps)

    # Multi-label BCE per sample
    x_bce_per_sample = -np.mean(test_Y * np.log(x_probs_clipped) + (1.0 - test_Y) * np.log(1.0 - x_probs_clipped), axis=1)
    a_bce_per_sample = -np.mean(test_Y * np.log(a_probs_clipped) + (1.0 - test_Y) * np.log(1.0 - a_probs_clipped), axis=1)

    # Paired Wilcoxon signed-rank test on per-record BCE loss
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(x_bce_per_sample, a_bce_per_sample, alternative="two-sided")
    paired_t_stat, paired_t_p = stats.ttest_rel(x_bce_per_sample, a_bce_per_sample)

    # Paired Bootstrap for delta Macro AUROC CI
    rng = np.random.RandomState(seed)
    n = len(test_Y)
    boot_delta_auroc = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = test_Y[idx]
        xp = xresnet_probs[idx]
        ap = model_a_probs[idx]
        try:
            x_auc_b = np.mean([roc_auc_score(yt[:, ci], xp[:, ci]) for ci in range(NUM_CLASSES)])
            a_auc_b = np.mean([roc_auc_score(yt[:, ci], ap[:, ci]) for ci in range(NUM_CLASSES)])
            boot_delta_auroc.append(float(x_auc_b - a_auc_b))
        except Exception:
            pass

    boot_delta_np = np.array(boot_delta_auroc)
    p_value_bootstrap = float(np.mean(boot_delta_np <= 0.0)) if macro_auroc_delta > 0 else float(np.mean(boot_delta_np >= 0.0))

    comparison_results = {
        "model_a_name": "Model A-ML (ECGResNet GAP Baseline)",
        "candidate_model_name": "Model XResNet1D (Bag-of-Tricks)",
        "macro_auroc": {
            "model_a": ma_metrics["macro_auroc"],
            "xresnet": xresnet_metrics["macro_auroc"],
            "delta_absolute": float(macro_auroc_delta),
            "delta_relative_pct": float(macro_auroc_rel_pct),
            "paired_bootstrap_95ci": [float(np.percentile(boot_delta_np, 2.5)), float(np.percentile(boot_delta_np, 97.5))],
            "bootstrap_p_value": p_value_bootstrap,
        },
        "macro_ap": {
            "model_a": ma_metrics["macro_ap"],
            "xresnet": xresnet_metrics["macro_ap"],
            "delta_absolute": float(macro_ap_delta),
        },
        "macro_f1": {
            "model_a": ma_metrics["macro_f1"],
            "xresnet": xresnet_metrics["macro_f1"],
            "delta_absolute": float(macro_f1_delta),
        },
        "subset_accuracy": {
            "model_a_pct": float(ma_metrics["subset_accuracy"] * 100),
            "xresnet_pct": float(xresnet_metrics["subset_accuracy"] * 100),
            "delta_pct_points": float(subset_acc_delta),
        },
        "hamming_loss": {
            "model_a": ma_metrics["hamming_loss"],
            "xresnet": xresnet_metrics["hamming_loss"],
            "delta_absolute": float(hamming_loss_delta),
        },
        "per_class_auroc_deltas": per_class_deltas,
        "statistical_tests": {
            "paired_bce_wilcoxon_statistic": float(wilcoxon_stat),
            "paired_bce_wilcoxon_p_value": float(wilcoxon_p),
            "paired_bce_ttest_statistic": float(paired_t_stat),
            "paired_bce_ttest_p_value": float(paired_t_p),
            "mean_bce_model_a": float(np.mean(a_bce_per_sample)),
            "mean_bce_xresnet": float(np.mean(x_bce_per_sample)),
        },
    }

    print(f"  Model A Macro AUROC : {ma_metrics['macro_auroc']:.4f}")
    print(f"  XResNet Macro AUROC : {xresnet_metrics['macro_auroc']:.4f}")
    print(f"  Delta Macro AUROC   : {macro_auroc_delta:+.4f} ({macro_auroc_rel_pct:+.2f}%)")
    print(f"  Delta 95% CI        : [{np.percentile(boot_delta_np, 2.5):+.4f}, {np.percentile(boot_delta_np, 97.5):+.4f}]")
    print(f"  Paired Wilcoxon p   : {wilcoxon_p:.4e}")

    return comparison_results


# =============================================================================
# Main Pipeline
# =============================================================================

def run_model_xresnet_fold10_pipeline():
    total_start = time.time()
    timing_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 7.2: XRESNET1D OFFICIAL PTB-XL BENCHMARK (FOLD 10 PROTOCOL)")
    print(f"Device                 : {device}")
    print(f"Target Output Directory: {PHASE7_XRESNET_DIR}")
    print(f"Target Checkpoint      : {CKPT_PATH}")
    print("=" * 80)

    # 1. Metadata & Split Loading
    print("\n--- STEP 1: LOADING DATASET & VERIFYING OFFICIAL FOLD PROTOCOL ---")
    t0 = time.time()
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)
    timing_dict["metadata_loading_time_sec"] = time.time() - t0

    print(f"  Train (Folds 1-8): {len(train_df):5d} records ({train_df['patient_id'].nunique()} patients)")
    print(f"  Val   (Fold 9)   : {len(val_df):5d} records ({val_df['patient_id'].nunique()} patients)")
    print(f"  Test  (Fold 10)  : {len(test_df):5d} records ({test_df['patient_id'].nunique()} patients)")

    # Assertions for split integrity
    assert train_df["strat_fold"].isin(range(1, 9)).all(), "Train split contains non-folds 1-8!"
    assert val_df["strat_fold"].eq(9).all(), "Validation split is not exactly fold 9!"
    assert test_df["strat_fold"].eq(10).all(), "Test split is not exactly fold 10!"

    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())

    assert len(train_pids & val_pids) == 0, "Patient overlap between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient overlap between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient overlap between val and test!"
    print("  Zero patient overlap strictly verified across all partitions.")

    # 2. Preprocessed Signals Loading
    print("\n--- STEP 2: LOADING PREPROCESSED SIGNAL ARRAYS ---")
    t0 = time.time()
    train_X, train_Y, train_ids = load_preprocessed_split_arrays(train_df, data_dir=DATA_DIR, verbose=True)
    val_X, val_Y, val_ids = load_preprocessed_split_arrays(val_df, data_dir=DATA_DIR, verbose=True)
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=True)
    timing_dict["signal_preprocessing_time_sec"] = time.time() - t0

    train_loader = DataLoader(TensorMultiLabelDataset(train_X, train_Y, train_ids), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorMultiLabelDataset(val_X, val_Y, val_ids), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorMultiLabelDataset(test_X, test_Y, test_ids), batch_size=128, shuffle=False, num_workers=0)

    # 3. Model Instantiation & Training
    print("\n--- STEP 3: INITIALIZING XRESNET1D BACKBONE ---")
    model = XResNet1D(num_classes=NUM_CLASSES).to(device)
    param_count = count_parameters(model)
    print(f"  XResNet1D Trainable Parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_auroc = 0.0
    best_epoch = 0
    history = []

    print("\n--- STARTING 20-EPOCH TRAINING OF XRESNET1D (FOLDS 1-8) ---")
    t_train = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ecgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * ecgs.size(0)
            total += labels.size(0)

        train_loss = running_loss / total

        # Validation on Fold 9
        model.eval()
        val_logits_list = []
        val_loss_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ecgs = batch["ecg"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ecgs)
                loss = criterion(logits, labels)
                val_loss_running += loss.item() * ecgs.size(0)
                val_logits_list.append(logits.cpu())

        val_loss = val_loss_running / len(val_Y)
        val_logits = torch.cat(val_logits_list, dim=0).numpy()
        val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()

        val_auc = float(roc_auc_score(val_Y, val_probs, average="macro"))
        val_ap = float(average_precision_score(val_Y, val_probs, average="macro"))
        val_f1_05 = float(f1_score(val_Y, (val_probs >= 0.5).astype(int), average="macro", zero_division=0))
        elapsed_ep = time.time() - t_ep

        is_best = val_auc > best_val_macro_auroc
        if is_best:
            best_val_macro_auroc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " [* BEST VAL AUROC]"
        else:
            marker = ""

        history.append({
            "model": "XResNet1D (Fold 10)",
            "seed": SEED,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_auc,
            "val_macro_ap": val_ap,
            "val_macro_f1_05": val_f1_05,
            "epoch_time_sec": elapsed_ep,
            "is_best": is_best,
        })

        print(
            f"[XResNet1D] Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | "
            f"Val Macro AUROC: {val_auc:.4f} | Val Macro AP: {val_ap:.4f} | "
            f"Val Macro F1 (0.5): {val_f1_05:.4f} | Time: {elapsed_ep:.2f}s{marker}"
        )

    timing_dict["training_time_sec"] = time.time() - t_train
    pd.DataFrame(history).to_csv(PHASE7_XRESNET_DIR / "model_xresnet_fold10_training_history.csv", index=False)

    # 4. Threshold Tuning Strictly on Fold 9 (Validation)
    print("\n--- STEP 4: TUNING DECISION THRESHOLDS ON FOLD 9 (VALIDATION) ---")
    t0 = time.time()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    model.eval()

    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            val_logits_list.append(model(batch["ecg"].to(device)).cpu())
    val_probs = torch.sigmoid(torch.cat(val_logits_list, dim=0)).numpy()
    optimal_thresholds = find_optimal_thresholds(val_Y, val_probs)
    timing_dict["val_threshold_tuning_time_sec"] = time.time() - t0

    for ci, cname in enumerate(CLASS_NAMES):
        print(f"  {cname}: optimal threshold = {optimal_thresholds[ci]:.2f}")

    # 5. Frozen Evaluation on Fold 10 Test Set
    print("\n--- STEP 5: EVALUATING FROZEN TEST SET (FOLD 10, N = 2,158) ---")
    t0 = time.time()
    test_logits_list = []
    test_loss_running = 0.0
    with torch.no_grad():
        for batch in test_loader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ecgs)
            test_loss_running += criterion(logits, labels).item() * ecgs.size(0)
            test_logits_list.append(logits.cpu())

    test_loss = test_loss_running / len(test_Y)
    test_logits = torch.cat(test_logits_list, dim=0).numpy()
    test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    timing_dict["test_inference_time_sec"] = time.time() - t0

    # 6. Compute Detailed Metrics
    th_05 = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    metrics_05 = compute_detailed_multilabel_metrics(test_Y, test_probs, th_05)
    metrics_opt = compute_detailed_multilabel_metrics(test_Y, test_probs, optimal_thresholds)

    print(f"\n--- OFFICIAL PTB-XL FOLD 10 XRESNET1D RESULTS ---")
    print(f"  Macro AUROC          : {metrics_opt['macro_auroc']:.4f}")
    print(f"  Macro AP (PR-AUC)    : {metrics_opt['macro_ap']:.4f}")
    print(f"  Macro F1 (Val-Tuned) : {metrics_opt['macro_f1']:.4f}")
    print(f"  Macro F1 (Th=0.5)    : {metrics_05['macro_f1']:.4f}")
    print(f"  Weighted F1          : {metrics_opt['weighted_f1']:.4f}")
    print(f"  Subset Accuracy      : {metrics_opt['subset_accuracy']*100:.2f}%")
    print(f"  Hamming Loss         : {metrics_opt['hamming_loss']:.4f}")
    print(f"  Test BCE Loss        : {test_loss:.4f}")

    # 7. Bootstrap Confidence Intervals
    t0 = time.time()
    boot_cis = compute_bootstrap_cis(test_Y, test_probs, optimal_thresholds, n_bootstraps=1000, seed=SEED)
    timing_dict["bootstrap_ci_time_sec"] = time.time() - t0

    # 8. Plot ROC Curves
    roc_plot_path = PHASE7_XRESNET_DIR / "model_xresnet_fold10_roc_curves.png"
    plot_roc_curves(test_Y, test_probs, CLASS_NAMES, roc_plot_path, metrics_opt["macro_auroc"])

    # 9. Paired Comparison with Model A
    ma_preds_path = PHASE7_MODEL_A_DIR / "model_a_fold10_predictions.csv"
    ma_metrics_path = PHASE7_MODEL_A_DIR / "model_a_fold10_metrics.json"
    comparison_data = compute_paired_comparison_with_model_a(
        test_Y=test_Y,
        xresnet_probs=test_probs,
        xresnet_metrics=metrics_opt,
        model_a_preds_path=ma_preds_path,
        model_a_metrics_path=ma_metrics_path,
        n_bootstraps=1000,
        seed=SEED,
    )
    with open(PHASE7_XRESNET_DIR / "model_xresnet_vs_model_a_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Save comparison table as CSV
    comp_rows = [
        {"metric": "Macro AUROC", "model_a": comparison_data["macro_auroc"]["model_a"], "xresnet1d": comparison_data["macro_auroc"]["xresnet"], "delta": comparison_data["macro_auroc"]["delta_absolute"], "relative_pct": comparison_data["macro_auroc"]["delta_relative_pct"]},
        {"metric": "Macro AP", "model_a": comparison_data["macro_ap"]["model_a"], "xresnet1d": comparison_data["macro_ap"]["xresnet"], "delta": comparison_data["macro_ap"]["delta_absolute"], "relative_pct": (comparison_data["macro_ap"]["delta_absolute"] / comparison_data["macro_ap"]["model_a"]) * 100},
        {"metric": "Macro F1 (Val-Tuned)", "model_a": comparison_data["macro_f1"]["model_a"], "xresnet1d": comparison_data["macro_f1"]["xresnet"], "delta": comparison_data["macro_f1"]["delta_absolute"], "relative_pct": (comparison_data["macro_f1"]["delta_absolute"] / comparison_data["macro_f1"]["model_a"]) * 100},
        {"metric": "Subset Accuracy (%)", "model_a": comparison_data["subset_accuracy"]["model_a_pct"], "xresnet1d": comparison_data["subset_accuracy"]["xresnet_pct"], "delta": comparison_data["subset_accuracy"]["delta_pct_points"], "relative_pct": (comparison_data["subset_accuracy"]["delta_pct_points"] / comparison_data["subset_accuracy"]["model_a_pct"]) * 100},
        {"metric": "Hamming Loss", "model_a": comparison_data["hamming_loss"]["model_a"], "xresnet1d": comparison_data["hamming_loss"]["xresnet"], "delta": comparison_data["hamming_loss"]["delta_absolute"], "relative_pct": (comparison_data["hamming_loss"]["delta_absolute"] / comparison_data["hamming_loss"]["model_a"]) * 100},
    ]
    for cname in CLASS_NAMES:
        row = comparison_data["per_class_auroc_deltas"][cname]
        comp_rows.append({
            "metric": f"{cname} AUROC",
            "model_a": row["model_a_auroc"],
            "xresnet1d": row["xresnet_auroc"],
            "delta": row["delta_auroc"],
            "relative_pct": row["rel_pct_change"],
        })
    pd.DataFrame(comp_rows).to_csv(PHASE7_XRESNET_DIR / "model_xresnet_vs_model_a_comparison.csv", index=False)

    # 10. Save Deliverables
    total_time = time.time() - total_start
    timing_dict["total_pipeline_time_sec"] = total_time
    ckpt_sha = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()

    pred_bin = np.zeros_like(test_Y, dtype=int)
    for ci in range(NUM_CLASSES):
        pred_bin[:, ci] = (test_probs[:, ci] >= optimal_thresholds[ci]).astype(int)

    preds_df = pd.DataFrame({
        "ecg_id": test_ids,
        "true_NORM": test_Y[:, 0].astype(int), "prob_NORM": test_probs[:, 0], "pred_NORM": pred_bin[:, 0],
        "true_STTC": test_Y[:, 1].astype(int), "prob_STTC": test_probs[:, 1], "pred_STTC": pred_bin[:, 1],
        "true_CD": test_Y[:, 2].astype(int), "prob_CD": test_probs[:, 2], "pred_CD": pred_bin[:, 2],
        "true_MI": test_Y[:, 3].astype(int), "prob_MI": test_probs[:, 3], "pred_MI": pred_bin[:, 3],
        "true_HYP": test_Y[:, 4].astype(int), "prob_HYP": test_probs[:, 4], "pred_HYP": pred_bin[:, 4],
    })
    preds_df.to_csv(PHASE7_XRESNET_DIR / "model_xresnet_fold10_predictions.csv", index=False)

    class_rows = []
    for ci, cname in enumerate(CLASS_NAMES):
        row = metrics_opt["per_class"][cname]
        c_boot_auc = boot_cis["per_class_auroc"][cname]
        c_boot_ap = boot_cis["per_class_ap"][cname]

        class_rows.append({
            "class_name": cname,
            "support": int(row["support"]),
            "prevalence_pct": float(row["support"] / len(test_Y) * 100),
            "optimal_val_threshold": row["threshold"],
            "auroc": row["auroc"],
            "auroc_95ci": f"[{c_boot_auc['ci_2.5']:.4f}, {c_boot_auc['ci_97.5']:.4f}]",
            "ap": row["ap"],
            "ap_95ci": f"[{c_boot_ap['ci_2.5']:.4f}, {c_boot_ap['ci_97.5']:.4f}]",
            "f1_val_tuned": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
        })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(PHASE7_XRESNET_DIR / "model_xresnet_fold10_classification_report.csv", index=False)

    with open(PHASE7_XRESNET_DIR / "model_xresnet_fold10_bootstrap_cis.json", "w") as f:
        json.dump(boot_cis, f, indent=2)

    master_metrics = {
        "model_name": "XResNet1D (Bag-of-Tricks 1D ResNet Fold 10)",
        "model_parameter_count": param_count,
        "seed": SEED,
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": best_epoch,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_protocol": "Official PTB-XL strat_fold (Train=Folds 1-8, Val=Fold 9, Test=Fold 10)",
        "train_records_count": len(train_df),
        "val_records_count": len(val_Y),
        "test_records_count": len(test_Y),
        "optimal_validation_thresholds": {c: float(optimal_thresholds[i]) for i, c in enumerate(CLASS_NAMES)},
        "test_loss_bce": float(test_loss),
        "test_metrics_default_0_5_threshold": metrics_05,
        "test_metrics_validation_tuned_thresholds": metrics_opt,
        "bootstrap_95ci": boot_cis,
        "paired_comparison_with_model_a": comparison_data,
        "timing_metrics": timing_dict,
    }
    with open(PHASE7_XRESNET_DIR / "model_xresnet_fold10_metrics.json", "w") as f:
        json.dump(master_metrics, f, indent=2)

    # 11. Markdown Report
    report_md = f"""# Phase 7.2 — XResNet1D — Official Fold-10 Benchmark Report

_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  
_Split: Official PTB-XL `strat_fold` (Train: Folds 1–8 = {len(train_df):,}, Val: Fold 9 = {len(val_Y):,}, Test: Fold 10 = {len(test_Y):,})_  
_Total Execution Time: {total_time:.2f} seconds (Training: {timing_dict['training_time_sec']:.2f}s, Inference: {timing_dict['test_inference_time_sec']:.2f}s, Bootstrap: {timing_dict['bootstrap_ci_time_sec']:.2f}s)_

---

## 1. Executive Summary & Architecture Specification

Experiment 7.2 introduces **XResNet1D**, adapting the "Bag of Tricks for Image Classification" (He et al., 2019) to 1-dimensional 12-lead electrocardiography.

### Architectural Innovations (vs Model A Baseline):
1. **ResNet-B (Stem Tweak)**: Replaced monolithic $7\\times 1$ conv with a 3-layer convolutional stem ($12 \\to 32 \\to 32 \\to 64$ channels) with BatchNorm and ReLU.
2. **ResNet-C (Conv Placement)**: Preserved receptive field in downsampling residual blocks.
3. **ResNet-D (Anti-Aliased Downsampling Shortcut)**: Placed `AvgPool1d(kernel_size=2, stride=2)` before $1\\times 1$ projection to eliminate subsampling artifacts.
4. **Parameter Count**: **{param_count:,} parameters** (Model A: 3,919,493 parameters; delta = {param_count - 3919493:+,d} params / +{(param_count - 3919493)/3919493*100:.2f}%).

### Checkpoint Integrity:
- **Checkpoint Path**: `{CKPT_PATH}`
- **SHA-256 Checksum**: `{ckpt_sha}`
- **Best Validation Epoch**: Epoch {best_epoch}

---

## 2. Benchmark Performance Comparison (Fold 10 Frozen Test Set, $N = 2,158$)

| Metric | Model A-ML (GAP Baseline) | **XResNet1D (Phase 7.2)** | Absolute $\\Delta$ | Relative Change (%) | Strodthoff et al. (2021) `resnet1d_wang` | Strodthoff et al. (2021) `xresnet1d101` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Macro AUROC** | **0.8868** | **{metrics_opt['macro_auroc']:.4f}** (95% CI: [{boot_cis['macro_auroc']['ci_2.5']:.4f}, {boot_cis['macro_auroc']['ci_97.5']:.4f}]) | **{comparison_data['macro_auroc']['delta_absolute']:+.4f}** | **{comparison_data['macro_auroc']['delta_relative_pct']:+.2f}%** | 0.925 (±0.006) | 0.932 (±0.005) |
| **Macro AP (PR-AUC)** | **0.7334** | **{metrics_opt['macro_ap']:.4f}** (95% CI: [{boot_cis['macro_ap']['ci_2.5']:.4f}, {boot_cis['macro_ap']['ci_97.5']:.4f}]) | **{comparison_data['macro_ap']['delta_absolute']:+.4f}** | **{(comparison_data['macro_ap']['delta_absolute']/comparison_data['macro_ap']['model_a'])*100:+.2f}%** | — | — |
| **Macro F1 (Val-Tuned)** | **0.6907** | **{metrics_opt['macro_f1']:.4f}** (95% CI: [{boot_cis['macro_f1']['ci_2.5']:.4f}, {boot_cis['macro_f1']['ci_97.5']:.4f}]) | **{comparison_data['macro_f1']['delta_absolute']:+.4f}** | **{(comparison_data['macro_f1']['delta_absolute']/comparison_data['macro_f1']['model_a'])*100:+.2f}%** | ~0.72–0.74 | ~0.74–0.76 |
| **Macro F1 (Th=0.5)** | **0.6214** | **{metrics_05['macro_f1']:.4f}** | **{metrics_05['macro_f1'] - 0.621352571:+.4f}** | — | — | — |
| **Weighted F1** | **0.7400** | **{metrics_opt['weighted_f1']:.4f}** | **{metrics_opt['weighted_f1'] - 0.740041347:+.4f}** | — | — | — |
| **Subset Exact Match Acc.** | **56.67%** | **{metrics_opt['subset_accuracy']*100:.2f}%** | **{comparison_data['subset_accuracy']['delta_pct_points']:+.2f}%** | — | — | ~60% |
| **Hamming Loss** | **0.1399** | **{metrics_opt['hamming_loss']:.4f}** | **{comparison_data['hamming_loss']['delta_absolute']:+.4f}** | — | — | — |
| **Test BCE Loss** | **0.3164** | **{test_loss:.4f}** | **{test_loss - 0.316438988:+.4f}** | — | — | — |

---

## 3. Per-Class Diagnostic Performance Breakdown

| Superclass | Support | Prev. | Val Thresh | Model A AUROC | **XResNet1D AUROC (95% CI)** | $\\Delta$ AUROC | XResNet1D AP | XResNet1D F1 | XResNet1D Recall | XResNet1D Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in class_rows:
        cname = r["class_name"]
        ma_c_auc = comparison_data["per_class_auroc_deltas"][cname]["model_a_auroc"]
        delta_c = comparison_data["per_class_auroc_deltas"][cname]["delta_auroc"]
        report_md += f"| **{cname}** | {r['support']} | {r['prevalence_pct']:.1f}% | {r['optimal_val_threshold']:.2f} | {ma_c_auc:.4f} | **{r['auroc']:.4f}** ({r['auroc_95ci']}) | **{delta_c:+.4f}** | {r['ap']:.4f} | **{r['f1_val_tuned']:.4f}** | {r['recall']*100:.1f}% | {r['precision']*100:.1f}% |\n"

    report_md += f"""
---

## 4. Statistical Hypothesis Testing on Paired Test Records ($N = 2,158$)

- **Paired Sample BCE Loss Comparison**:
  - Model A Mean Loss: `{comparison_data['statistical_tests']['mean_bce_model_a']:.4f}`
  - XResNet1D Mean Loss: `{comparison_data['statistical_tests']['mean_bce_xresnet']:.4f}`
  - Wilcoxon Signed-Rank Test $p$-value: `{comparison_data['statistical_tests']['paired_bce_wilcoxon_p_value']:.4e}`
  - Paired $t$-test $p$-value: `{comparison_data['statistical_tests']['paired_bce_ttest_p_value']:.4e}`
- **Bootstrap 95% CI on $\\Delta$ Macro AUROC**:
  - `[{comparison_data['macro_auroc']['paired_bootstrap_95ci'][0]:+.4f}, {comparison_data['macro_auroc']['paired_bootstrap_95ci'][1]:+.4f}]`
  - Empirical bootstrap $p$-value: `{comparison_data['macro_auroc']['bootstrap_p_value']:.4f}`

---

## 5. Deliverables Preserved in [`results/phase7/benchmark_fold10/model_xresnet/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/)

- [`checkpoints/model_xresnet_fold10_best.pth`](file:///c:/Users/ASUS/Desktop/ECG_Research/checkpoints/model_xresnet_fold10_best.pth)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_metrics.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_metrics.json)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_classification_report.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_classification_report.csv)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_roc_curves.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_roc_curves.png)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_bootstrap_cis.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_bootstrap_cis.json)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_training_history.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_training_history.csv)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_vs_model_a_comparison.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_vs_model_a_comparison.csv)
- [`results/phase7/benchmark_fold10/model_xresnet/model_xresnet_vs_model_a_comparison.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/model_xresnet_vs_model_a_comparison.json)
- [`results/phase7/benchmark_fold10/model_xresnet/MODEL_XRESNET_FOLD10_REPORT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_xresnet/MODEL_XRESNET_FOLD10_REPORT.md)
"""

    report_path = PHASE7_XRESNET_DIR / "MODEL_XRESNET_FOLD10_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSaved Report to {report_path}")

    # 12. Deliverable Verification
    print("\n--- STEP 12: VERIFYING DELIVERABLES ---")
    expected_files = [
        "model_xresnet_fold10_metrics.json",
        "model_xresnet_fold10_classification_report.csv",
        "model_xresnet_fold10_predictions.csv",
        "model_xresnet_fold10_roc_curves.png",
        "model_xresnet_fold10_bootstrap_cis.json",
        "model_xresnet_fold10_training_history.csv",
        "model_xresnet_vs_model_a_comparison.csv",
        "model_xresnet_vs_model_a_comparison.json",
        "MODEL_XRESNET_FOLD10_REPORT.md",
    ]
    for fname in expected_files:
        fpath = PHASE7_XRESNET_DIR / fname
        assert fpath.exists(), f"MISSING: {fpath}"
        print(f"  OK: {fname} ({fpath.stat().st_size} bytes)")

    print("\n" + "=" * 80)
    print("PHASE 7.2 XRESNET1D BENCHMARK COMPLETE")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    run_model_xresnet_fold10_pipeline()

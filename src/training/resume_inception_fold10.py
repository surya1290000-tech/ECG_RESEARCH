"""
Phase 7.5 — InceptionTime Resume Training (Epochs 7-20) + Full Evaluation
==========================================================================

Resumes InceptionTime1D training from the Epoch 6 best checkpoint
(Val Macro AUROC = 0.8933, saved at checkpoints/model_inception_fold10_best.pth)
and continues training for epochs 7 through 20.

The optimizer is re-initialized (Adam, lr=1e-3, weight_decay=1e-4) since only
model.state_dict() was preserved. This is standard practice for warm-restart
fine-tuning.

After training completes:
  - Threshold tuning on Fold 9 (validation)
  - Frozen evaluation on Fold 10 (test)
  - Bootstrap confidence intervals (1,000 resamples)
  - Full deliverable generation in results/phase7/benchmark_fold10/model_inception/
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

# Ensure immediate unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Set PyTorch to utilize all 20 CPU threads for maximum throughput
torch.set_num_threads(20)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inceptiontime1d import InceptionTime1D, count_parameters, NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
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

PHASE7_INCEPTION_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_inception"
PHASE7_MODEL_A_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_a"

PHASE7_INCEPTION_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CKPT_PATH = CHECKPOINT_DIR / "model_inception_fold10_best.pth"
CKPT_FULL_PATH = CHECKPOINT_DIR / "model_inception_fold10_full.pth"  # Full checkpoint for resume

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Resume configuration
RESUME_FROM_EPOCH = 6
BEST_VAL_AUROC_SO_FAR = 0.8933  # From Epoch 6

# Epoch timings from completed epochs (for training_history.csv reconstruction)
COMPLETED_HISTORY = [
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 1, "train_loss": 0.3741, "val_loss": 0.3339, "val_macro_auroc": 0.8687, "val_macro_ap": 0.7015, "val_macro_f1_05": 0.5721, "epoch_time_sec": 21.03*60, "is_best": True},
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 2, "train_loss": 0.3319, "val_loss": 0.3198, "val_macro_auroc": 0.8812, "val_macro_ap": 0.7326, "val_macro_f1_05": 0.6678, "epoch_time_sec": 32.23*60, "is_best": True},
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 3, "train_loss": 0.3194, "val_loss": 0.3220, "val_macro_auroc": 0.8861, "val_macro_ap": 0.7445, "val_macro_f1_05": 0.5850, "epoch_time_sec": 15.79*60, "is_best": True},
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 4, "train_loss": 0.3126, "val_loss": 0.3161, "val_macro_auroc": 0.8852, "val_macro_ap": 0.7410, "val_macro_f1_05": 0.6330, "epoch_time_sec": 10.86*60, "is_best": False},
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 5, "train_loss": 0.3048, "val_loss": 0.3201, "val_macro_auroc": 0.8894, "val_macro_ap": 0.7379, "val_macro_f1_05": 0.6572, "epoch_time_sec": 44.68*60, "is_best": True},
    {"model": "InceptionTime1D (Fold 10)", "seed": 42, "epoch": 6, "train_loss": 0.2988, "val_loss": 0.3092, "val_macro_auroc": 0.8933, "val_macro_ap": 0.7530, "val_macro_f1_05": 0.6772, "epoch_time_sec": 312.11*60, "is_best": True},
]


# =============================================================================
# Validation Threshold Search Helper
# =============================================================================

def find_optimal_thresholds(val_targets: np.ndarray, val_probs: np.ndarray) -> np.ndarray:
    """Find per-class optimal decision thresholds maximizing F1 on Validation Set."""
    thresholds = np.zeros(NUM_CLASSES, dtype=np.float32)
    grid = np.arange(0.05, 0.951, 0.01)

    for ci in range(NUM_CLASSES):
        y_true = val_targets[:, ci]
        y_prob = val_probs[:, ci]
        best_th = 0.5
        best_f1 = -1.0

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

    # 4. Subset Accuracy (exact match)
    subset_accuracy = float(np.mean(np.all(y_pred == y_true, axis=1)))

    # 5. Hamming Loss
    hamming = float(np.mean(y_pred != y_true))

    # 6. Per-class detailed
    per_class = {}
    for ci, cname in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_pred[:, ci] == 1) & (y_true[:, ci] == 1)))
        fp = int(np.sum((y_pred[:, ci] == 1) & (y_true[:, ci] == 0)))
        fn = int(np.sum((y_pred[:, ci] == 0) & (y_true[:, ci] == 1)))
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0

        per_class[cname] = {
            "auroc": per_class_auroc[cname],
            "ap": per_class_ap[cname],
            "threshold": float(thresholds[ci]),
            "f1": float(f1_val),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "support": int(np.sum(y_true[:, ci])),
        }

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming,
        "per_class": per_class,
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def compute_bootstrap_cis(
    y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray,
    n_bootstraps: int = 1000, seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_auroc, boot_ap, boot_f1, boot_wf1, boot_sub, boot_ham = [], [], [], [], [], []
    boot_class_auroc = {c: [] for c in CLASS_NAMES}
    boot_class_ap = {c: [] for c in CLASS_NAMES}
    boot_class_f1 = {c: [] for c in CLASS_NAMES}

    for b_idx in range(n_bootstraps):
        if (b_idx + 1) % 250 == 0:
            print(f"  Bootstrap resample {b_idx + 1}/{n_bootstraps}", flush=True)
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
# Main Resume Training & Evaluation Pipeline
# =============================================================================

def run_resume_pipeline():
    total_start = time.time()
    timing_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80, flush=True)
    print("PHASE 7.5: INCEPTIONTIME RESUME FROM EPOCH 6 (EPOCHS 7-20)", flush=True)
    print(f"Device                 : {device}", flush=True)
    print(f"Resume Checkpoint      : {CKPT_PATH}", flush=True)
    print(f"Resume From Epoch      : {RESUME_FROM_EPOCH}", flush=True)
    print(f"Best Val AUROC So Far  : {BEST_VAL_AUROC_SO_FAR}", flush=True)
    print(f"Target Output Directory: {PHASE7_INCEPTION_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Metadata & Split Loading
    print("\n--- STEP 1: LOADING DATASET & VERIFYING OFFICIAL FOLD PROTOCOL ---", flush=True)
    t0 = time.time()
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)
    timing_dict["metadata_loading_time_sec"] = time.time() - t0

    print(f"  Train (Folds 1-8): {len(train_df):5d} records ({train_df['patient_id'].nunique()} patients)", flush=True)
    print(f"  Val   (Fold 9)   : {len(val_df):5d} records ({val_df['patient_id'].nunique()} patients)", flush=True)
    print(f"  Test  (Fold 10)  : {len(test_df):5d} records ({test_df['patient_id'].nunique()} patients)", flush=True)

    assert train_df["strat_fold"].isin(range(1, 9)).all(), "Train split contains non-folds 1-8!"
    assert val_df["strat_fold"].eq(9).all(), "Validation split is not exactly fold 9!"
    assert test_df["strat_fold"].eq(10).all(), "Test split is not exactly fold 10!"

    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())
    assert len(train_pids & val_pids) == 0, "Patient overlap between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient overlap between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient overlap between val and test!"
    print("  Zero patient overlap strictly verified across all partitions.", flush=True)

    # 2. Load Cached Preprocessed Arrays
    print("\n--- STEP 2: LOADING PREPROCESSED SIGNAL ARRAYS ---", flush=True)
    t0 = time.time()
    train_X, train_Y, train_ids = load_preprocessed_split_arrays(train_df, data_dir=DATA_DIR, verbose=True)
    val_X, val_Y, val_ids = load_preprocessed_split_arrays(val_df, data_dir=DATA_DIR, verbose=True)
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=True)
    timing_dict["signal_preprocessing_time_sec"] = time.time() - t0

    train_loader = DataLoader(TensorMultiLabelDataset(train_X, train_Y, train_ids), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorMultiLabelDataset(val_X, val_Y, val_ids), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorMultiLabelDataset(test_X, test_Y, test_ids), batch_size=128, shuffle=False, num_workers=0)

    # 3. Load Model from Epoch 6 Checkpoint
    print("\n--- STEP 3: LOADING INCEPTIONTIME1D FROM EPOCH 6 CHECKPOINT ---", flush=True)
    model = InceptionTime1D(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    param_count = count_parameters(model)
    print(f"  InceptionTime1D Trainable Parameters: {param_count:,}", flush=True)

    ckpt_sha_pre = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()
    print(f"  Loaded Checkpoint SHA-256: {ckpt_sha_pre}", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_auroc = BEST_VAL_AUROC_SO_FAR
    best_epoch = RESUME_FROM_EPOCH
    history = list(COMPLETED_HISTORY)  # Copy epochs 1-6
    total_batches = len(train_loader)

    # 4. Resume Training: Epochs 7-20
    print(f"\n--- RESUMING TRAINING: EPOCHS {RESUME_FROM_EPOCH + 1}-{NUM_EPOCHS} ---", flush=True)
    t_train = time.time()

    for epoch in range(RESUME_FROM_EPOCH + 1, NUM_EPOCHS + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        total = 0

        for b_idx, batch in enumerate(train_loader, 1):
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ecgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * ecgs.size(0)
            total += labels.size(0)

            if b_idx % 200 == 0 or b_idx == total_batches:
                elapsed_min = (time.time() - t_ep) / 60.0
                cur_loss = running_loss / total
                print(f"  [Epoch {epoch:02d}/{NUM_EPOCHS}] Batch [{b_idx:4d}/{total_batches}] | Train BCE: {cur_loss:.4f} | Elapsed: {elapsed_min:.2f}m", flush=True)

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
            # Save best model weights
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " [* BEST VAL AUROC - SAVED]"
        else:
            marker = ""

        # Always save full checkpoint for resume capability
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_auroc": best_val_macro_auroc,
            "best_epoch": best_epoch,
        }, CKPT_FULL_PATH)

        history.append({
            "model": "InceptionTime1D (Fold 10)",
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
            f"==> [InceptionTime1D] Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | "
            f"Val Macro AUROC: {val_auc:.4f} | Val Macro AP: {val_ap:.4f} | "
            f"Val Macro F1 (0.5): {val_f1_05:.4f} | Time: {elapsed_ep/60:.2f}m{marker}",
            flush=True,
        )

    timing_dict["training_time_sec"] = time.time() - t_train
    pd.DataFrame(history).to_csv(PHASE7_INCEPTION_DIR / "training_history.csv", index=False)
    print(f"\n  Training complete. Best epoch: {best_epoch}, Best Val Macro AUROC: {best_val_macro_auroc:.4f}", flush=True)

    # 5. Threshold Tuning Strictly on Fold 9 (Validation)
    print("\n--- STEP 5: TUNING DECISION THRESHOLDS ON FOLD 9 (VALIDATION) ---", flush=True)
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
        print(f"  {cname}: optimal threshold = {optimal_thresholds[ci]:.2f}", flush=True)

    # 6. Frozen Evaluation on Fold 10 Test Set
    print("\n--- STEP 6: EVALUATING FROZEN TEST SET (FOLD 10, N = 2,158) ---", flush=True)
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

    # 7. Compute Detailed Metrics
    th_05 = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    metrics_05 = compute_detailed_multilabel_metrics(test_Y, test_probs, th_05)
    metrics_opt = compute_detailed_multilabel_metrics(test_Y, test_probs, optimal_thresholds)

    print(f"\n--- OFFICIAL PTB-XL FOLD 10 INCEPTIONTIME1D RESULTS ---", flush=True)
    print(f"  Best Epoch           : {best_epoch}", flush=True)
    print(f"  Macro AUROC          : {metrics_opt['macro_auroc']:.4f}", flush=True)
    print(f"  Macro AP (PR-AUC)    : {metrics_opt['macro_ap']:.4f}", flush=True)
    print(f"  Macro F1 (Val-Tuned) : {metrics_opt['macro_f1']:.4f}", flush=True)
    print(f"  Macro F1 (Th=0.5)    : {metrics_05['macro_f1']:.4f}", flush=True)
    print(f"  Weighted F1          : {metrics_opt['weighted_f1']:.4f}", flush=True)
    print(f"  Subset Accuracy      : {metrics_opt['subset_accuracy']*100:.2f}%", flush=True)
    print(f"  Hamming Loss         : {metrics_opt['hamming_loss']:.4f}", flush=True)
    print(f"  Test BCE Loss        : {test_loss:.4f}", flush=True)

    # 8. Bootstrap Confidence Intervals
    print("\n--- STEP 8: COMPUTING BOOTSTRAP 95% CONFIDENCE INTERVALS (1,000 RESAMPLES) ---", flush=True)
    t0 = time.time()
    boot_cis = compute_bootstrap_cis(test_Y, test_probs, optimal_thresholds, n_bootstraps=1000, seed=SEED)
    timing_dict["bootstrap_ci_time_sec"] = time.time() - t0

    # 9. Save Deliverables
    total_time = time.time() - total_start
    timing_dict["total_pipeline_time_sec"] = total_time
    ckpt_sha = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()

    # predictions.csv
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
    preds_df.to_csv(PHASE7_INCEPTION_DIR / "predictions.csv", index=False)

    # classification_report.csv
    class_rows = []
    for ci, cname in enumerate(CLASS_NAMES):
        row = metrics_opt["per_class"][cname]
        c_boot_auc = boot_cis["per_class_auroc"][cname]
        c_boot_ap = boot_cis["per_class_ap"][cname]

        class_rows.append({
            "class_name": cname,
            "support": int(row["support"]),
            "prevalence_pct": float(row["support"] / len(test_Y) * 100),
            "optimal_val_threshold": float(row["threshold"]),
            "auroc": float(row["auroc"]),
            "auroc_95ci": f"[{c_boot_auc['ci_2.5']:.4f}, {c_boot_auc['ci_97.5']:.4f}]",
            "ap": float(row["ap"]),
            "ap_95ci": f"[{c_boot_ap['ci_2.5']:.4f}, {c_boot_ap['ci_97.5']:.4f}]",
            "f1_val_tuned": float(row["f1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(PHASE7_INCEPTION_DIR / "classification_report.csv", index=False)

    # bootstrap_cis.json
    with open(PHASE7_INCEPTION_DIR / "bootstrap_cis.json", "w") as f:
        json.dump(boot_cis, f, indent=2)

    # metrics.json
    master_metrics = {
        "model_name": "InceptionTime1D (Multi-Scale Temporal CNN Fold 10)",
        "model_parameter_count": param_count,
        "seed": SEED,
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": best_epoch,
        "total_epochs_trained": NUM_EPOCHS,
        "resumed_from_epoch": RESUME_FROM_EPOCH,
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
        "timing_metrics": timing_dict,
    }
    with open(PHASE7_INCEPTION_DIR / "metrics.json", "w") as f:
        json.dump(master_metrics, f, indent=2)

    # Load baseline model results for comparison
    with open(PHASE7_MODEL_A_DIR / "model_a_fold10_metrics.json", "r") as f:
        model_a_data = json.load(f)
    ma_metrics = model_a_data["test_metrics_validation_tuned_thresholds"]

    # 10. Markdown Report
    report_md = f"""# Phase 7.5 — InceptionTime Multi-Scale Representation Benchmark Report

_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  
_Split: Official PTB-XL `strat_fold` (Train: Folds 1–8 = {len(train_df):,}, Val: Fold 9 = {len(val_df):,}, Test: Fold 10 = {len(test_df):,})_  
_Model: InceptionTime1D (6 Inception Blocks, Multi-Scale Kernels 9/19/39, GAP + Dropout(0.3))_  
_Parameter Budget: {param_count:,} parameters (Model A: 3,919,493 parameters; delta = {param_count - 3919493:+,d} / {(param_count - 3919493)/3919493*100:+.2f}%)_  
_Checkpoint: `{CKPT_PATH}` (SHA-256: `{ckpt_sha}`)_  
_Best Epoch: {best_epoch} / {NUM_EPOCHS} (Resumed from Epoch {RESUME_FROM_EPOCH})_  
_Execution Time: Training = {timing_dict['training_time_sec']:.2f}s | Test Inference = {timing_dict['test_inference_time_sec']:.2f}s | Total = {total_time:.2f}s_

---

## 1. Executive Summary & Architectural Motivation

Experiment 7.5 evaluates **multi-scale temporal feature extraction** via a 1D Inception network (Ismail Fawaz et al., 2020) to determine whether capturing ECG dynamics across multiple receptive field scales (Branch A: 90ms, Branch B: 190ms, Branch C: 390ms at 100Hz) improves diagnostic discrimination over fixed-kernel ResNet architectures.

---

## 2. 3-Way Benchmark Performance Comparison (Fold 10 Frozen Test Set, $N = 2,158$)

| Metric | Model A-ML (ECGResNet GAP) | Model XResNet1D (Phase 7.2) | **InceptionTime1D (Phase 7.5)** | $\\Delta$ vs Model A | Relative Change (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Parameters** | 3,919,493 | 3,931,525 | **{param_count:,}** | {param_count - 3919493:+,d} | {(param_count - 3919493)/3919493*100:+.2f}% |
| **Macro AUROC** | **0.8868** | 0.8775 | **{metrics_opt['macro_auroc']:.4f}** (95% CI: [{boot_cis['macro_auroc']['ci_2.5']:.4f}, {boot_cis['macro_auroc']['ci_97.5']:.4f}]) | **{metrics_opt['macro_auroc'] - ma_metrics['macro_auroc']:+.4f}** | **{(metrics_opt['macro_auroc'] - ma_metrics['macro_auroc'])/ma_metrics['macro_auroc']*100:+.2f}%** |
| **Macro AP (PR-AUC)** | **0.7334** | 0.7276 | **{metrics_opt['macro_ap']:.4f}** (95% CI: [{boot_cis['macro_ap']['ci_2.5']:.4f}, {boot_cis['macro_ap']['ci_97.5']:.4f}]) | **{metrics_opt['macro_ap'] - ma_metrics['macro_ap']:+.4f}** | **{(metrics_opt['macro_ap'] - ma_metrics['macro_ap'])/ma_metrics['macro_ap']*100:+.2f}%** |
| **Macro F1 (Val-Tuned)** | **0.6907** | 0.6680 | **{metrics_opt['macro_f1']:.4f}** (95% CI: [{boot_cis['macro_f1']['ci_2.5']:.4f}, {boot_cis['macro_f1']['ci_97.5']:.4f}]) | **{metrics_opt['macro_f1'] - ma_metrics['macro_f1']:+.4f}** | **{(metrics_opt['macro_f1'] - ma_metrics['macro_f1'])/ma_metrics['macro_f1']*100:+.2f}%** |
| **Weighted F1** | **0.7400** | 0.7249 | **{metrics_opt['weighted_f1']:.4f}** | **{metrics_opt['weighted_f1'] - ma_metrics['weighted_f1']:+.4f}** | — |
| **Subset Accuracy** | **56.67%** | 54.49% | **{metrics_opt['subset_accuracy']*100:.2f}%** | **{(metrics_opt['subset_accuracy'] - ma_metrics['subset_accuracy'])*100:+.2f}%** | — |
| **Hamming Loss** | **0.1399** | 0.1478 | **{metrics_opt['hamming_loss']:.4f}** | **{metrics_opt['hamming_loss'] - ma_metrics['hamming_loss']:+.4f}** | — |

---

## 3. Per-Class Diagnostic Breakdown (Fold 10 Frozen Test Set, $N = 2,158$)

| Superclass | Support | Optimal Val Thresh | Model A AUROC | XResNet AUROC | **InceptionTime AUROC** | InceptionTime AP | InceptionTime F1 | InceptionTime Recall | InceptionTime Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    xresnet_aurocs = {"NORM": 0.9254, "STTC": 0.9126, "CD": 0.8931, "MI": 0.9062, "HYP": 0.7503}
    for r in class_rows:
        cname = r["class_name"]
        ma_auc = ma_metrics["per_class"][cname]["auroc"]
        xr_auc = xresnet_aurocs.get(cname, 0.0)
        report_md += f"| **{cname}** | {r['support']} | {r['optimal_val_threshold']:.2f} | {ma_auc:.4f} | {xr_auc:.4f} | **{r['auroc']:.4f}** ({r['auroc_95ci']}) | {r['ap']:.4f} | **{r['f1_val_tuned']:.4f}** | {r['recall']*100:.1f}% | {r['precision']*100:.1f}% |\n"

    report_md += f"""
---

## 4. Training Curve Summary (All 20 Epochs)

| Epoch | Train BCE | Val BCE | Val Macro AUROC | Val Macro AP | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for h in history:
        best_mark = "✅" if h["is_best"] else ""
        report_md += f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['val_macro_auroc']:.4f} | {h['val_macro_ap']:.4f} | {best_mark} |\n"

    report_md += f"""
---

## 5. Deliverables

- [`checkpoints/model_inception_fold10_best.pth`](file:///c:/Users/ASUS/Desktop/ECG_Research/checkpoints/model_inception_fold10_best.pth)
- [`results/phase7/benchmark_fold10/model_inception/metrics.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/metrics.json)
- [`results/phase7/benchmark_fold10/model_inception/predictions.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/predictions.csv)
- [`results/phase7/benchmark_fold10/model_inception/classification_report.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/classification_report.csv)
- [`results/phase7/benchmark_fold10/model_inception/bootstrap_cis.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/bootstrap_cis.json)
- [`results/phase7/benchmark_fold10/model_inception/training_history.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/training_history.csv)
- [`results/phase7/benchmark_fold10/model_inception/MODEL_INCEPTION_REPORT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/MODEL_INCEPTION_REPORT.md)
"""

    report_path = PHASE7_INCEPTION_DIR / "MODEL_INCEPTION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved: {report_path}", flush=True)

    # 11. Deliverable Verification
    print("\n--- STEP 11: VERIFYING DELIVERABLES ---", flush=True)
    expected_files = [
        "metrics.json",
        "predictions.csv",
        "classification_report.csv",
        "bootstrap_cis.json",
        "training_history.csv",
        "MODEL_INCEPTION_REPORT.md",
    ]
    for ef in expected_files:
        p = PHASE7_INCEPTION_DIR / ef
        assert p.exists(), f"MISSING DELIVERABLE: {p}"
        print(f"  OK: {ef} ({p.stat().st_size:,} bytes)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("PHASE 7.5 INCEPTIONTIME BENCHMARK COMPLETE", flush=True)
    print(f"Best Epoch: {best_epoch} | Best Val Macro AUROC: {best_val_macro_auroc:.4f}", flush=True)
    print(f"Test Macro AUROC: {metrics_opt['macro_auroc']:.4f}", flush=True)
    print(f"Total Execution Time: {total_time:.2f} seconds", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_resume_pipeline()

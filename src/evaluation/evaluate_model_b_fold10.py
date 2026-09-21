"""
Phase 9.2 — Official Fold-10 Evaluation of ECGResNet-Attention
=============================================================

Authoritative evaluation script for ECGResNet-Attention (Model B) on the frozen
official PTB-XL Fold-10 test set (N = 2,158 records).

Strict Scientific Controls:
  - Evaluation ONLY — no optimizer, no loss.backward(), no training loop.
  - Frozen official Fold-10 partition (N = 2,158 records, zero patient overlap).
  - Model loaded with strict=True and verified parameter count (3,920,006).
  - Checkpoint SHA-256 verified before evaluation.
  - Thresholds derived strictly from Fold-9 validation predictions; never tuned on Fold 10.
  - 1,000 bootstrap resamples (seed=42) for 95% confidence intervals.
  - Generates comprehensive metrics JSON, predictions CSV, bootstrap intervals, and report.

Output Directory: results/phase9/model_b_fold10_evaluation/
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
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
from src.models.attention_pooling import ECGResNetAttention, build_attention_model
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
)

OUT_DIR = RESULTS_DIR / "phase9" / "model_b_fold10_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CHECKPOINT_DIR / "model_b_fold10_best.pth"
EXPECTED_SHA256 = "8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa"
EXPECTED_PARAMS = 3920006

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
torch.set_num_threads(14)


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Validation Threshold Search Helper (Fold 9 ONLY)
# =============================================================================

def find_optimal_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    """Find per-class optimal decision thresholds maximizing F1 on Fold 9 validation set."""
    thresholds = np.zeros(NUM_CLASSES, dtype=np.float32)
    grid = np.arange(0.05, 0.951, 0.01)

    for ci in range(NUM_CLASSES):
        best_th = 0.5
        best_f1 = -1.0
        for th in grid:
            preds = (y_probs[:, ci] >= th).astype(int)
            score = f1_score(y_true[:, ci], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        thresholds[ci] = best_th

    return thresholds


# =============================================================================
# Multi-Label Metrics Computation
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
    y_pred = (y_probs >= thresholds.reshape(1, -1)).astype(int)

    # 1. Macro AUROC
    per_class_auroc = {}
    for ci, name in enumerate(CLASS_NAMES):
        try:
            per_class_auroc[name] = float(roc_auc_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_auroc[name] = 0.0
    macro_auroc = float(np.mean(list(per_class_auroc.values())))

    # 2. Macro Average Precision
    per_class_ap = {}
    for ci, name in enumerate(CLASS_NAMES):
        try:
            per_class_ap[name] = float(average_precision_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_ap[name] = 0.0
    macro_ap = float(np.mean(list(per_class_ap.values())))

    # 3. Macro & Weighted F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 4. Exact Match & Hamming Loss
    subset_accuracy = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_loss = float(np.mean(y_pred != y_true))

    # 5. Per-class metrics
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for ci, name in enumerate(CLASS_NAMES):
        yt = y_true[:, ci]
        yp = y_pred[:, ci]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class[name] = {
            "threshold": float(thresholds[ci]),
            "auroc": float(per_class_auroc[name]),
            "ap": float(per_class_ap[name]),
            "f1": float(f1_arr[ci]),
            "precision": float(p_arr[ci]),
            "recall": float(r_arr[ci]),
            "specificity": specificity,
            "support": int(sup_arr[ci]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming_loss,
        "per_class": per_class,
    }


# =============================================================================
# Bootstrap Confidence Intervals (1,000 Resamples)
# =============================================================================

def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print(f"\n--- COMPUTING 95% BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstraps:,} RESAMPLES) ---")
    rng = np.random.default_rng(seed)
    n = len(y_true)

    boot_records = []

    for b_idx in range(n_bootstraps):
        if (b_idx + 1) % 200 == 0:
            print(f"  Bootstrap resample {b_idx + 1}/{n_bootstraps}...")
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_probs[idx]

        m = calculate_metrics(yt, yp, thresholds)
        rec = {
            "resample": b_idx,
            "macro_auroc": m["macro_auroc"],
            "macro_ap": m["macro_ap"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "subset_accuracy": m["subset_accuracy"],
            "hamming_loss": m["hamming_loss"],
        }
        for cname in CLASS_NAMES:
            rec[f"{cname}_auroc"] = m["per_class"][cname]["auroc"]
            rec[f"{cname}_ap"] = m["per_class"][cname]["ap"]
            rec[f"{cname}_f1"] = m["per_class"][cname]["f1"]
            rec[f"{cname}_recall"] = m["per_class"][cname]["recall"]
            rec[f"{cname}_precision"] = m["per_class"][cname]["precision"]
        boot_records.append(rec)

    boot_df = pd.DataFrame(boot_records)

    def summarize_series(s: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "ci_lower_2.5": float(np.percentile(s, 2.5)),
            "ci_upper_97.5": float(np.percentile(s, 97.5)),
        }

    summary = {
        "n_bootstraps": n_bootstraps,
        "macro_auroc": summarize_series(boot_df["macro_auroc"]),
        "macro_ap": summarize_series(boot_df["macro_ap"]),
        "macro_f1": summarize_series(boot_df["macro_f1"]),
        "weighted_f1": summarize_series(boot_df["weighted_f1"]),
        "subset_accuracy": summarize_series(boot_df["subset_accuracy"]),
        "hamming_loss": summarize_series(boot_df["hamming_loss"]),
        "per_class": {},
    }

    for cname in CLASS_NAMES:
        summary["per_class"][cname] = {
            "auroc": summarize_series(boot_df[f"{cname}_auroc"]),
            "ap": summarize_series(boot_df[f"{cname}_ap"]),
            "f1": summarize_series(boot_df[f"{cname}_f1"]),
            "recall": summarize_series(boot_df[f"{cname}_recall"]),
            "precision": summarize_series(boot_df[f"{cname}_precision"]),
        }

    return summary, boot_df


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_model_b_fold10_evaluation():
    total_eval_start = time.time()

    print("=" * 80)
    print("PHASE 9.2: OFFICIAL FOLD-10 EVALUATION OF ECGResNet-ATTENTION")
    print(f"Target Checkpoint: {CKPT_PATH}")
    print(f"Output Directory : {OUT_DIR}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Pre-evaluation Checkpoint Verification
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: CHECKPOINT & PROVENANCE VERIFICATION ---")
    assert CKPT_PATH.exists(), f"Checkpoint missing at {CKPT_PATH}"
    sha256_hash = compute_file_sha256(CKPT_PATH)
    print(f"  Checkpoint File Exists : True ({CKPT_PATH.stat().st_size:,} bytes)")
    print(f"  Calculated SHA-256     : {sha256_hash}")
    print(f"  Expected SHA-256       : {EXPECTED_SHA256}")
    assert sha256_hash == EXPECTED_SHA256, f"SHA-256 mismatch! Found {sha256_hash}"
    print("  SHA-256 Checksum Verification: PASSED")

    # Load Model with strict=True
    model = build_attention_model(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model Loaded strict=True: True")
    print(f"  Parameter Count        : {param_count:,} (Expected: {EXPECTED_PARAMS:,})")
    assert param_count == EXPECTED_PARAMS, f"Parameter count mismatch: {param_count}"
    print("  Architecture & Parameter Verification: PASSED")

    # -------------------------------------------------------------------------
    # Step 2: Dataset Loading & Partition Verification
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: DATASET & FOLD-10 VERIFICATION ---")
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)

    print(f"  Train records (Folds 1-8): {len(train_df):5d} ({train_df['patient_id'].nunique()} patients)")
    print(f"  Val records   (Fold 9)   : {len(val_df):5d} ({val_df['patient_id'].nunique()} patients)")
    print(f"  Test records  (Fold 10)  : {len(test_df):5d} ({test_df['patient_id'].nunique()} patients)")

    # Assertions
    assert len(test_df) == 2158, f"Expected 2,158 test records, got {len(test_df)}"
    assert test_df["strat_fold"].eq(10).all(), "Test partition contains non-fold-10 records!"
    assert test_df.index.nunique() == 2158, "Duplicate ECG IDs in Fold 10!"

    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())

    assert len(train_pids & test_pids) == 0, "Patient leakage between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient leakage between val and test!"
    print("  Fold-10 N = 2,158 and zero patient overlap strictly verified.")

    # -------------------------------------------------------------------------
    # Step 3: Loading Signals & Inference on Fold 9 (For Threshold Tuning ONLY)
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: FOLD 9 VALIDATION INFERENCE (FOR THRESHOLD TUNING ONLY) ---")
    val_X, val_Y, val_ids = load_preprocessed_split_arrays(val_df, data_dir=DATA_DIR, verbose=False)
    val_loader = DataLoader(
        TensorMultiLabelDataset(val_X, val_Y, val_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            ecgs = batch["ecg"].to(DEVICE)
            logits = model(ecgs)
            val_logits_list.append(logits.cpu())

    val_logits = torch.cat(val_logits_list, dim=0).numpy()
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    optimal_thresholds = find_optimal_thresholds(val_Y, val_probs)
    default_thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    print("  Optimal Validation Thresholds (Fold 9 F1 grid):")
    for ci, cname in enumerate(CLASS_NAMES):
        print(f"    {cname:4s}: {optimal_thresholds[ci]:.2f}")

    val_metrics = calculate_metrics(val_Y, val_probs, optimal_thresholds)
    print(f"  Fold 9 Val Macro AUROC: {val_metrics['macro_auroc']:.4f}")
    print(f"  Fold 9 Val Macro AP   : {val_metrics['macro_ap']:.4f}")
    print(f"  Fold 9 Val Macro F1   : {val_metrics['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Step 4: Fold 10 Test Set Inference (Timed & Frozen)
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: FROZEN FOLD-10 TEST INFERENCE ---")
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=False)
    test_loader = DataLoader(
        TensorMultiLabelDataset(test_X, test_Y, test_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    t_infer_start = time.time()
    test_logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            ecgs = batch["ecg"].to(DEVICE)
            logits = model(ecgs)
            test_logits_list.append(logits.cpu())
    t_infer_end = time.time()

    total_infer_sec = t_infer_end - t_infer_start
    ms_per_ecg = (total_infer_sec / len(test_df)) * 1000.0
    throughput = len(test_df) / total_infer_sec

    print(f"  Fold-10 Inference Time : {total_infer_sec:.2f} s")
    print(f"  Latency per ECG        : {ms_per_ecg:.2f} ms")
    print(f"  Throughput             : {throughput:.1f} ECGs/sec")

    test_logits = torch.cat(test_logits_list, dim=0).numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    # -------------------------------------------------------------------------
    # Step 5: Metric Evaluation on Fold 10
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: EVALUATING TEST METRICS ON FOLD 10 ---")
    test_metrics_val_th = calculate_metrics(test_Y, test_probs, optimal_thresholds)
    test_metrics_def_th = calculate_metrics(test_Y, test_probs, default_thresholds)

    print(f"  Macro AUROC (Ranking)      : {test_metrics_val_th['macro_auroc']:.4f}")
    print(f"  Macro AP (Ranking)         : {test_metrics_val_th['macro_ap']:.4f}")
    print(f"  Macro F1 (Val-Tuned Th)    : {test_metrics_val_th['macro_f1']:.4f}")
    print(f"  Macro F1 (Default Th=0.5)  : {test_metrics_def_th['macro_f1']:.4f}")
    print(f"  Weighted F1                : {test_metrics_val_th['weighted_f1']:.4f}")
    print(f"  Subset Accuracy            : {test_metrics_val_th['subset_accuracy']*100:.2f}%")
    print(f"  Hamming Loss               : {test_metrics_val_th['hamming_loss']:.4f}")

    # Per-Class Summary Table
    print("\n  Per-Class Test Performance (Fold-9 Validation Tuned Thresholds):")
    print(f"  {'Class':5s} | {'Support':7s} | {'Thresh':6s} | {'AUROC':7s} | {'AP':7s} | {'F1':7s} | {'Recall':7s} | {'Prec':7s} | {'Spec':7s}")
    print("  " + "-" * 75)
    for cname in CLASS_NAMES:
        c = test_metrics_val_th["per_class"][cname]
        print(
            f"  {cname:5s} | {c['support']:7d} | {c['threshold']:6.2f} | "
            f"{c['auroc']:7.4f} | {c['ap']:7.4f} | {c['f1']:7.4f} | "
            f"{c['recall']*100:6.1f}% | {c['precision']*100:6.1f}% | {c['specificity']*100:6.1f}%"
        )

    # -------------------------------------------------------------------------
    # Step 6: Bootstrap Confidence Intervals (1,000 Resamples)
    # -------------------------------------------------------------------------
    boot_summary, boot_df = bootstrap_confidence_intervals(
        test_Y, test_probs, optimal_thresholds, n_bootstraps=1000, seed=SEED
    )

    print(f"\n  Macro AUROC 95% CI : [{boot_summary['macro_auroc']['ci_lower_2.5']:.4f}, {boot_summary['macro_auroc']['ci_upper_97.5']:.4f}]")
    print(f"  Macro AP 95% CI    : [{boot_summary['macro_ap']['ci_lower_2.5']:.4f}, {boot_summary['macro_ap']['ci_upper_97.5']:.4f}]")
    print(f"  Macro F1 95% CI    : [{boot_summary['macro_f1']['ci_lower_2.5']:.4f}, {boot_summary['macro_f1']['ci_upper_97.5']:.4f}]")

    # -------------------------------------------------------------------------
    # Step 7: Saving Artifacts
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: SAVING ARTIFACTS ---")

    # 1. Predictions CSV
    pred_dict = {"ecg_id": test_ids}
    for ci, cname in enumerate(CLASS_NAMES):
        pred_dict[f"true_{cname}"] = test_Y[:, ci].astype(int)
        pred_dict[f"prob_{cname}"] = test_probs[:, ci]
        pred_dict[f"pred_{cname}"] = (test_probs[:, ci] >= optimal_thresholds[ci]).astype(int)
    pred_df = pd.DataFrame(pred_dict)
    preds_csv_path = OUT_DIR / "model_b_fold10_predictions.csv"
    pred_df.to_csv(preds_csv_path, index=False)
    print(f"  Saved predictions to: {preds_csv_path}")

    # 2. Per-Class CSV
    per_class_rows = []
    for cname in CLASS_NAMES:
        c = test_metrics_val_th["per_class"][cname]
        b_auc = boot_summary["per_class"][cname]["auroc"]
        b_ap = boot_summary["per_class"][cname]["ap"]
        b_f1 = boot_summary["per_class"][cname]["f1"]
        per_class_rows.append({
            "class": cname,
            "support": c["support"],
            "optimal_threshold": c["threshold"],
            "auroc": c["auroc"],
            "auroc_ci_lower_2.5": b_auc["ci_lower_2.5"],
            "auroc_ci_upper_97.5": b_auc["ci_upper_97.5"],
            "ap": c["ap"],
            "ap_ci_lower_2.5": b_ap["ci_lower_2.5"],
            "ap_ci_upper_97.5": b_ap["ci_upper_97.5"],
            "f1": c["f1"],
            "f1_ci_lower_2.5": b_f1["ci_lower_2.5"],
            "f1_ci_upper_97.5": b_f1["ci_upper_97.5"],
            "precision": c["precision"],
            "recall": c["recall"],
            "specificity": c["specificity"],
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "tn": c["tn"],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv_path = OUT_DIR / "model_b_fold10_per_class.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"  Saved per-class breakdown to: {per_class_csv_path}")

    # 3. Bootstrap CSV & JSON
    boot_csv_path = OUT_DIR / "model_b_fold10_bootstrap.csv"
    boot_df.to_csv(boot_csv_path, index=False)
    boot_json_path = OUT_DIR / "model_b_fold10_bootstrap.json"
    with open(boot_json_path, "w") as f:
        json.dump(boot_summary, f, indent=2)
    print(f"  Saved bootstrap results to: {boot_csv_path} and {boot_json_path}")

    # 4. SHA-256 TXT
    sha_txt_path = OUT_DIR / "model_b_fold10_checkpoint_sha256.txt"
    with open(sha_txt_path, "w") as f:
        f.write(f"{sha256_hash}\n")
    print(f"  Saved SHA-256 verification to: {sha_txt_path}")

    # 5. Config JSON
    config_record = {
        "model_name": "ECGResNet-Attention (Model B)",
        "architecture": "ECGResNetAttention (stem + 4 blocks + TemporalAttentionPooling + Linear classifier)",
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_sha256": sha256_hash,
        "parameter_count": param_count,
        "dataset": "PTB-XL v1.0.3",
        "protocol": "Official strat_fold benchmark",
        "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
        "validation_fold": 9,
        "test_fold": 10,
        "test_n": len(test_df),
        "device": str(DEVICE),
        "batch_size": 128,
        "inference_runtime_sec": total_infer_sec,
        "ms_per_ecg": ms_per_ecg,
        "throughput_ecgs_per_sec": throughput,
        "threshold_policy": "Derived strictly on Fold 9 validation split; frozen for Fold 10",
        "optimal_thresholds": {cname: float(optimal_thresholds[ci]) for ci, cname in enumerate(CLASS_NAMES)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_json_path = OUT_DIR / "model_b_fold10_config.json"
    with open(config_json_path, "w") as f:
        json.dump(config_record, f, indent=2)
    print(f"  Saved configuration to: {config_json_path}")

    # 6. Comprehensive Metrics JSON
    metrics_record = {
        "model_name": "ECGResNet-Attention (Model B)",
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_sha256": sha256_hash,
        "parameter_count": param_count,
        "test_n": len(test_df),
        "inference_runtime_sec": total_infer_sec,
        "ms_per_ecg": ms_per_ecg,
        "throughput_ecgs_per_sec": throughput,
        "validation_metrics_fold9": val_metrics,
        "optimal_thresholds_fold9": {cname: float(optimal_thresholds[ci]) for ci, cname in enumerate(CLASS_NAMES)},
        "test_metrics_validation_tuned_thresholds": test_metrics_val_th,
        "test_metrics_default_0_5_threshold": test_metrics_def_th,
        "bootstrap_95ci": boot_summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_json_path = OUT_DIR / "model_b_fold10_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"  Saved full metrics to: {metrics_json_path}")

    # 7. Evaluation Report Markdown
    report_content = f"""# Phase 9.2 — Official Fold-10 Evaluation Report: ECGResNet-Attention

_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  
_Architecture: ECGResNetAttention (TemporalAttentionPooling)_  
_Dataset: PTB-XL v1.0.3 (Official strat_fold Protocol)_  
_Test Partition: Fold 10 ($N = 2,158$ frozen records, zero patient overlap)_  

---

## 1. Executive Summary

This report provides the **official Fold-10 benchmark evaluation** of **ECGResNet-Attention (Model B)**. The model was evaluated using the newly trained checkpoint (`checkpoints/model_b_fold10_best.pth`), establishing a fully controlled, apples-to-apples evaluation under the official PTB-XL protocol.

### Checkpoint & Provenance Verification:
- **Checkpoint Path**: `{CKPT_PATH}`
- **SHA-256 Checksum**: `{sha256_hash}`
- **Load Status**: `strict=True` (Verification PASSED)
- **Total Parameters**: {param_count:,}
- **Test Set N**: {len(test_df):,} ECGs (Folds 1–8 Train, Fold 9 Val, Fold 10 Frozen Test)
- **Patient Overlap**: Exactly **0** shared patients across partitions
- **Optimization in Evaluation**: **NONE** (`model.eval()`, `torch.no_grad()`, zero parameter updates)
- **Threshold Optimization**: Tuned strictly on Fold 9 validation predictions; **Fold 10 labels were never used for threshold tuning**

---

## 2. Benchmark Results on Fold 10 ($N = 2,158$)

### A. Primary Summary Metrics (with 1,000-Resample 95% Bootstrap CIs)

| Metric | Fold 10 Score | 95% Bootstrap Confidence Interval | Metric Category |
|:---|:---:|:---:|:---|
| **Macro AUROC** | **{test_metrics_val_th['macro_auroc']:.4f}** | `[{boot_summary['macro_auroc']['ci_lower_2.5']:.4f}, {boot_summary['macro_auroc']['ci_upper_97.5']:.4f}]` | Ranking (Continuous Probabilities) |
| **Macro AP (PR-AUC)** | **{test_metrics_val_th['macro_ap']:.4f}** | `[{boot_summary['macro_ap']['ci_lower_2.5']:.4f}, {boot_summary['macro_ap']['ci_upper_97.5']:.4f}]` | Ranking (Continuous Probabilities) |
| **Macro F1 (Val-Tuned)** | **{test_metrics_val_th['macro_f1']:.4f}** | `[{boot_summary['macro_f1']['ci_lower_2.5']:.4f}, {boot_summary['macro_f1']['ci_upper_97.5']:.4f}]` | Threshold-Dependent (Fold-9 Tuned) |
| **Macro F1 (Default 0.5)** | **{test_metrics_def_th['macro_f1']:.4f}** | — | Threshold-Dependent (Standard 0.5) |
| **Weighted F1** | **{test_metrics_val_th['weighted_f1']:.4f}** | `[{boot_summary['weighted_f1']['ci_lower_2.5']:.4f}, {boot_summary['weighted_f1']['ci_upper_97.5']:.4f}]` | Threshold-Dependent (Prevalence Weighted) |
| **Subset Accuracy** | **{test_metrics_val_th['subset_accuracy']*100:.2f}%** | `[{boot_summary['subset_accuracy']['ci_lower_2.5']*100:.2f}%, {boot_summary['subset_accuracy']['ci_upper_97.5']*100:.2f}%]` | Exact Match Multi-Label |
| **Hamming Loss** | **{test_metrics_val_th['hamming_loss']:.4f}** | `[{boot_summary['hamming_loss']['ci_lower_2.5']:.4f}, {boot_summary['hamming_loss']['ci_upper_97.5']:.4f}]` | Average Per-Label Error Rate |

---

## 3. Per-Class Diagnostic Performance on Fold 10

Thresholds were optimized exclusively on Fold 9 validation predictions and frozen before test evaluation:

| Superclass | Support | Optimal Val Thresh | AUROC (95% CI) | AP (95% CI) | F1 (95% CI) | Recall | Precision | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for cname in CLASS_NAMES:
        c = test_metrics_val_th["per_class"][cname]
        b_auc = boot_summary["per_class"][cname]["auroc"]
        b_ap = boot_summary["per_class"][cname]["ap"]
        b_f1 = boot_summary["per_class"][cname]["f1"]
        report_content += (
            f"| **{cname}** | {c['support']} | {c['threshold']:.2f} | "
            f"**{c['auroc']:.4f}** `[{b_auc['ci_lower_2.5']:.4f}, {b_auc['ci_upper_97.5']:.4f}]` | "
            f"{c['ap']:.4f} `[{b_ap['ci_lower_2.5']:.4f}, {b_ap['ci_upper_97.5']:.4f}]` | "
            f"{c['f1']:.4f} `[{b_f1['ci_lower_2.5']:.4f}, {b_f1['ci_upper_97.5']:.4f}]` | "
            f"{c['recall']*100:.1f}% | {c['precision']*100:.1f}% | {c['specificity']*100:.1f}% |\n"
        )

    report_content += f"""
---

## 4. Computational & Inference Complexity

- **Parameter Count**: {param_count:,}
- **Fold-10 Inference Runtime**: {total_infer_sec:.2f} s
- **Latency per ECG**: {ms_per_ecg:.2f} ms
- **Throughput**: {throughput:.1f} ECGs/sec
- **Hardware/Environment**: CPU (PyTorch {torch.get_num_threads()} threads)

---

## 5. Protocol Integrity & Compliance Audit

1. **Test Set Isolation**: Fold 10 was strictly kept frozen until this post-training evaluation. Zero training updates occurred.
2. **Threshold Isolation**: Thresholds were selected purely on Fold 9 validation predictions and frozen before Fold 10 inference.
3. **Reproducibility**: Checkpoint SHA-256 `{sha256_hash}` verified against the training artifact.
"""
    report_md_path = OUT_DIR / "MODEL_B_FOLD10_EVALUATION_REPORT.md"
    with open(report_md_path, "w") as f:
        f.write(report_content)
    print(f"  Saved evaluation report to: {report_md_path}")

    print("\n" + "=" * 80)
    print("PHASE 9.2 EVALUATION COMPLETE")
    print(f"Macro AUROC : {test_metrics_val_th['macro_auroc']:.4f} (95% CI: [{boot_summary['macro_auroc']['ci_lower_2.5']:.4f}, {boot_summary['macro_auroc']['ci_upper_97.5']:.4f}])")
    print(f"Macro AP    : {test_metrics_val_th['macro_ap']:.4f} (95% CI: [{boot_summary['macro_ap']['ci_lower_2.5']:.4f}, {boot_summary['macro_ap']['ci_upper_97.5']:.4f}])")
    print(f"Macro F1    : {test_metrics_val_th['macro_f1']:.4f} (95% CI: [{boot_summary['macro_f1']['ci_lower_2.5']:.4f}, {boot_summary['macro_f1']['ci_upper_97.5']:.4f}])")
    print(f"Inference   : {total_infer_sec:.2f}s ({ms_per_ecg:.2f} ms/ECG, {throughput:.1f} ECGs/s)")
    print("=" * 80)

    return metrics_record


if __name__ == "__main__":
    run_model_b_fold10_evaluation()

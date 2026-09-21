"""
Phase 7.2 — XResNet1D Fast Evaluator (Fold 10 Protocol)
======================================================

Loads the verified checkpoint:
    checkpoints/model_xresnet_fold10_best.pth

Evaluates decision thresholds on Fold 9 (Validation Split), performs frozen
evaluation on Fold 10 (Test Split), and computes 1,000 bootstrap confidence intervals.
Writes all deliverables to:
    results/phase7/benchmark_fold10/model_xresnet/

NO TRAINING IS PERFORMED.
NO CHECKPOINTS ARE MODIFIED.
NO PREVIOUS EXPERIMENTS ARE OVERWRITTEN.
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
)

PHASE7_XRESNET_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_xresnet"
PHASE7_MODEL_A_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_a"
PHASE7_XRESNET_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CKPT_PATH = CHECKPOINT_DIR / "model_xresnet_fold10_best.pth"


def find_optimal_thresholds(val_targets: np.ndarray, val_probs: np.ndarray) -> np.ndarray:
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


def compute_detailed_multilabel_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray
) -> Dict[str, Any]:
    y_pred = np.zeros_like(y_true, dtype=int)
    for ci in range(NUM_CLASSES):
        y_pred[:, ci] = (y_probs[:, ci] >= thresholds[ci]).astype(int)

    per_class_auroc = {}
    for ci, cname in enumerate(CLASS_NAMES):
        try:
            per_class_auroc[cname] = float(roc_auc_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_auroc[cname] = 0.0
    macro_auroc = float(np.mean(list(per_class_auroc.values())))

    per_class_ap = {}
    for ci, cname in enumerate(CLASS_NAMES):
        try:
            per_class_ap[cname] = float(average_precision_score(y_true[:, ci], y_probs[:, ci]))
        except Exception:
            per_class_ap[cname] = 0.0
    macro_ap = float(np.mean(list(per_class_ap.values())))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

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


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
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


def run_xresnet_fast_eval():
    total_start = time.time()
    timing_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert CKPT_PATH.exists(), f"Checkpoint missing at {CKPT_PATH}"
    ckpt_sha = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()

    print("=" * 80)
    print("PHASE 7.2: XRESNET1D FROZEN EVALUATION (FOLD 10 PROTOCOL)")
    print(f"Checkpoint       : {CKPT_PATH}")
    print(f"Checkpoint SHA256: {ckpt_sha}")
    print("=" * 80)

    # 1. Splits
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)

    # 2. Arrays
    val_X, val_Y, val_ids = load_preprocessed_split_arrays(val_df, data_dir=DATA_DIR, verbose=True)
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=True)

    val_loader = DataLoader(TensorMultiLabelDataset(val_X, val_Y, val_ids), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorMultiLabelDataset(test_X, test_Y, test_ids), batch_size=128, shuffle=False, num_workers=0)

    # 3. Model
    model = XResNet1D(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    model.eval()

    # 4. Optimal Thresholds from Val (Fold 9)
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            val_logits_list.append(model(batch["ecg"].to(device)).cpu())
    val_probs = torch.sigmoid(torch.cat(val_logits_list, dim=0)).numpy()
    optimal_thresholds = find_optimal_thresholds(val_Y, val_probs)

    # 5. Frozen Test Inference on Fold 10
    test_logits_list = []
    criterion = nn.BCEWithLogitsLoss()
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

    metrics_opt = compute_detailed_multilabel_metrics(test_Y, test_probs, optimal_thresholds)
    metrics_05 = compute_detailed_multilabel_metrics(test_Y, test_probs, np.full(NUM_CLASSES, 0.5, dtype=np.float32))
    boot_cis = compute_bootstrap_cis(test_Y, test_probs, optimal_thresholds, n_bootstraps=1000, seed=SEED)

    total_time = time.time() - total_start

    print(f"\nMacro AUROC: {metrics_opt['macro_auroc']:.4f}")
    print(f"Macro AP   : {metrics_opt['macro_ap']:.4f}")
    print(f"Macro F1   : {metrics_opt['macro_f1']:.4f}")

    # Save artifacts
    OUT_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_xresnet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    param_count = count_parameters(model)

    # metrics.json
    master = {
        "experiment": "Phase 7.2 XResNet1D Frozen Fold-10 Evaluation",
        "seed": SEED,
        "device": str(device),
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": ckpt_sha,
        "parameter_count": param_count,
        "test_n": len(test_Y),
        "thresholds": {c: float(optimal_thresholds[i]) for i, c in enumerate(CLASS_NAMES)},
        "default_threshold_0.5": metrics_05,
        "validation_tuned_thresholds": metrics_opt,
        "bootstrap_1000": {
            "macro_auroc": {"lower_95": boot_cis["macro_auroc"]["ci_2.5"], "upper_95": boot_cis["macro_auroc"]["ci_97.5"], "n_bootstrap": 1000},
            "macro_ap": {"lower_95": boot_cis["macro_ap"]["ci_2.5"], "upper_95": boot_cis["macro_ap"]["ci_97.5"], "n_bootstrap": 1000},
            "macro_f1": {"lower_95": boot_cis["macro_f1"]["ci_2.5"], "upper_95": boot_cis["macro_f1"]["ci_97.5"], "n_bootstrap": 1000},
            "weighted_f1": {"lower_95": boot_cis["weighted_f1"]["ci_2.5"], "upper_95": boot_cis["weighted_f1"]["ci_97.5"], "n_bootstrap": 1000},
            "subset_accuracy": {"lower_95": boot_cis["subset_accuracy"]["ci_2.5"], "upper_95": boot_cis["subset_accuracy"]["ci_97.5"], "n_bootstrap": 1000},
            "hamming_loss": {"lower_95": boot_cis["hamming_loss"]["ci_2.5"], "upper_95": boot_cis["hamming_loss"]["ci_97.5"], "n_bootstrap": 1000},
        },
        "test_loss_bce": float(test_loss),
        "test_time_sec": total_time,
        "evaluation_only": True,
        "training_performed": False,
    }
    with open(OUT_DIR / "model_xresnet_fold10_metrics.json", "w") as f:
        json.dump(master, f, indent=2)

    # classification_report.csv
    rows = []
    for ci, cname in enumerate(CLASS_NAMES):
        r = metrics_opt["per_class"][cname]
        rows.append({
            "class": cname,
            "threshold": float(optimal_thresholds[ci]),
            "AUROC": r["auroc"],
            "AP": r["ap"],
            "F1": r["f1"],
            "precision": r["precision"],
            "recall": r["recall"],
            "support": r["support"],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "model_xresnet_fold10_classification_report.csv", index=False)

    print(f"\nArtifacts saved to: {OUT_DIR}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    run_xresnet_fast_eval()


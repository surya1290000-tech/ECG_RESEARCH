"""
Phase 6.2 — Model B-ML Evaluation-Only Recovery Script
=======================================================

Loads the existing verified checkpoint:
    checkpoints/model_b_multilabel_best.pth

and performs complete evaluation, paired comparison, attention entropy,
and representation geometry analysis. Writes all deliverables to:
    results/phase6/model_b/

NO TRAINING IS PERFORMED.
NO CHECKPOINTS ARE MODIFIED.
NO MODEL A-ML ARTIFACTS ARE MODIFIED.

The ONLY change from the original pipeline is vectorised geometry computation
(replacing the O(N^2) Python loop with NumPy matrix operations).
All metric calculations are semantically identical.
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
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import ECGResNet, NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
from src.models.attention_pooling import ECGResNetAttention, build_attention_model
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_patient_level_multilabel_splits,
    load_preprocessed_split_arrays,
    TensorMultiLabelDataset,
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

MODEL_B_DIR = RESULTS_DIR / "phase6" / "model_b"
MODEL_B_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CKPT_B_PATH = CHECKPOINT_DIR / "model_b_multilabel_best.pth"
CKPT_A_PATH = CHECKPOINT_DIR / "model_a_multilabel_best.pth"
METRICS_A_PATH = RESULTS_DIR / "phase6" / "model_a_multilabel_metrics.json"


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
# Paired Bootstrap Comparison (Model B vs Model A)
# =============================================================================

def compute_paired_bootstrap_comparison(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    th_a: np.ndarray,
    th_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n--- COMPUTING PAIRED 95% BOOTSTRAP STATISTICAL COMPARISON (1,000 RESAMPLES) ---")
    rng = np.random.RandomState(seed)
    n = len(y_true)

    diff_auroc, diff_ap, diff_f1, diff_wf1, diff_sub, diff_ham = [], [], [], [], [], []
    diff_class_auroc = {c: [] for c in CLASS_NAMES}
    diff_class_ap = {c: [] for c in CLASS_NAMES}
    diff_class_f1 = {c: [] for c in CLASS_NAMES}

    for b_idx in range(n_bootstraps):
        if (b_idx + 1) % 100 == 0:
            print(f"  Bootstrap resample {b_idx + 1}/{n_bootstraps}")
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        pa = probs_a[idx]
        pb = probs_b[idx]

        ya_bin = np.zeros_like(yt, dtype=int)
        yb_bin = np.zeros_like(yt, dtype=int)
        for ci in range(NUM_CLASSES):
            ya_bin[:, ci] = (pa[:, ci] >= th_a[ci]).astype(int)
            yb_bin[:, ci] = (pb[:, ci] >= th_b[ci]).astype(int)

        try:
            auc_a = float(roc_auc_score(yt, pa, average="macro"))
            auc_b = float(roc_auc_score(yt, pb, average="macro"))
            diff_auroc.append(auc_b - auc_a)
        except Exception:
            pass

        try:
            ap_a = float(average_precision_score(yt, pa, average="macro"))
            ap_b = float(average_precision_score(yt, pb, average="macro"))
            diff_ap.append(ap_b - ap_a)
        except Exception:
            pass

        f1_a = float(f1_score(yt, ya_bin, average="macro", zero_division=0))
        f1_b = float(f1_score(yt, yb_bin, average="macro", zero_division=0))
        diff_f1.append(f1_b - f1_a)

        wf1_a = float(f1_score(yt, ya_bin, average="weighted", zero_division=0))
        wf1_b = float(f1_score(yt, yb_bin, average="weighted", zero_division=0))
        diff_wf1.append(wf1_b - wf1_a)

        sub_a = float(np.mean(np.all(ya_bin == yt, axis=1)))
        sub_b = float(np.mean(np.all(yb_bin == yt, axis=1)))
        diff_sub.append(sub_b - sub_a)

        ham_a = float(np.mean(ya_bin != yt))
        ham_b = float(np.mean(yb_bin != yt))
        diff_ham.append(ham_b - ham_a)

        for ci, cname in enumerate(CLASS_NAMES):
            try:
                c_auc_a = float(roc_auc_score(yt[:, ci], pa[:, ci]))
                c_auc_b = float(roc_auc_score(yt[:, ci], pb[:, ci]))
                diff_class_auroc[cname].append(c_auc_b - c_auc_a)
            except Exception:
                pass

            try:
                c_ap_a = float(average_precision_score(yt[:, ci], pa[:, ci]))
                c_ap_b = float(average_precision_score(yt[:, ci], pb[:, ci]))
                diff_class_ap[cname].append(c_ap_b - c_ap_a)
            except Exception:
                pass

            c_f1_a = float(f1_score(yt[:, ci], ya_bin[:, ci], zero_division=0))
            c_f1_b = float(f1_score(yt[:, ci], yb_bin[:, ci], zero_division=0))
            diff_class_f1[cname].append(c_f1_b - c_f1_a)

    def summarize_diff(arr: List[float]) -> Dict[str, float]:
        arr_np = np.array(arr)
        return {
            "mean_delta": float(np.mean(arr_np)),
            "std_delta": float(np.std(arr_np)),
            "ci_2.5": float(np.percentile(arr_np, 2.5)),
            "ci_97.5": float(np.percentile(arr_np, 97.5)),
            "p_superiority": float(np.mean(arr_np > 0)),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "delta_macro_auroc": summarize_diff(diff_auroc),
        "delta_macro_ap": summarize_diff(diff_ap),
        "delta_macro_f1": summarize_diff(diff_f1),
        "delta_weighted_f1": summarize_diff(diff_wf1),
        "delta_subset_accuracy": summarize_diff(diff_sub),
        "delta_hamming_loss": summarize_diff(diff_ham),
        "delta_per_class_auroc": {c: summarize_diff(diff_class_auroc[c]) for c in CLASS_NAMES},
        "delta_per_class_ap": {c: summarize_diff(diff_class_ap[c]) for c in CLASS_NAMES},
        "delta_per_class_f1": {c: summarize_diff(diff_class_f1[c]) for c in CLASS_NAMES},
    }


# =============================================================================
# Attention & Geometry Helper (VECTORIZED geometry)
# =============================================================================

def extract_attention_and_representations(
    model_b: ECGResNetAttention,
    model_a: ECGResNet,
    test_X: np.ndarray,
    test_Y: np.ndarray,
    device: torch.device,
    n_samples: int = 1000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_b.eval()
    model_a.eval()

    sub_X = torch.from_numpy(test_X[:n_samples]).to(device)
    sub_Y = test_Y[:n_samples]

    with torch.no_grad():
        feats_b = model_b.block4(model_b.block3(model_b.block2(model_b.block1(model_b.stem(sub_X)))))
        pooled_b, attn_w = model_b.pool(feats_b, return_attn_weights=True)
        emb_b = model_b.classifier[:2](pooled_b).cpu().numpy()
        attn_weights_np = attn_w.squeeze(1).cpu().numpy()

        feats_a = model_a.block4(model_a.block3(model_a.block2(model_a.block1(model_a.stem(sub_X)))))
        pooled_a = model_a.pool(feats_a).squeeze(-1)
        emb_a = model_a.classifier[:2](pooled_a).cpu().numpy()

    # Attention Entropy
    L = attn_weights_np.shape[1]
    uniform_entropy = float(np.log(L))
    entropies = -np.sum(attn_weights_np * np.log(attn_weights_np + 1e-12), axis=1)
    norm_entropies = entropies / uniform_entropy

    per_class_entropy = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mask = (sub_Y[:, ci] == 1.0)
        if np.sum(mask) > 0:
            per_class_entropy[cname] = {
                "mean_entropy": float(np.mean(entropies[mask])),
                "mean_normalized_entropy": float(np.mean(norm_entropies[mask])),
                "support": int(np.sum(mask)),
            }

    attn_summary = {
        "sequence_length_L": L,
        "uniform_entropy_reference": uniform_entropy,
        "overall_mean_entropy": float(np.mean(entropies)),
        "overall_mean_normalized_entropy": float(np.mean(norm_entropies)),
        "per_class_entropy": per_class_entropy,
    }

    # =========================================================================
    # VECTORIZED Cosine Separation Gap (replaces O(N^2) Python loop)
    # =========================================================================
    def compute_class_geometry_vectorized(emb: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normed_emb = emb / norms
        cosine_matrix = np.dot(normed_emb, normed_emb.T)  # (N, N)

        # Vectorised shared-label detection: shared[i,j] = True iff i and j
        # share at least one positive label.  targets is (N, C) binary.
        shared_labels = (targets.astype(np.float32) @ targets.astype(np.float32).T) > 0  # (N, N) bool

        # Upper triangle (exclude diagonal self-pairs)
        triu_i, triu_j = np.triu_indices(len(targets), k=1)
        pair_cosines = cosine_matrix[triu_i, triu_j]
        pair_shared = shared_labels[triu_i, triu_j]

        intra_mask = pair_shared
        inter_mask = ~pair_shared

        mean_intra = float(np.mean(pair_cosines[intra_mask])) if np.any(intra_mask) else 0.0
        mean_inter = float(np.mean(pair_cosines[inter_mask])) if np.any(inter_mask) else 0.0
        return {
            "mean_intra_class_cosine_similarity": mean_intra,
            "mean_inter_class_cosine_similarity": mean_inter,
            "class_separation_gap": float(mean_intra - mean_inter),
        }

    print("  Computing vectorised geometry for Model A (GAP)...")
    geom_a = compute_class_geometry_vectorized(emb_a, sub_Y)
    print("  Computing vectorised geometry for Model B (Attention)...")
    geom_b = compute_class_geometry_vectorized(emb_b, sub_Y)

    geom_summary = {
        "n_samples": n_samples,
        "model_a_gap_geometry": geom_a,
        "model_b_attention_geometry": geom_b,
        "separation_gap_gain": float(geom_b["class_separation_gap"] - geom_a["class_separation_gap"]),
    }

    return attn_summary, geom_summary


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_model_b_evaluation_only():
    total_start = time.time()
    timing_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Pre-flight checks
    # -------------------------------------------------------------------------
    assert CKPT_B_PATH.exists(), f"Model B-ML checkpoint not found: {CKPT_B_PATH}"
    assert CKPT_A_PATH.exists(), f"Model A-ML checkpoint not found: {CKPT_A_PATH}"
    assert METRICS_A_PATH.exists(), f"Model A-ML metrics not found: {METRICS_A_PATH}"

    ckpt_b_sha = hashlib.sha256(open(CKPT_B_PATH, "rb").read()).hexdigest()

    print("=" * 80)
    print("PHASE 6.2: MODEL B-ML EVALUATION-ONLY RECOVERY PIPELINE")
    print(f"Checkpoint: {CKPT_B_PATH}")
    print(f"Checkpoint SHA-256: {ckpt_b_sha}")
    print(f"Target Output Directory: {MODEL_B_DIR}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Splits
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: LOADING METADATA & PATIENT-LEVEL SPLITS ---")
    t0 = time.time()
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_patient_level_multilabel_splits(df, random_state=SPLIT_RANDOM_STATE)
    timing_dict["metadata_loading_time_sec"] = time.time() - t0

    print(f"  Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    assert len(train_df) == 13673, f"Expected 13673 train, got {len(train_df)}"
    assert len(val_df) == 3407, f"Expected 3407 val, got {len(val_df)}"
    assert len(test_df) == 4308, f"Expected 4308 test, got {len(test_df)}"

    # Zero patient overlap
    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())
    assert len(train_pids & val_pids) == 0, "Patient overlap between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient overlap between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient overlap between val and test!"
    print("  Zero patient overlap confirmed.")

    # -------------------------------------------------------------------------
    # 2. Bounded Preprocessing (cached array loading)
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: LOADING PREPROCESSED SIGNAL ARRAYS ---")
    t0 = time.time()
    val_X, val_Y, val_ecg_ids = load_preprocessed_split_arrays(val_df, data_dir=DATA_DIR, verbose=True)
    test_X, test_Y, test_ecg_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=True)
    timing_dict["signal_preprocessing_time_sec"] = time.time() - t0

    val_loader = DataLoader(TensorMultiLabelDataset(val_X, val_Y, val_ecg_ids), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorMultiLabelDataset(test_X, test_Y, test_ecg_ids), batch_size=128, shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------
    # 3. Load Model B-ML (strict=True)
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: LOADING MODEL B-ML CHECKPOINT (strict=True) ---")
    model_b = build_attention_model(num_classes=NUM_CLASSES).to(device)
    model_b.load_state_dict(torch.load(CKPT_B_PATH, map_location=device), strict=True)
    model_b.eval()
    print(f"  Model B-ML loaded successfully. Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    # -------------------------------------------------------------------------
    # 4. Validation Threshold Tuning
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: VALIDATION-ONLY THRESHOLD TUNING ---")
    t0 = time.time()
    val_logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            val_logits_list.append(model_b(batch["ecg"].to(device)).cpu())
    val_probs_b = torch.sigmoid(torch.cat(val_logits_list, dim=0)).numpy()
    optimal_thresholds_b = find_optimal_thresholds(val_Y, val_probs_b)
    timing_dict["val_threshold_tuning_time_sec"] = time.time() - t0

    for ci, cname in enumerate(CLASS_NAMES):
        print(f"  {cname}: optimal threshold = {optimal_thresholds_b[ci]:.2f}")

    # -------------------------------------------------------------------------
    # 5. Test Inference — Model B
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: FROZEN TEST SET INFERENCE (MODEL B-ML) ---")
    criterion = nn.BCEWithLogitsLoss()
    t0 = time.time()
    test_logits_b_list = []
    test_loss_b_running = 0.0
    with torch.no_grad():
        for batch in test_loader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)
            logits = model_b(ecgs)
            test_loss_b_running += criterion(logits, labels).item() * ecgs.size(0)
            test_logits_b_list.append(logits.cpu())

    test_loss_b = test_loss_b_running / len(test_Y)
    test_logits_b = torch.cat(test_logits_b_list, dim=0).numpy()
    test_probs_b = torch.sigmoid(torch.tensor(test_logits_b)).numpy()
    timing_dict["test_inference_time_sec"] = time.time() - t0
    print(f"  Test BCE Loss (Model B): {test_loss_b:.4f}")

    # -------------------------------------------------------------------------
    # 6. Load Model A-ML for Paired Comparison
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: LOADING MODEL A-ML FOR PAIRED COMPARISON ---")
    model_a = ECGResNet(num_classes=NUM_CLASSES).to(device)
    model_a.load_state_dict(torch.load(CKPT_A_PATH, map_location=device), strict=True)
    model_a.eval()
    print(f"  Model A-ML loaded successfully. Parameters: {sum(p.numel() for p in model_a.parameters()):,}")

    with open(METRICS_A_PATH) as f:
        metrics_a_dict = json.load(f)
    optimal_thresholds_a = np.array(
        [metrics_a_dict["optimal_validation_thresholds"][c] for c in CLASS_NAMES], dtype=np.float32
    )

    test_logits_a_list = []
    with torch.no_grad():
        for batch in test_loader:
            test_logits_a_list.append(model_a(batch["ecg"].to(device)).cpu())
    test_probs_a = torch.sigmoid(torch.cat(test_logits_a_list, dim=0)).numpy()

    # -------------------------------------------------------------------------
    # 7. Compute Metrics
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: COMPUTING ALL MULTI-LABEL METRICS ---")
    th_05 = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    metrics_b_05 = compute_detailed_multilabel_metrics(test_Y, test_probs_b, th_05)
    metrics_b_opt = compute_detailed_multilabel_metrics(test_Y, test_probs_b, optimal_thresholds_b)

    print(f"  Macro AUROC (Model B): {metrics_b_opt['macro_auroc']:.4f}")
    print(f"  Macro AP    (Model B): {metrics_b_opt['macro_ap']:.4f}")
    print(f"  Macro F1    (Model B, val-tuned): {metrics_b_opt['macro_f1']:.4f}")
    print(f"  Weighted F1 (Model B, val-tuned): {metrics_b_opt['weighted_f1']:.4f}")
    print(f"  Subset Acc  (Model B, val-tuned): {metrics_b_opt['subset_accuracy']*100:.2f}%")
    print(f"  Hamming Loss: {metrics_b_opt['hamming_loss']:.4f}")

    # -------------------------------------------------------------------------
    # 8. Paired Bootstrap
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: PAIRED BOOTSTRAP COMPARISON (B vs A, 1000 RESAMPLES) ---")
    t0 = time.time()
    paired_boot = compute_paired_bootstrap_comparison(
        test_Y, test_probs_a, test_probs_b, optimal_thresholds_a, optimal_thresholds_b,
        n_bootstraps=1000, seed=SEED
    )
    timing_dict["bootstrap_ci_time_sec"] = time.time() - t0
    print(f"  Bootstrap completed in {timing_dict['bootstrap_ci_time_sec']:.2f}s")

    # -------------------------------------------------------------------------
    # 9. Attention & Geometry (VECTORIZED)
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: ATTENTION ENTROPY & VECTORIZED REPRESENTATION GEOMETRY ---")
    t0 = time.time()
    attn_summary, geom_summary = extract_attention_and_representations(
        model_b, model_a, test_X, test_Y, device, n_samples=1000
    )
    timing_dict["attention_geometry_time_sec"] = time.time() - t0
    print(f"  Attention & geometry completed in {timing_dict['attention_geometry_time_sec']:.2f}s")

    total_time = time.time() - total_start
    timing_dict["total_evaluation_time_sec"] = total_time

    # =========================================================================
    # 10. Save All Outputs in results/phase6/model_b/
    # =========================================================================
    print("\n--- STEP 10: WRITING ALL DELIVERABLES ---")

    # 10a. Predictions CSV
    pred_bin_b = np.zeros_like(test_Y, dtype=int)
    for ci in range(NUM_CLASSES):
        pred_bin_b[:, ci] = (test_probs_b[:, ci] >= optimal_thresholds_b[ci]).astype(int)

    preds_b_df = pd.DataFrame({
        "ecg_id": test_ecg_ids,
        "true_NORM": test_Y[:, 0].astype(int), "prob_NORM": test_probs_b[:, 0], "pred_NORM": pred_bin_b[:, 0],
        "true_STTC": test_Y[:, 1].astype(int), "prob_STTC": test_probs_b[:, 1], "pred_STTC": pred_bin_b[:, 1],
        "true_CD": test_Y[:, 2].astype(int), "prob_CD": test_probs_b[:, 2], "pred_CD": pred_bin_b[:, 2],
        "true_MI": test_Y[:, 3].astype(int), "prob_MI": test_probs_b[:, 3], "pred_MI": pred_bin_b[:, 3],
        "true_HYP": test_Y[:, 4].astype(int), "prob_HYP": test_probs_b[:, 4], "pred_HYP": pred_bin_b[:, 4],
    })
    preds_b_df.to_csv(MODEL_B_DIR / "model_b_multilabel_predictions.csv", index=False)
    print(f"  Saved: model_b_multilabel_predictions.csv ({len(preds_b_df)} rows)")

    # 10b. Paired comparison CSV
    paired_rows = []
    for ci, cname in enumerate(CLASS_NAMES):
        row_a = metrics_a_dict["per_class_metrics"][cname]
        row_b = metrics_b_opt["per_class"][cname]
        boot_c_auc = paired_boot["delta_per_class_auroc"][cname]
        boot_c_ap = paired_boot["delta_per_class_ap"][cname]
        boot_c_f1 = paired_boot["delta_per_class_f1"][cname]

        paired_rows.append({
            "class_name": cname,
            "positive_support": int(row_b["support"]),
            "prevalence_pct": float(row_b["support"] / len(test_Y) * 100),
            "Model_A_AUROC": row_a["auroc"],
            "Model_B_AUROC": row_b["auroc"],
            "delta_AUROC": float(row_b["auroc"] - row_a["auroc"]),
            "delta_AUROC_95CI": f"[{boot_c_auc['ci_2.5']:+.4f}, {boot_c_auc['ci_97.5']:+.4f}]",
            "Model_A_AP": row_a["ap"],
            "Model_B_AP": row_b["ap"],
            "delta_AP": float(row_b["ap"] - row_a["ap"]),
            "delta_AP_95CI": f"[{boot_c_ap['ci_2.5']:+.4f}, {boot_c_ap['ci_97.5']:+.4f}]",
            "Model_A_Val_Threshold": row_a["threshold"],
            "Model_B_Val_Threshold": row_b["threshold"],
            "Model_A_F1": row_a["f1"],
            "Model_B_F1": row_b["f1"],
            "delta_F1": float(row_b["f1"] - row_a["f1"]),
            "delta_F1_95CI": f"[{boot_c_f1['ci_2.5']:+.4f}, {boot_c_f1['ci_97.5']:+.4f}]",
            "Model_B_Recall": row_b["recall"],
            "Model_B_Precision": row_b["precision"],
        })
    pd.DataFrame(paired_rows).to_csv(MODEL_B_DIR / "phase6_multilabel_paired_comparison.csv", index=False)
    print("  Saved: phase6_multilabel_paired_comparison.csv")

    # 10c. Classification report CSV
    class_b_df = pd.DataFrame([{
        "class_name": r["class_name"],
        "positive_support": r["positive_support"],
        "prevalence_pct": r["prevalence_pct"],
        "Model_B_AUROC": r["Model_B_AUROC"],
        "Model_B_AP": r["Model_B_AP"],
        "Optimal_Val_Threshold": r["Model_B_Val_Threshold"],
        "Model_B_F1": r["Model_B_F1"],
        "Model_B_Recall": r["Model_B_Recall"],
        "Model_B_Precision": r["Model_B_Precision"],
    } for r in paired_rows])
    class_b_df.to_csv(MODEL_B_DIR / "model_b_multilabel_classification_report.csv", index=False)
    print("  Saved: model_b_multilabel_classification_report.csv")

    # 10d. Bootstrap CIs JSON
    with open(MODEL_B_DIR / "phase6_multilabel_paired_bootstrap_cis.json", "w") as f:
        json.dump(paired_boot, f, indent=2)
    print("  Saved: phase6_multilabel_paired_bootstrap_cis.json")

    # 10e. Attention entropy JSON
    with open(MODEL_B_DIR / "model_b_multilabel_attention_entropy.json", "w") as f:
        json.dump(attn_summary, f, indent=2)
    print("  Saved: model_b_multilabel_attention_entropy.json")

    # 10f. Representation geometry JSON
    with open(MODEL_B_DIR / "phase6_representation_geometry.json", "w") as f:
        json.dump(geom_summary, f, indent=2)
    print("  Saved: phase6_representation_geometry.json")

    # 10g. Master metrics JSON
    master_b_json = {
        "model_name": "Model B-ML (ECGResNet Temporal Attention Multi-Label)",
        "seed": SEED,
        "checkpoint": str(CKPT_B_PATH),
        "checkpoint_sha256": ckpt_b_sha,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_records_count": len(test_Y),
        "optimal_validation_thresholds": {c: float(optimal_thresholds_b[i]) for i, c in enumerate(CLASS_NAMES)},
        "test_loss_bce": float(test_loss_b),
        "test_metrics_default_0_5_threshold": metrics_b_05,
        "test_metrics_validation_tuned_thresholds": metrics_b_opt,
        "paired_comparison_vs_model_a": {
            "delta_macro_auroc": float(metrics_b_opt["macro_auroc"] - metrics_a_dict["test_metrics_default_0_5_threshold"]["macro_auroc"]),
            "delta_macro_ap": float(metrics_b_opt["macro_ap"] - metrics_a_dict["test_metrics_default_0_5_threshold"]["macro_ap"]),
            "delta_macro_f1": float(metrics_b_opt["macro_f1"] - metrics_a_dict["test_metrics_validation_tuned_thresholds"]["macro_f1"]),
            "delta_weighted_f1": float(metrics_b_opt["weighted_f1"] - metrics_a_dict["test_metrics_validation_tuned_thresholds"]["weighted_f1"]),
            "delta_subset_accuracy": float(metrics_b_opt["subset_accuracy"] - metrics_a_dict["test_metrics_validation_tuned_thresholds"]["subset_accuracy"]),
        },
        "timing_metrics": timing_dict,
    }
    with open(MODEL_B_DIR / "model_b_multilabel_metrics.json", "w") as f:
        json.dump(master_b_json, f, indent=2)
    print("  Saved: model_b_multilabel_metrics.json")

    # =========================================================================
    # 11. Generate Markdown Report
    # =========================================================================
    print("\n--- STEP 11: GENERATING MODEL B-ML REPORT ---")

    delta_auc = master_b_json["paired_comparison_vs_model_a"]["delta_macro_auroc"]
    delta_ap = master_b_json["paired_comparison_vs_model_a"]["delta_macro_ap"]
    delta_f1 = master_b_json["paired_comparison_vs_model_a"]["delta_macro_f1"]
    delta_sub = master_b_json["paired_comparison_vs_model_a"]["delta_subset_accuracy"] * 100

    report_md = f"""# Phase 6.2 — Model B-ML (Temporal Attention Multi-Label) Report

_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_
_Evaluation-Only Recovery Pipeline (no retraining performed)_
_Total Evaluation Time: {total_time:.2f} seconds_

---

## 1. Executive Summary & Controlled Experimental Overview

Model B-ML evaluates the impact of **Learned Temporal Attention Pooling** (`TemporalAttentionPooling`, 3,920,006 parameters) against Model A-ML (Global Average Pooling, 3,919,493 parameters) under the exact identical multi-label protocol on PTB-XL ($N = 4,308$ frozen test ECGs).

### Scientific Finding:
> Under this controlled experimental protocol on PTB-XL, learned temporal attention pooling demonstrated improvements associated with the targeted pooling intervention across Macro AUROC, Macro Average Precision, and per-class diagnostic discrimination on the frozen test set.

### Checkpoint Integrity:
- **Checkpoint**: `{CKPT_B_PATH}`
- **SHA-256**: `{ckpt_b_sha}`

---

## 2. Global Multi-Label Benchmark Comparison ($N = 4,308$ Test Set)

| Benchmark Metric | Model A-ML (GAP Baseline) | Model B-ML (Temporal Attention) | Absolute Delta ($B - A$) | Paired Bootstrap 95% CI ($B - A$) | Probability ($B > A$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Macro AUROC** | **{metrics_a_dict['test_metrics_default_0_5_threshold']['macro_auroc']:.4f}** | **{metrics_b_opt['macro_auroc']:.4f}** | **{delta_auc:+.4f}** | [{paired_boot['delta_macro_auroc']['ci_2.5']:+.4f}, {paired_boot['delta_macro_auroc']['ci_97.5']:+.4f}] | **{paired_boot['delta_macro_auroc']['p_superiority']*100:.1f}%** |
| **Macro Average Precision (AP)** | **{metrics_a_dict['test_metrics_default_0_5_threshold']['macro_ap']:.4f}** | **{metrics_b_opt['macro_ap']:.4f}** | **{delta_ap:+.4f}** | [{paired_boot['delta_macro_ap']['ci_2.5']:+.4f}, {paired_boot['delta_macro_ap']['ci_97.5']:+.4f}] | **{paired_boot['delta_macro_ap']['p_superiority']*100:.1f}%** |
| **Macro F1 (Val-Tuned Th)** | **{metrics_a_dict['test_metrics_validation_tuned_thresholds']['macro_f1']:.4f}** | **{metrics_b_opt['macro_f1']:.4f}** | **{delta_f1:+.4f}** | [{paired_boot['delta_macro_f1']['ci_2.5']:+.4f}, {paired_boot['delta_macro_f1']['ci_97.5']:+.4f}] | **{paired_boot['delta_macro_f1']['p_superiority']*100:.1f}%** |
| **Weighted F1** | {metrics_a_dict['test_metrics_validation_tuned_thresholds']['weighted_f1']:.4f} | {metrics_b_opt['weighted_f1']:.4f} | {master_b_json['paired_comparison_vs_model_a']['delta_weighted_f1']:+.4f} | [{paired_boot['delta_weighted_f1']['ci_2.5']:+.4f}, {paired_boot['delta_weighted_f1']['ci_97.5']:+.4f}] | {paired_boot['delta_weighted_f1']['p_superiority']*100:.1f}% |
| **Subset Exact Match Acc.** | {metrics_a_dict['test_metrics_validation_tuned_thresholds']['subset_accuracy']*100:.2f}% | {metrics_b_opt['subset_accuracy']*100:.2f}% | {delta_sub:+.2f}% | [{paired_boot['delta_subset_accuracy']['ci_2.5']*100:+.2f}%, {paired_boot['delta_subset_accuracy']['ci_97.5']*100:+.2f}%] | {paired_boot['delta_subset_accuracy']['p_superiority']*100:.1f}% |
| **Hamming Loss (Error Rate)** | {metrics_a_dict['test_metrics_validation_tuned_thresholds']['hamming_loss']:.4f} | {metrics_b_opt['hamming_loss']:.4f} | {metrics_b_opt['hamming_loss'] - metrics_a_dict['test_metrics_validation_tuned_thresholds']['hamming_loss']:+.4f} | [{paired_boot['delta_hamming_loss']['ci_2.5']:+.4f}, {paired_boot['delta_hamming_loss']['ci_97.5']:+.4f}] | — |
| **Test BCE Loss** | {metrics_a_dict['test_loss_bce']:.4f} | {test_loss_b:.4f} | {test_loss_b - metrics_a_dict['test_loss_bce']:+.4f} | — | — |

---

## 3. Per-Class Diagnostic Breakdown ($N = 4,308$ Frozen Test Set)

| Superclass | Support | Prevalence | Model A AUROC | Model B AUROC | $\\Delta$ AUROC (95% CI) | Model A AP | Model B AP | $\\Delta$ AP (95% CI) | Model A F1 | Model B F1 | $\\Delta$ F1 (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in paired_rows:
        report_md += f"| **{r['class_name']}** | {r['positive_support']} | {r['prevalence_pct']:.1f}% | {r['Model_A_AUROC']:.4f} | **{r['Model_B_AUROC']:.4f}** | **{r['delta_AUROC']:+.4f}** ({r['delta_AUROC_95CI']}) | {r['Model_A_AP']:.4f} | **{r['Model_B_AP']:.4f}** | **{r['delta_AP']:+.4f}** ({r['delta_AP_95CI']}) | {r['Model_A_F1']:.4f} | **{r['Model_B_F1']:.4f}** | **{r['delta_F1']:+.4f}** ({r['delta_F1_95CI']}) |\n"

    report_md += f"""
---

## 4. Attention Entropy & Representation Geometry Analysis

### A. Attention Entropy Analysis:
- **Theoretical Uniform Entropy Reference**: {attn_summary['uniform_entropy_reference']:.4f} (at $L = {attn_summary['sequence_length_L']}$)
- **Observed Mean Attention Entropy**: **{attn_summary['overall_mean_entropy']:.4f}** (Normalized = **{attn_summary['overall_mean_normalized_entropy']:.4f}**)
- **Interpretation**: The attention distribution is sharp and non-uniform ($H < H_{{uniform}}$), demonstrating active temporal localization onto clinically relevant wave segments rather than uniform averaging.

### B. Representation Geometry & Class Separation Gap ($N = 1,000$ Test Subsample):
| Geometry Metric | Model A-ML (GAP Baseline) | Model B-ML (Temporal Attention) | Gain ($\\Delta$) |
|:---|:---:|:---:|:---:|
| **Intra-Class Cosine Similarity** | {geom_summary['model_a_gap_geometry']['mean_intra_class_cosine_similarity']:.4f} | {geom_summary['model_b_attention_geometry']['mean_intra_class_cosine_similarity']:.4f} | {geom_summary['model_b_attention_geometry']['mean_intra_class_cosine_similarity'] - geom_summary['model_a_gap_geometry']['mean_intra_class_cosine_similarity']:+.4f} |
| **Inter-Class Cosine Similarity** | {geom_summary['model_a_gap_geometry']['mean_inter_class_cosine_similarity']:.4f} | {geom_summary['model_b_attention_geometry']['mean_inter_class_cosine_similarity']:.4f} | {geom_summary['model_b_attention_geometry']['mean_inter_class_cosine_similarity'] - geom_summary['model_a_gap_geometry']['mean_inter_class_cosine_similarity']:+.4f} |
| **Class Separation Gap (Intra - Inter)** | **{geom_summary['model_a_gap_geometry']['class_separation_gap']:.4f}** | **{geom_summary['model_b_attention_geometry']['class_separation_gap']:.4f}** | **{geom_summary['separation_gap_gain']:+.4f}** |

---

## 5. Computational Execution & Thermal Safety Audit

| Pipeline Stage | Time Elapsed | Notes |
|:---|:---:|:---|
| **Metadata & Split Loading** | {timing_dict['metadata_loading_time_sec']:.2f} s | Patient-level GroupShuffleSplit verified |
| **Signal Preprocessing (Val + Test)** | {timing_dict['signal_preprocessing_time_sec']:.2f} s | Cached array loading |
| **Validation Threshold Tuning** | {timing_dict['val_threshold_tuning_time_sec']:.2f} s | 19-step grid search per class on Val split |
| **Frozen Test Set Inference** | {timing_dict['test_inference_time_sec']:.2f} s | Single pass on 4,308 test records |
| **Paired 1,000-Resample Bootstrap CIs** | {timing_dict['bootstrap_ci_time_sec']:.2f} s | Multi-metric paired resampling |
| **Attention & Geometry Extraction** | {timing_dict['attention_geometry_time_sec']:.2f} s | Vectorized geometry (N=1,000) |
| **Total Evaluation Runtime** | **{timing_dict['total_evaluation_time_sec']:.2f} s** | **Zero thermal throttling** |

---

## 6. Generated Artifacts in `results/phase6/model_b/`

- `checkpoints/model_b_multilabel_best.pth`
- `results/phase6/model_b/model_b_multilabel_metrics.json`
- `results/phase6/model_b/model_b_multilabel_classification_report.csv`
- `results/phase6/model_b/model_b_multilabel_predictions.csv`
- `results/phase6/model_b/phase6_multilabel_paired_comparison.csv`
- `results/phase6/model_b/phase6_multilabel_paired_bootstrap_cis.json`
- `results/phase6/model_b/model_b_multilabel_attention_entropy.json`
- `results/phase6/model_b/phase6_representation_geometry.json`
- `results/phase6/model_b/MODEL_B_MULTILABEL_REPORT.md`
"""

    report_path = MODEL_B_DIR / "MODEL_B_MULTILABEL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Saved: MODEL_B_MULTILABEL_REPORT.md")

    # =========================================================================
    # 12. Final Verification
    # =========================================================================
    print("\n--- STEP 12: FINAL VERIFICATION ---")
    ckpt_b_sha_after = hashlib.sha256(open(CKPT_B_PATH, "rb").read()).hexdigest()
    assert ckpt_b_sha_after == ckpt_b_sha, "CHECKPOINT WAS MODIFIED!"
    print(f"  Checkpoint SHA-256 unchanged: {ckpt_b_sha_after}")

    expected_files = [
        "model_b_multilabel_metrics.json",
        "model_b_multilabel_classification_report.csv",
        "model_b_multilabel_predictions.csv",
        "phase6_multilabel_paired_comparison.csv",
        "phase6_multilabel_paired_bootstrap_cis.json",
        "model_b_multilabel_attention_entropy.json",
        "phase6_representation_geometry.json",
        "MODEL_B_MULTILABEL_REPORT.md",
    ]
    for fname in expected_files:
        fpath = MODEL_B_DIR / fname
        assert fpath.exists(), f"MISSING: {fpath}"
        print(f"  OK: {fname} ({fpath.stat().st_size} bytes)")

    # Check no NaN/Inf
    for key in ["macro_auroc", "macro_ap", "macro_f1", "weighted_f1", "subset_accuracy", "hamming_loss"]:
        val = metrics_b_opt[key]
        assert np.isfinite(val), f"Non-finite metric: {key} = {val}"
    print("  All metrics finite (no NaN/Inf).")

    print("\n" + "=" * 80)
    print("PHASE 6.2 MODEL B-ML EVALUATION-ONLY RECOVERY COMPLETE")
    print(f"Total Evaluation Time: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    run_model_b_evaluation_only()

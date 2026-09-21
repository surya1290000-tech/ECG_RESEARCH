"""
Phase 8.1 — Lead-Group Occlusion Sensitivity Analysis for HYP
============================================================
Evaluates prediction sensitivity of ECGResNet-GAP and InceptionTime1D
under systematic lead-group masking on the official PTB-XL Fold 10 test set.

STRICT RESEARCH PROTOCOL:
- Pure evaluation/inference only.
- Zero model training, fine-tuning, or parameter updates.
- Zero modification to checkpoints or Fold-10 partition.
- Uses exact validation-frozen thresholds from Phase 7.3/7.5:
    ECGResNet-GAP: th_HYP = 0.25
    InceptionTime1D: th_HYP = 0.23
- No test labels are used for threshold tuning.
- Lead masking replaces selected leads with 0.0 AFTER Z-score normalization
  (representing zero standardized signal amplitude).
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy import stats
import matplotlib.pyplot as plt

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_ptbxl_fold10_splits,
    load_preprocessed_split_arrays,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
)
from src.models.ecg_resnet import ECGResNet
from src.models.inceptiontime1d import InceptionTime1D

OUT_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "hyp_occlusion"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_MODEL_A = CHECKPOINT_DIR / "model_a_fold10_best.pth"
CKPT_INCEPTION = CHECKPOINT_DIR / "model_inception_fold10_best.pth"

# Exact validation-frozen thresholds from Phase 7.3 and 7.5
FROZEN_THRESHOLDS = {
    "ECGResNet-GAP": {"NORM": 0.40, "STTC": 0.25, "CD": 0.45, "MI": 0.40, "HYP": 0.25},
    "InceptionTime1D": {"NORM": 0.36, "STTC": 0.32, "CD": 0.37, "MI": 0.43, "HYP": 0.23},
}

# PTB-XL Lead mapping:
# 0: I, 1: II, 2: III, 3: AVR, 4: AVL, 5: AVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MASKING_CONDITIONS = {
    "NONE": [],
    "MASK_LIMB": [0, 1, 2, 3, 4, 5],
    "MASK_PRECORDIAL": [6, 7, 8, 9, 10, 11],
    "MASK_V1_V2": [6, 7],
    "MASK_V5_V6": [10, 11],
    "MASK_I_AVL": [0, 4],
    "MASK_V1_V2_V5_V6": [6, 7, 10, 11],
}


def run_occlusion_analysis():
    print("=" * 80)
    print("PHASE 8.1: LEAD-GROUP OCCLUSION SENSITIVITY ANALYSIS FOR HYP")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: VERIFY CHECKPOINTS AND DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Verifying checkpoints, metadata, and test set...")
    assert CKPT_MODEL_A.exists(), f"Missing Model A checkpoint: {CKPT_MODEL_A}"
    assert CKPT_INCEPTION.exists(), f"Missing Inception checkpoint: {CKPT_INCEPTION}"

    sha_a = hashlib.sha256(open(CKPT_MODEL_A, "rb").read()).hexdigest()
    sha_inc = hashlib.sha256(open(CKPT_INCEPTION, "rb").read()).hexdigest()
    print(f"ECGResNet-GAP Checkpoint SHA256: {sha_a}")
    print(f"InceptionTime Checkpoint SHA256 : {sha_inc}")

    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)

    test_n = len(test_df)
    assert test_n == 2158, f"Expected 2,158 test records, got {test_n}"
    assert len(set(train_df['patient_id']) & set(test_df['patient_id'])) == 0, "Patient leakage detected!"
    assert len(set(val_df['patient_id']) & set(test_df['patient_id'])) == 0, "Patient leakage detected!"
    print("Patient isolation verified: 0 patient overlap across all splits.")

    # Load preprocessed arrays (cached)
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=True)
    assert test_X.shape == (2158, 12, 1000), f"Expected shape (2158, 12, 1000), got {test_X.shape}"
    assert test_Y.shape == (2158, 5), f"Expected shape (2158, 5), got {test_Y.shape}"

    hyp_idx = CLASS_TO_ID["HYP"]  # 4
    y_true_hyp = test_Y[:, hyp_idx].astype(int)
    hyp_support = int(y_true_hyp.sum())
    assert hyp_support == 262, f"Expected 262 HYP records, got {hyp_support}"
    print(f"Fold 10 Test Support verified: N={test_n}, HYP Support={hyp_support}")

    # Subgroups
    is_hyp = (y_true_hyp == 1)
    is_norm = (test_Y[:, CLASS_TO_ID["NORM"]] == 1)
    is_sttc = (test_Y[:, CLASS_TO_ID["STTC"]] == 1)
    is_cd = (test_Y[:, CLASS_TO_ID["CD"]] == 1)
    is_mi = (test_Y[:, CLASS_TO_ID["MI"]] == 1)

    is_isolated_hyp = is_hyp & (~is_sttc) & (~is_cd) & (~is_mi)
    is_hyp_sttc = is_hyp & is_sttc
    is_hyp_mi = is_hyp & is_mi
    is_hyp_cd = is_hyp & is_cd

    print(f"Subgroups: Isolated HYP={is_isolated_hyp.sum()}, HYP+STTC={is_hyp_sttc.sum()}, HYP+MI={is_hyp_mi.sum()}, HYP+CD={is_hyp_cd.sum()}")

    # Sub-phenotype SCP codes in Fold 10
    test_df_indexed = test_df.copy().set_index("ecg_id" if "ecg_id" in test_df.columns else test_df.index)
    parsed_scps = [test_df_indexed.loc[eid, "scp_codes_parsed"] for eid in test_ids]

    is_lvh = np.array(["LVH" in scp for scp in parsed_scps]) & is_hyp
    is_rvh = np.array(["RVH" in scp for scp in parsed_scps]) & is_hyp
    is_lao = np.array(["LAO/LAE" in scp for scp in parsed_scps]) & is_hyp
    print(f"Phenotypes: LVH={is_lvh.sum()}, RVH={is_rvh.sum()}, LAO/LAE={is_lao.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: RUN INFERENCE FOR BOTH MODELS AND ALL CONDITIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 2 & 3] Running occlusion inference across 7 masking conditions...")

    models = {
        "ECGResNet-GAP": ECGResNet(num_classes=NUM_CLASSES).to(device),
        "InceptionTime1D": InceptionTime1D(num_classes=NUM_CLASSES).to(device),
    }
    models["ECGResNet-GAP"].load_state_dict(torch.load(CKPT_MODEL_A, map_location=device), strict=True)
    models["InceptionTime1D"].load_state_dict(torch.load(CKPT_INCEPTION, map_location=device), strict=True)

    for m in models.values():
        m.eval()

    # Predictions dictionary: {model_name: {condition: (N, 5) probs}}
    raw_probs = {m_name: {} for m_name in models}

    for cond_name, mask_leads in MASKING_CONDITIONS.items():
        t0 = time.time()
        # Clone tensor and mask leads
        X_curr = test_X.copy()
        if len(mask_leads) > 0:
            X_curr[:, mask_leads, :] = 0.0  # Zero standardized signal

        tensor_x = torch.from_numpy(X_curr).to(device)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=256, shuffle=False)

        for m_name, model in models.items():
            probs_list = []
            with torch.no_grad():
                for (batch_x,) in loader:
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_list.append(probs)
            raw_probs[m_name][cond_name] = np.vstack(probs_list)
        print(f"Condition '{cond_name}' evaluated for both models in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 4: PRIMARY HYP ANALYSIS & METRIC COMPUTATION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Computing detailed HYP performance & occlusion metrics...")

    def compute_condition_stats(y_true, probs, baseline_probs, threshold):
        y_pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        auroc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        delta_prob = probs - baseline_probs
        prob_drop = baseline_probs - probs
        abs_change = np.abs(delta_prob)

        return {
            "auroc": float(auroc),
            "ap": float(ap),
            "f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "specificity": float(spec),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "mean_prob": float(probs.mean()),
            "median_prob": float(np.median(probs)),
            "mean_prob_drop": float(prob_drop.mean()),
            "median_prob_drop": float(np.median(prob_drop)),
            "mean_abs_prob_change": float(abs_change.mean()),
        }

    metrics_master = {}
    detailed_rows = []

    subgroups = {
        "All_Fold10": np.ones(test_n, dtype=bool),
        "HYP_Positive": is_hyp,
        "Isolated_HYP": is_isolated_hyp,
        "HYP_plus_STTC": is_hyp_sttc,
        "HYP_plus_MI": is_hyp_mi,
        "HYP_plus_CD": is_hyp_cd,
    }

    for m_name in models:
        metrics_master[m_name] = {}
        th_hyp = FROZEN_THRESHOLDS[m_name]["HYP"]
        base_probs_hyp = raw_probs[m_name]["NONE"][:, hyp_idx]

        for cond_name in MASKING_CONDITIONS:
            curr_probs_hyp = raw_probs[m_name][cond_name]["HYP" if False else None, hyp_idx] if False else raw_probs[m_name][cond_name][:, hyp_idx]
            cond_stats = compute_condition_stats(y_true_hyp, curr_probs_hyp, base_probs_hyp, th_hyp)
            metrics_master[m_name][cond_name] = cond_stats

            # Compute subgroup-level statistics
            sub_stats = {}
            for sub_name, mask in subgroups.items():
                p_sub = curr_probs_hyp[mask]
                p_base_sub = base_probs_hyp[mask]
                y_sub = y_true_hyp[mask]
                drop_sub = p_base_sub - p_sub
                sens_sub = float((p_sub >= th_hyp).mean()) if sub_name != "All_Fold10" else cond_stats["recall"]

                sub_stats[sub_name] = {
                    "N": int(mask.sum()),
                    "mean_prob": float(p_sub.mean()),
                    "median_prob": float(np.median(p_sub)),
                    "mean_prob_drop": float(drop_sub.mean()),
                    "median_prob_drop": float(np.median(drop_sub)),
                    "sensitivity": float(sens_sub),
                }

                detailed_rows.append({
                    "model": m_name,
                    "condition": cond_name,
                    "subgroup": sub_name,
                    "N": int(mask.sum()),
                    "mean_baseline_prob": float(p_base_sub.mean()),
                    "mean_masked_prob": float(p_sub.mean()),
                    "mean_prob_drop": float(drop_sub.mean()),
                    "median_prob_drop": float(np.median(drop_sub)),
                    "mean_abs_change": float(np.abs(p_sub - p_base_sub).mean()),
                    "sensitivity_recall": float(sens_sub),
                    "threshold": th_hyp,
                })

            metrics_master[m_name][cond_name]["subgroups"] = sub_stats

    # Save metrics JSON
    with open(OUT_DIR / "hyp_occlusion_metrics.json", "w") as f:
        json.dump(metrics_master, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 5: SUB-PHENOTYPE ANALYSIS (LVH, RVH, LAO/LAE)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Computing sub-phenotype masking sensitivity...")
    phenotypes = {
        "LVH": is_lvh,
        "RVH": is_rvh,
        "LAO/LAE": is_lao,
    }
    phenotype_rows = []
    for m_name in models:
        th_hyp = FROZEN_THRESHOLDS[m_name]["HYP"]
        base_probs_hyp = raw_probs[m_name]["NONE"][:, hyp_idx]

        for p_name, mask in phenotypes.items():
            base_p_pheno = base_probs_hyp[mask]
            base_sens = float((base_p_pheno >= th_hyp).mean())

            for cond_name in MASKING_CONDITIONS:
                curr_p_pheno = raw_probs[m_name][cond_name][:, hyp_idx][mask]
                curr_sens = float((curr_p_pheno >= th_hyp).mean())
                drop_pheno = base_p_pheno - curr_p_pheno

                phenotype_rows.append({
                    "model": m_name,
                    "phenotype": p_name,
                    "N": int(mask.sum()),
                    "condition": cond_name,
                    "baseline_sensitivity": base_sens,
                    "masked_sensitivity": curr_sens,
                    "mean_prob_drop": float(drop_pheno.mean()),
                    "median_prob_drop": float(np.median(drop_pheno)),
                    "mean_baseline_prob": float(base_p_pheno.mean()),
                    "mean_masked_prob": float(curr_p_pheno.mean()),
                })
    df_phenotype = pd.DataFrame(phenotype_rows)
    df_phenotype.to_csv(OUT_DIR / "hyp_phenotype_occlusion_sensitivity.csv", index=False)

    # -------------------------------------------------------------------------
    # STEP 6 & 7: PREDICTION CSV & PAIRED BOOTSTRAP MODEL COMPARISON
    # -------------------------------------------------------------------------
    print("\n[STEP 6 & 7] Running 1,000 paired bootstrap resamples for Model Comparison...")

    # Build predictions CSV
    pred_data = {
        "ecg_id": test_ids,
        "true_HYP": y_true_hyp,
        "true_NORM": test_Y[:, 0].astype(int),
        "true_STTC": test_Y[:, 1].astype(int),
        "true_CD": test_Y[:, 2].astype(int),
        "true_MI": test_Y[:, 3].astype(int),
        "is_isolated_HYP": is_isolated_hyp.astype(int),
        "is_LVH": is_lvh.astype(int),
        "is_RVH": is_rvh.astype(int),
        "is_LAO": is_lao.astype(int),
    }
    for m_short, m_name in [("A", "ECGResNet-GAP"), ("Inc", "InceptionTime1D")]:
        base_p = raw_probs[m_name]["NONE"][:, hyp_idx]
        pred_data[f"prob_{m_short}_NONE"] = base_p
        for cond_name in MASKING_CONDITIONS:
            if cond_name == "NONE":
                continue
            p_cond = raw_probs[m_name][cond_name][:, hyp_idx]
            pred_data[f"prob_{m_short}_{cond_name}"] = p_cond
            pred_data[f"drop_{m_short}_{cond_name}"] = base_p - p_cond

    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(OUT_DIR / "hyp_occlusion_predictions.csv", index=False)
    print(f"Saved {len(df_preds)} record-level predictions to: hyp_occlusion_predictions.csv")

    # Bootstrap comparison: InceptionTime - ECGResNet
    rng = np.random.RandomState(42)
    n_boot = 1000
    bootstrap_rows = []

    th_a = FROZEN_THRESHOLDS["ECGResNet-GAP"]["HYP"]
    th_inc = FROZEN_THRESHOLDS["InceptionTime1D"]["HYP"]

    for cond_name in MASKING_CONDITIONS:
        p_a = raw_probs["ECGResNet-GAP"][cond_name][:, hyp_idx]
        base_p_a = raw_probs["ECGResNet-GAP"]["NONE"][:, hyp_idx]
        drop_a = base_p_a - p_a

        p_inc = raw_probs["InceptionTime1D"][cond_name][:, hyp_idx]
        base_p_inc = raw_probs["InceptionTime1D"]["NONE"][:, hyp_idx]
        drop_inc = base_p_inc - p_inc

        # Point estimates
        auc_a = roc_auc_score(y_true_hyp, p_a)
        auc_inc = roc_auc_score(y_true_hyp, p_inc)
        ap_a = average_precision_score(y_true_hyp, p_a)
        ap_inc = average_precision_score(y_true_hyp, p_inc)
        f1_a = f1_score(y_true_hyp, (p_a >= th_a).astype(int), zero_division=0)
        f1_inc = f1_score(y_true_hyp, (p_inc >= th_inc).astype(int), zero_division=0)

        # Isolated HYP sensitivity
        iso_sens_a = float((p_a[is_isolated_hyp] >= th_a).mean())
        iso_sens_inc = float((p_inc[is_isolated_hyp] >= th_inc).mean())

        # Bootstraps
        b_diff_auc = []
        b_diff_ap = []
        b_diff_f1 = []
        b_diff_drop = []
        b_diff_iso_sens = []

        for _ in range(n_boot):
            b_idx = rng.randint(0, test_n, test_n)
            yt = y_true_hyp[b_idx]
            if len(np.unique(yt)) < 2:
                continue

            # AUROC & AP
            b_auc_a = roc_auc_score(yt, p_a[b_idx])
            b_auc_inc = roc_auc_score(yt, p_inc[b_idx])
            b_ap_a = average_precision_score(yt, p_a[b_idx])
            b_ap_inc = average_precision_score(yt, p_inc[b_idx])
            b_f1_a = f1_score(yt, (p_a[b_idx] >= th_a).astype(int), zero_division=0)
            b_f1_inc = f1_score(yt, (p_inc[b_idx] >= th_inc).astype(int), zero_division=0)

            b_diff_auc.append(b_auc_inc - b_auc_a)
            b_diff_ap.append(b_ap_inc - b_ap_a)
            b_diff_f1.append(b_f1_inc - b_f1_a)
            b_diff_drop.append(drop_inc[b_idx].mean() - drop_a[b_idx].mean())

            # Isolated subset in bootstrap
            b_iso_mask = is_isolated_hyp[b_idx]
            if b_iso_mask.sum() > 0:
                b_s_a = (p_a[b_idx][b_iso_mask] >= th_a).mean()
                b_s_inc = (p_inc[b_idx][b_iso_mask] >= th_inc).mean()
                b_diff_iso_sens.append(b_s_inc - b_s_a)

        def ci_str(arr):
            return f"[{np.percentile(arr, 2.5):+.4f}, {np.percentile(arr, 97.5):+.4f}]"

        bootstrap_rows.append({
            "condition": cond_name,
            "AUROC_Model_A": auc_a,
            "AUROC_Inception": auc_inc,
            "AUROC_Diff (Inc - A)": auc_inc - auc_a,
            "AUROC_Diff_95CI": ci_str(b_diff_auc),
            "AP_Model_A": ap_a,
            "AP_Inception": ap_inc,
            "AP_Diff (Inc - A)": ap_inc - ap_a,
            "AP_Diff_95CI": ci_str(b_diff_ap),
            "F1_Model_A": f1_a,
            "F1_Inception": f1_inc,
            "F1_Diff (Inc - A)": f1_inc - f1_a,
            "F1_Diff_95CI": ci_str(b_diff_f1),
            "Mean_Drop_Model_A": float(drop_a.mean()),
            "Mean_Drop_Inception": float(drop_inc.mean()),
            "Mean_Drop_Diff (Inc - A)": float(drop_inc.mean() - drop_a.mean()),
            "Mean_Drop_Diff_95CI": ci_str(b_diff_drop),
            "Isolated_Sens_Model_A": iso_sens_a,
            "Isolated_Sens_Inception": iso_sens_inc,
            "Isolated_Sens_Diff (Inc - A)": iso_sens_inc - iso_sens_a,
            "Isolated_Sens_Diff_95CI": ci_str(b_diff_iso_sens) if len(b_diff_iso_sens) > 0 else "N/A",
        })

    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(OUT_DIR / "hyp_occlusion_bootstrap.csv", index=False)
    print("Saved paired bootstrap results to: hyp_occlusion_bootstrap.csv")

    # -------------------------------------------------------------------------
    # STEP 8: PUBLICATION-QUALITY VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Generating publication-quality plots...")
    conditions_list = ["MASK_LIMB", "MASK_PRECORDIAL", "MASK_V1_V2", "MASK_V5_V6", "MASK_I_AVL", "MASK_V1_V2_V5_V6"]
    cond_labels = ["Limb (I-aVF)", "Precordial (V1-V6)", "V1-V2 (Septal)", "V5-V6 (Lateral)", "I-aVL (High Lat)", "V1,V2,V5,V6"]

    # 1. Probability Drop by Lead Group (All HYP vs Isolated HYP)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(conditions_list))
    w = 0.2

    drop_a_all = [df_preds.loc[is_hyp, f"drop_A_{c}"].mean() for c in conditions_list]
    drop_inc_all = [df_preds.loc[is_hyp, f"drop_Inc_{c}"].mean() for c in conditions_list]
    drop_a_iso = [df_preds.loc[is_isolated_hyp, f"drop_A_{c}"].mean() for c in conditions_list]
    drop_inc_iso = [df_preds.loc[is_isolated_hyp, f"drop_Inc_{c}"].mean() for c in conditions_list]

    ax.bar(x - 1.5 * w, drop_a_all, w, label="ECGResNet-GAP (All HYP)", color="#1f77b4")
    ax.bar(x - 0.5 * w, drop_inc_all, w, label="InceptionTime1D (All HYP)", color="#aec7e8")
    ax.bar(x + 0.5 * w, drop_a_iso, w, label="ECGResNet-GAP (Isolated HYP)", color="#d62728")
    ax.bar(x + 1.5 * w, drop_inc_iso, w, label="InceptionTime1D (Isolated HYP)", color="#ff9896")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean HYP Probability Drop ($P_{unmasked} - P_{masked}$)")
    ax.set_title("HYP Probability Sensitivity by Lead-Group Masking Condition (Fold 10)", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hyp_probability_drop.png")
    plt.close()

    # 2. Isolated HYP Probability Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    for ax_i, (m_short, m_title) in enumerate([("A", "ECGResNet-GAP"), ("Inc", "InceptionTime1D")]):
        data_to_plot = [
            df_preds.loc[is_isolated_hyp, f"prob_{m_short}_NONE"],
            df_preds.loc[is_isolated_hyp, f"prob_{m_short}_MASK_LIMB"],
            df_preds.loc[is_isolated_hyp, f"prob_{m_short}_MASK_PRECORDIAL"],
            df_preds.loc[is_isolated_hyp, f"prob_{m_short}_MASK_V1_V2_V5_V6"],
        ]
        box = axes[ax_i].boxplot(data_to_plot, patch_artist=True, labels=["Baseline", "Mask Limb", "Mask Precordial", "Mask V1,V2,V5,V6"])
        colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[ax_i].set_title(f"{m_title} — Isolated HYP Probabilities", fontweight="bold")
        axes[ax_i].axhline(FROZEN_THRESHOLDS[m_title]["HYP"], color="red", linestyle="--", label=f"Threshold ({FROZEN_THRESHOLDS[m_title]['HYP']})")
        axes[ax_i].legend(loc="upper right")
        axes[ax_i].grid(axis="y", linestyle=":", alpha=0.6)
        axes[ax_i].set_ylabel("Predicted HYP Probability" if ax_i == 0 else "")
    plt.suptitle("Isolated HYP Probability Distributions Under Major Masking Conditions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "isolated_hyp_probability.png", bbox_inches="tight")
    plt.close()

    # 3. Subgroup Sensitivity by Masking Condition
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
    df_det = pd.DataFrame(detailed_rows)
    subgroups_to_plot = ["Isolated_HYP", "HYP_plus_STTC", "HYP_plus_MI", "HYP_plus_CD"]
    sub_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax_i, m_name in enumerate(models):
        sub_df = df_det[df_det["model"] == m_name]
        x = np.arange(len(conditions_list))
        w = 0.2
        for s_i, (s_name, c) in enumerate(zip(subgroups_to_plot, sub_colors)):
            s_vals = [sub_df[(sub_df["subgroup"] == s_name) & (sub_df["condition"] == cond)]["sensitivity_recall"].iloc[0] * 100 for cond in conditions_list]
            axes[ax_i].bar(x + (s_i - 1.5) * w, s_vals, w, label=s_name.replace("_", " "), color=c, alpha=0.85)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(cond_labels, rotation=20, ha="right")
        axes[ax_i].set_title(f"{m_name} Sensitivity by Pathology Subgroup", fontweight="bold")
        axes[ax_i].grid(axis="y", linestyle=":", alpha=0.6)
        if ax_i == 0:
            axes[ax_i].set_ylabel("Sensitivity / Recall (%)")
        axes[ax_i].legend(loc="upper right")
    plt.suptitle("HYP Sensitivity Across Pathological Subgroups Under Masking Conditions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "subgroup_sensitivity.png", bbox_inches="tight")
    plt.close()

    # 4. Heatmap of Mean Probability Drop across Subgroups
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
    sub_keys = ["All_Fold10", "HYP_Positive", "Isolated_HYP", "HYP_plus_STTC", "HYP_plus_MI", "HYP_plus_CD"]
    sub_key_labels = ["All Records", "All HYP+", "Isolated HYP", "HYP + STTC", "HYP + MI", "HYP + CD"]

    for ax_i, m_name in enumerate(models):
        sub_df = df_det[df_det["model"] == m_name]
        mat = np.zeros((len(sub_keys), len(conditions_list)))
        for r_i, s_name in enumerate(sub_keys):
            for c_i, cond in enumerate(conditions_list):
                mat[r_i, c_i] = sub_df[(sub_df["subgroup"] == s_name) & (sub_df["condition"] == cond)]["mean_prob_drop"].iloc[0]

        im = axes[ax_i].imshow(mat, cmap="Reds", aspect="auto", vmin=0, vmax=0.15)
        axes[ax_i].set_xticks(range(len(conditions_list)))
        axes[ax_i].set_xticklabels(cond_labels, rotation=25, ha="right")
        axes[ax_i].set_yticks(range(len(sub_keys)))
        axes[ax_i].set_yticklabels(sub_key_labels if ax_i == 0 else [])
        axes[ax_i].set_title(f"{m_name} Probability Drop", fontweight="bold")

        for r_i in range(len(sub_keys)):
            for c_i in range(len(conditions_list)):
                val = mat[r_i, c_i]
                axes[ax_i].text(c_i, r_i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.08 else "black", fontsize=8)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Probability Drop ($P_{unmasked} - P_{masked}$)")
    plt.suptitle("Lead-Group Occlusion Effect Across Diagnostic Subgroups", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(PLOTS_DIR / "masking_effect_heatmap.png", bbox_inches="tight")
    plt.close()

    # 5. Sub-phenotype Masking Sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    pheno_keys = ["LVH", "RVH", "LAO/LAE"]
    pheno_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax_i, m_name in enumerate(models):
        sub_df = df_phenotype[df_phenotype["model"] == m_name]
        x = np.arange(len(conditions_list))
        w = 0.25
        for p_i, (p_name, c) in enumerate(zip(pheno_keys, pheno_colors)):
            vals = [sub_df[(sub_df["phenotype"] == p_name) & (sub_df["condition"] == cond)]["masked_sensitivity"].iloc[0] * 100 for cond in conditions_list]
            axes[ax_i].bar(x + (p_i - 1) * w, vals, w, label=f"{p_name} (N={sub_df[sub_df['phenotype']==p_name]['N'].iloc[0]})", color=c, alpha=0.85)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(cond_labels, rotation=20, ha="right")
        axes[ax_i].set_title(f"{m_name} Sub-Phenotype Sensitivity", fontweight="bold")
        axes[ax_i].grid(axis="y", linestyle=":", alpha=0.6)
        if ax_i == 0:
            axes[ax_i].set_ylabel("Sensitivity / Recall (%)")
        axes[ax_i].legend(loc="upper right")
    plt.suptitle("Diagnostic Sub-Phenotype Sensitivity Under Lead-Group Occlusion", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phenotype_masking_effect.png", bbox_inches="tight")
    plt.close()

    # 6. Model Comparison (AUROC & AP under Occlusion)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    all_conds = ["NONE"] + conditions_list
    all_labels = ["Baseline (None)"] + cond_labels

    auroc_a = [metrics_master["ECGResNet-GAP"][c]["auroc"] for c in all_conds]
    auroc_inc = [metrics_master["InceptionTime1D"][c]["auroc"] for c in all_conds]
    ap_a = [metrics_master["ECGResNet-GAP"][c]["ap"] for c in all_conds]
    ap_inc = [metrics_master["InceptionTime1D"][c]["ap"] for c in all_conds]

    x = np.arange(len(all_conds))
    w = 0.35

    ax1.bar(x - w/2, auroc_a, w, label="ECGResNet-GAP", color="#1f77b4")
    ax1.bar(x + w/2, auroc_inc, w, label="InceptionTime1D", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=25, ha="right")
    ax1.set_ylabel("HYP AUROC")
    ax1.set_ylim(0.65, 0.85)
    ax1.set_title("HYP AUROC Across Occlusion Conditions", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.bar(x - w/2, ap_a, w, label="ECGResNet-GAP", color="#1f77b4")
    ax2.bar(x + w/2, ap_inc, w, label="InceptionTime1D", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=25, ha="right")
    ax2.set_ylabel("HYP Average Precision (AP)")
    ax2.set_ylim(0.20, 0.50)
    ax2.set_title("HYP AP Across Occlusion Conditions", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend()

    plt.suptitle("Model Comparison Under Lead-Group Occlusion (Fold 10)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()

    print(f"\nAll deliverables and plots successfully created under: {OUT_DIR}")


if __name__ == "__main__":
    run_occlusion_analysis()

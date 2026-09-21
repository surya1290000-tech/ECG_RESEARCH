"""
Phase 8.2 — Single-Lead HYP Sensitivity Profile (Official Fold-10 Benchmark)
============================================================================
Evaluates prediction sensitivity of ECGResNet-GAP and InceptionTime1D
under systematic single-lead masking across all 12 standard ECG leads on the
official PTB-XL Fold 10 test set (N = 2,158).

STRICT RESEARCH PROTOCOL:
- Pure evaluation/inference only.
- Zero model training, fine-tuning, or parameter updates.
- Zero modification to checkpoints or Fold-10 partition.
- Uses exact validation-frozen thresholds from Phase 7.3/7.5:
    ECGResNet-GAP: th_HYP = 0.25
    InceptionTime1D: th_HYP = 0.23
- No test labels are used for threshold tuning.
- Lead masking replaces exactly ONE lead channel with 0.0 AFTER Z-score normalization
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

OUT_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "hyp_single_lead"
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
# 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]

# 13 Conditions: NONE + 12 single leads
SINGLE_LEAD_CONDITIONS = {"NONE": None}
for idx, l_name in enumerate(LEAD_NAMES):
    SINGLE_LEAD_CONDITIONS[f"MASK_{l_name}"] = idx


def run_single_lead_analysis():
    print("=" * 80)
    print("PHASE 8.2: SINGLE-LEAD HYP SENSITIVITY PROFILE (OFFICIAL FOLD 10)")
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
    is_sttc = (test_Y[:, CLASS_TO_ID["STTC"]] == 1)
    is_cd = (test_Y[:, CLASS_TO_ID["CD"]] == 1)
    is_mi = (test_Y[:, CLASS_TO_ID["MI"]] == 1)

    is_isolated_hyp = is_hyp & (~is_sttc) & (~is_cd) & (~is_mi)
    is_hyp_sttc = is_hyp & is_sttc
    is_hyp_mi = is_hyp & is_mi
    is_hyp_cd = is_hyp & is_cd

    assert is_isolated_hyp.sum() == 56, f"Expected 56 isolated HYP records, got {is_isolated_hyp.sum()}"
    assert is_hyp_sttc.sum() == 155, f"Expected 155 HYP+STTC records, got {is_hyp_sttc.sum()}"
    print(f"Subgroups verified: Isolated HYP={is_isolated_hyp.sum()}, HYP+STTC={is_hyp_sttc.sum()}, HYP+MI={is_hyp_mi.sum()}, HYP+CD={is_hyp_cd.sum()}")

    # Sub-phenotype SCP codes in Fold 10
    test_df_indexed = test_df.copy().set_index("ecg_id" if "ecg_id" in test_df.columns else test_df.index)
    parsed_scps = [test_df_indexed.loc[eid, "scp_codes_parsed"] for eid in test_ids]

    is_lvh = np.array(["LVH" in scp for scp in parsed_scps]) & is_hyp
    is_rvh = np.array(["RVH" in scp for scp in parsed_scps]) & is_hyp
    is_lao = np.array(["LAO/LAE" in scp for scp in parsed_scps]) & is_hyp
    assert is_lvh.sum() == 214, f"Expected 214 LVH records, got {is_lvh.sum()}"
    assert is_rvh.sum() == 12, f"Expected 12 RVH records, got {is_rvh.sum()}"
    assert is_lao.sum() == 42, f"Expected 42 LAO/LAE records, got {is_lao.sum()}"
    print(f"Phenotypes verified: LVH={is_lvh.sum()}, RVH={is_rvh.sum()}, LAO/LAE={is_lao.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: RUN INFERENCE FOR BOTH MODELS ACROSS ALL 13 CONDITIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 2 & 3] Running single-lead inference across 13 masking conditions...")

    models = {
        "ECGResNet-GAP": ECGResNet(num_classes=NUM_CLASSES).to(device),
        "InceptionTime1D": InceptionTime1D(num_classes=NUM_CLASSES).to(device),
    }
    models["ECGResNet-GAP"].load_state_dict(torch.load(CKPT_MODEL_A, map_location=device), strict=True)
    models["InceptionTime1D"].load_state_dict(torch.load(CKPT_INCEPTION, map_location=device), strict=True)

    for m in models.values():
        m.eval()

    raw_probs = {m_name: {} for m_name in models}

    for cond_name, mask_idx in SINGLE_LEAD_CONDITIONS.items():
        t0 = time.time()
        X_curr = test_X.copy()
        if mask_idx is not None:
            X_curr[:, mask_idx, :] = 0.0  # Zero standardized signal for this lead

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
        print(f"Condition '{cond_name:<10}' evaluated for both models in {time.time() - t0:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 4: GLOBAL METRICS & MASKING ARTIFACT (GLOBAL DRIFT) ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[STEP 4 & 10] Computing global metrics and global probability drift across all 2,158 records...")

    def compute_stats(y_true, probs, baseline_probs, threshold):
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
            "mean_prob_change": float(delta_prob.mean()),
            "median_prob_change": float(np.median(delta_prob)),
            "mean_prob_drop": float(prob_drop.mean()),
            "median_prob_drop": float(np.median(prob_drop)),
            "mean_abs_prob_change": float(abs_change.mean()),
        }

    metrics_master = {}
    drift_rows = []
    detailed_subgroup_rows = []

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

        for cond_name, mask_idx in SINGLE_LEAD_CONDITIONS.items():
            curr_probs_hyp = raw_probs[m_name][cond_name][:, hyp_idx]
            cond_stats = compute_stats(y_true_hyp, curr_probs_hyp, base_probs_hyp, th_hyp)
            metrics_master[m_name][cond_name] = cond_stats

            # Global drift on all 2,158 records (including non-HYP)
            lead_name = cond_name.replace("MASK_", "")
            delta_all = curr_probs_hyp - base_probs_hyp
            drift_rows.append({
                "model": m_name,
                "condition": cond_name,
                "lead": lead_name,
                "lead_type": "None" if lead_name == "NONE" else ("Precordial" if lead_name in PRECORDIAL_LEADS else "Limb"),
                "mean_global_delta": float(delta_all.mean()),
                "median_global_delta": float(np.median(delta_all)),
                "std_global_delta": float(delta_all.std()),
                "mean_abs_delta": float(np.abs(delta_all).mean()),
                "hyp_auroc": cond_stats["auroc"],
                "hyp_ap": cond_stats["ap"],
                "hyp_f1": cond_stats["f1"],
                "hyp_recall": cond_stats["recall"],
                "hyp_specificity": cond_stats["specificity"],
            })

            # Subgroup breakdowns
            sub_stats = {}
            for sub_name, mask in subgroups.items():
                p_sub = curr_probs_hyp[mask]
                p_base_sub = base_probs_hyp[mask]
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

                detailed_subgroup_rows.append({
                    "model": m_name,
                    "condition": cond_name,
                    "lead": lead_name,
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

    # Save metrics JSON and drift CSV
    with open(OUT_DIR / "hyp_single_lead_metrics.json", "w") as f:
        json.dump(metrics_master, f, indent=2)

    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv(OUT_DIR / "hyp_single_lead_global_drift.csv", index=False)
    print("Saved global drift metrics to: hyp_single_lead_global_drift.csv")

    # -------------------------------------------------------------------------
    # STEP 5: SUB-PHENOTYPE ANALYSIS (LVH, RVH, LAO/LAE)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Computing single-lead sensitivity for sub-phenotypes...")
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

            for cond_name, mask_idx in SINGLE_LEAD_CONDITIONS.items():
                curr_p_pheno = raw_probs[m_name][cond_name][:, hyp_idx][mask]
                curr_sens = float((curr_p_pheno >= th_hyp).mean())
                drop_pheno = base_p_pheno - curr_p_pheno

                phenotype_rows.append({
                    "model": m_name,
                    "phenotype": p_name,
                    "N": int(mask.sum()),
                    "condition": cond_name,
                    "lead": cond_name.replace("MASK_", ""),
                    "lead_type": "None" if cond_name == "NONE" else ("Precordial" if cond_name.replace("MASK_", "") in PRECORDIAL_LEADS else "Limb"),
                    "baseline_sensitivity": base_sens,
                    "masked_sensitivity": curr_sens,
                    "mean_prob_drop": float(drop_pheno.mean()),
                    "median_prob_drop": float(np.median(drop_pheno)),
                    "mean_baseline_prob": float(base_p_pheno.mean()),
                    "mean_masked_prob": float(curr_p_pheno.mean()),
                })
    df_phenotype = pd.DataFrame(phenotype_rows)
    df_phenotype.to_csv(OUT_DIR / "hyp_single_lead_subphenotypes.csv", index=False)
    print("Saved phenotype sensitivity to: hyp_single_lead_subphenotypes.csv")

    # -------------------------------------------------------------------------
    # STEP 6 & 8: PREDICTIONS CSV & 1,000 PAIRED BOOTSTRAP MODEL COMPARISONS
    # -------------------------------------------------------------------------
    print("\n[STEP 6 & 8] Saving predictions CSV and running 1,000 paired bootstrap resamples...")

    # Predictions CSV
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
        for l_name in LEAD_NAMES:
            p_cond = raw_probs[m_name][f"MASK_{l_name}"][:, hyp_idx]
            pred_data[f"prob_{m_short}_{l_name}"] = p_cond
            pred_data[f"drop_{m_short}_{l_name}"] = base_p - p_cond

    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(OUT_DIR / "hyp_single_lead_predictions.csv", index=False)
    print(f"Saved {len(df_preds)} record-level predictions to: hyp_single_lead_predictions.csv")

    # 1,000 paired bootstrap comparison: InceptionTime - ECGResNet
    rng = np.random.RandomState(42)
    n_boot = 1000
    bootstrap_rows = []

    th_a = FROZEN_THRESHOLDS["ECGResNet-GAP"]["HYP"]
    th_inc = FROZEN_THRESHOLDS["InceptionTime1D"]["HYP"]

    for cond_name in SINGLE_LEAD_CONDITIONS:
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

        # Isolated HYP metrics
        iso_sens_a = float((p_a[is_isolated_hyp] >= th_a).mean())
        iso_sens_inc = float((p_inc[is_isolated_hyp] >= th_inc).mean())
        iso_drop_a = float(drop_a[is_isolated_hyp].mean())
        iso_drop_inc = float(drop_inc[is_isolated_hyp].mean())

        b_diff_auc = []
        b_diff_ap = []
        b_diff_f1 = []
        b_diff_drop = []
        b_diff_iso_drop = []
        b_diff_iso_sens = []

        for _ in range(n_boot):
            b_idx = rng.randint(0, test_n, test_n)
            yt = y_true_hyp[b_idx]
            if len(np.unique(yt)) < 2:
                continue

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

            b_iso_mask = is_isolated_hyp[b_idx]
            if b_iso_mask.sum() > 0:
                b_s_a = (p_a[b_idx][b_iso_mask] >= th_a).mean()
                b_s_inc = (p_inc[b_idx][b_iso_mask] >= th_inc).mean()
                b_diff_iso_sens.append(b_s_inc - b_s_a)
                b_diff_iso_drop.append(drop_inc[b_idx][b_iso_mask].mean() - drop_a[b_idx][b_iso_mask].mean())

        def ci_str(arr):
            return f"[{np.percentile(arr, 2.5):+.4f}, {np.percentile(arr, 97.5):+.4f}]"

        bootstrap_rows.append({
            "condition": cond_name,
            "lead": cond_name.replace("MASK_", ""),
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
            "Isolated_Drop_Diff (Inc - A)": iso_drop_inc - iso_drop_a,
            "Isolated_Drop_Diff_95CI": ci_str(b_diff_iso_drop) if len(b_diff_iso_drop) > 0 else "N/A",
        })

    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(OUT_DIR / "hyp_single_lead_bootstrap.csv", index=False)
    print("Saved paired bootstrap results to: hyp_single_lead_bootstrap.csv")

    # -------------------------------------------------------------------------
    # STEP 9: PUBLICATION-QUALITY VISUALIZATIONS (8 FIGURES)
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Generating 8 publication-quality plots...")
    df_det = pd.DataFrame(detailed_subgroup_rows)
    lead_conds = [f"MASK_{l}" for l in LEAD_NAMES]

    # Plot 1: 12-Lead HYP Probability-Drop Profile (ECGResNet vs InceptionTime)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    x = np.arange(len(LEAD_NAMES))
    w = 0.35

    drop_a_hyp = [df_det[(df_det["model"] == "ECGResNet-GAP") & (df_det["condition"] == f"MASK_{l}") & (df_det["subgroup"] == "HYP_Positive")]["mean_prob_drop"].iloc[0] for l in LEAD_NAMES]
    drop_inc_hyp = [df_det[(df_det["model"] == "InceptionTime1D") & (df_det["condition"] == f"MASK_{l}") & (df_det["subgroup"] == "HYP_Positive")]["mean_prob_drop"].iloc[0] for l in LEAD_NAMES]

    ax.bar(x - w/2, drop_a_hyp, w, label="ECGResNet-GAP", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, drop_inc_hyp, w, label="InceptionTime1D", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES, fontweight="bold")
    ax.set_ylabel("Mean HYP Probability Drop ($P_{baseline} - P_{masked}$)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Single-Lead HYP Probability Drop Profile on HYP-Positive ECGs (Fold 10)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lead_probability_drop.png")
    plt.close()

    # Plot 2: 12-Lead AUROC Degradation Profile
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    base_auc_a = metrics_master["ECGResNet-GAP"]["NONE"]["auroc"]
    base_auc_inc = metrics_master["InceptionTime1D"]["NONE"]["auroc"]

    deg_auc_a = [base_auc_a - metrics_master["ECGResNet-GAP"][f"MASK_{l}"]["auroc"] for l in LEAD_NAMES]
    deg_auc_inc = [base_auc_inc - metrics_master["InceptionTime1D"][f"MASK_{l}"]["auroc"] for l in LEAD_NAMES]

    ax.bar(x - w/2, deg_auc_a, w, label="ECGResNet-GAP", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, deg_auc_inc, w, label="InceptionTime1D", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES, fontweight="bold")
    ax.set_ylabel("AUROC Degradation ($AUROC_{baseline} - AUROC_{masked}$)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Single-Lead AUROC Degradation Profile (Fold 10)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lead_auroc_degradation.png")
    plt.close()

    # Plot 3: 12-Lead AP Degradation Profile
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    base_ap_a = metrics_master["ECGResNet-GAP"]["NONE"]["ap"]
    base_ap_inc = metrics_master["InceptionTime1D"]["NONE"]["ap"]

    deg_ap_a = [base_ap_a - metrics_master["ECGResNet-GAP"][f"MASK_{l}"]["ap"] for l in LEAD_NAMES]
    deg_ap_inc = [base_ap_inc - metrics_master["InceptionTime1D"][f"MASK_{l}"]["ap"] for l in LEAD_NAMES]

    ax.bar(x - w/2, deg_ap_a, w, label="ECGResNet-GAP", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, deg_ap_inc, w, label="InceptionTime1D", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES, fontweight="bold")
    ax.set_ylabel("AP Degradation ($AP_{baseline} - AP_{masked}$)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Single-Lead Average Precision Degradation Profile (Fold 10)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lead_ap_degradation.png")
    plt.close()

    # Plot 4: Isolated HYP Probability Drop Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)
    sub_keys = ["Isolated_HYP", "HYP_plus_STTC", "HYP_plus_MI", "HYP_plus_CD"]
    sub_key_labels = ["Isolated HYP (N=56)", "HYP + STTC (N=155)", "HYP + MI (N=79)", "HYP + CD (N=75)"]

    for ax_i, m_name in enumerate(models):
        mat = np.zeros((len(sub_keys), len(LEAD_NAMES)))
        for r_i, s_name in enumerate(sub_keys):
            for c_i, l_name in enumerate(LEAD_NAMES):
                mat[r_i, c_i] = df_det[(df_det["model"] == m_name) & (df_det["subgroup"] == s_name) & (df_det["condition"] == f"MASK_{l_name}")]["mean_prob_drop"].iloc[0]

        im = axes[ax_i].imshow(mat, cmap="coolwarm", aspect="auto", vmin=-0.08, vmax=0.08)
        axes[ax_i].set_xticks(range(len(LEAD_NAMES)))
        axes[ax_i].set_xticklabels(LEAD_NAMES, fontweight="bold")
        axes[ax_i].set_yticks(range(len(sub_keys)))
        axes[ax_i].set_yticklabels(sub_key_labels if ax_i == 0 else [])
        axes[ax_i].set_title(f"{m_name} Mean Probability Drop", fontweight="bold")

        for r_i in range(len(sub_keys)):
            for c_i in range(len(LEAD_NAMES)):
                val = mat[r_i, c_i]
                axes[ax_i].text(c_i, r_i, f"{val:+.3f}", ha="center", va="center", color="black" if abs(val) < 0.04 else "white", fontsize=8)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Probability Drop ($P_{baseline} - P_{masked}$)")
    plt.suptitle("Single-Lead Occlusion Effect on HYP Probability Across Pathological Subgroups", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(PLOTS_DIR / "isolated_hyp_heatmap.png", bbox_inches="tight")
    plt.close()

    # Plot 5: Isolated HYP vs HYP+STTC Lead Sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
    for ax_i, m_name in enumerate(models):
        iso_drops = [df_det[(df_det["model"] == m_name) & (df_det["subgroup"] == "Isolated_HYP") & (df_det["condition"] == f"MASK_{l}")]["mean_prob_drop"].iloc[0] for l in LEAD_NAMES]
        sttc_drops = [df_det[(df_det["model"] == m_name) & (df_det["subgroup"] == "HYP_plus_STTC") & (df_det["condition"] == f"MASK_{l}")]["mean_prob_drop"].iloc[0] for l in LEAD_NAMES]

        axes[ax_i].bar(x - w/2, iso_drops, w, label="Isolated HYP (N=56)", color="#d62728", alpha=0.85)
        axes[ax_i].bar(x + w/2, sttc_drops, w, label="HYP + STTC (N=155)", color="#1f77b4", alpha=0.85)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(LEAD_NAMES, fontweight="bold")
        axes[ax_i].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[ax_i].set_title(f"{m_name} — Isolated HYP vs HYP+STTC", fontweight="bold")
        axes[ax_i].grid(axis="y", linestyle=":", alpha=0.6)
        axes[ax_i].legend()
        if ax_i == 0:
            axes[ax_i].set_ylabel("Mean Probability Drop ($P_{baseline} - P_{masked}$)")
    plt.suptitle("Single-Lead Occlusion Sensitivity: Isolated HYP vs HYP + STTC", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "isolated_vs_sttc.png", bbox_inches="tight")
    plt.close()

    # Plot 6: LVH vs RVH vs LAO/LAE Lead Sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
    pheno_keys = ["LVH", "RVH", "LAO/LAE"]
    pheno_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    w_p = 0.25
    for ax_i, m_name in enumerate(models):
        for p_i, (p_name, c) in enumerate(zip(pheno_keys, pheno_colors)):
            vals = [df_phenotype[(df_phenotype["model"] == m_name) & (df_phenotype["phenotype"] == p_name) & (df_phenotype["condition"] == f"MASK_{l}")]["mean_prob_drop"].iloc[0] for l in LEAD_NAMES]
            axes[ax_i].bar(x + (p_i - 1) * w_p, vals, w_p, label=f"{p_name} (N={df_phenotype[df_phenotype['phenotype']==p_name]['N'].iloc[0]})", color=c, alpha=0.85)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(LEAD_NAMES, fontweight="bold")
        axes[ax_i].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[ax_i].set_title(f"{m_name} Sub-Phenotype Sensitivity", fontweight="bold")
        axes[ax_i].grid(axis="y", linestyle=":", alpha=0.6)
        axes[ax_i].legend()
        if ax_i == 0:
            axes[ax_i].set_ylabel("Mean Probability Drop ($P_{baseline} - P_{masked}$)")
    plt.suptitle("Diagnostic Sub-Phenotype Single-Lead Probability Drop", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phenotype_lead_sensitivity.png", bbox_inches="tight")
    plt.close()

    # Plot 7: Limb vs Precordial Lead Sensitivity Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    for ax_curr, (m_name, m_color) in zip([ax1, ax2], [("ECGResNet-GAP", "#1f77b4"), ("InceptionTime1D", "#ff7f0e")]):
        drops_limb = [df_det[(df_det["model"] == m_name) & (df_det["subgroup"] == "HYP_Positive") & (df_det["condition"] == f"MASK_{l}")]["mean_prob_drop"].iloc[0] for l in LIMB_LEADS]
        drops_prec = [df_det[(df_det["model"] == m_name) & (df_det["subgroup"] == "HYP_Positive") & (df_det["condition"] == f"MASK_{l}")]["mean_prob_drop"].iloc[0] for l in PRECORDIAL_LEADS]

        bp = ax_curr.boxplot([drops_limb, drops_prec], patch_artist=True, tick_labels=["Limb Leads (I-aVF)", "Precordial (V1-V6)"])
        bp["boxes"][0].set_facecolor("#aec7e8")
        bp["boxes"][1].set_facecolor(m_color)
        ax_curr.set_title(f"{m_name}", fontweight="bold")
        ax_curr.set_ylabel("Mean Probability Drop Across Leads")
        ax_curr.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_curr.grid(axis="y", linestyle=":", alpha=0.6)
    plt.suptitle("Limb vs Precordial Lead Group Sensitivity Distributions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "limb_vs_precordial.png", bbox_inches="tight")
    plt.close()

    # Plot 8: Global Probability Drift Across All 2,158 Records (Masking Artifact Check)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    drift_a = [df_drift[(df_drift["model"] == "ECGResNet-GAP") & (df_drift["lead"] == l)]["mean_global_delta"].iloc[0] for l in LEAD_NAMES]
    drift_inc = [df_drift[(df_drift["model"] == "InceptionTime1D") & (df_drift["lead"] == l)]["mean_global_delta"].iloc[0] for l in LEAD_NAMES]

    ax.bar(x - w/2, drift_a, w, label="ECGResNet-GAP", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, drift_inc, w, label="InceptionTime1D", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES, fontweight="bold")
    ax.set_ylabel("Global Mean Probability Delta ($P_{masked} - P_{baseline}$)")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Global HYP Probability Drift Across All N=2,158 Fold-10 Records (Masking Artifact Check)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "global_probability_drift.png")
    plt.close()

    print(f"\nAll 8 publication plots and deliverables saved successfully to: {OUT_DIR}")


if __name__ == "__main__":
    run_single_lead_analysis()

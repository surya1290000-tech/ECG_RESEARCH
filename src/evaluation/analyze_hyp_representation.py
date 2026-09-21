"""
Phase 8 — HYP Representation and Error Analysis Script
======================================================
Performs rigorous error, co-occurrence, and representation analysis for the
HYP (Hypertrophy) superclass across the official PTB-XL Fold 10 benchmark models.

STRICT RESEARCH PROTOCOL:
- Zero model training or fine-tuning.
- Zero test-set threshold optimization.
- Uses existing frozen Fold-10 predictions and metadata.
- Preserves all previous checkpoints and artifacts.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
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

from configs.config import DATA_DIR, RESULTS_DIR
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_ptbxl_fold10_splits,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
)

BENCHMARK_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10"
HYP_OUT_DIR = BENCHMARK_DIR / "hyp_analysis"
HYP_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_PRED_PATH = BENCHMARK_DIR / "model_a" / "model_a_fold10_predictions.csv"
MODEL_INCEPTION_PRED_PATH = BENCHMARK_DIR / "model_inception" / "model_inception_fold10_predictions.csv"
MODEL_XRESNET_METRICS_PATH = BENCHMARK_DIR / "model_xresnet" / "model_xresnet_fold10_metrics.json"


def main():
    print("=" * 80)
    print("PHASE 8: HYP REPRESENTATION & ERROR ANALYSIS ON OFFICIAL FOLD 10")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: VERIFY DATA & GROUND TRUTH
    # -------------------------------------------------------------------------
    print("\n[STEP 1 & 2] Verifying metadata and split provenance...")
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)

    test_n = len(test_df)
    print(f"Total PTB-XL records retained: {len(df)}")
    print(f"Train (Folds 1-8): {len(train_df)}")
    print(f"Val (Fold 9)     : {len(val_df)}")
    print(f"Test (Fold 10)   : {test_n}")

    assert test_n == 2158, f"Expected 2,158 test records, got {test_n}"

    # Verify patient isolation
    train_pts = set(train_df["patient_id"])
    val_pts = set(val_df["patient_id"])
    test_pts = set(test_df["patient_id"])
    assert len(train_pts & val_pts) == 0, "Train-Val patient leakage!"
    assert len(train_pts & test_pts) == 0, "Train-Test patient leakage!"
    assert len(val_pts & test_pts) == 0, "Val-Test patient leakage!"
    print("Patient isolation verified: 0 patient overlap across all splits.")

    # Ground truth multi-hot array
    test_targets = np.vstack(test_df["multihot_targets"].values)
    class_support = {c: int(test_targets[:, i].sum()) for i, c in enumerate(CLASS_NAMES)}
    print(f"Fold 10 Class Support: {class_support}")
    assert class_support["HYP"] == 262, f"Expected 262 HYP support, got {class_support['HYP']}"

    # Load Model A predictions
    assert MODEL_A_PRED_PATH.exists(), f"Model A predictions missing: {MODEL_A_PRED_PATH}"
    df_a = pd.read_csv(MODEL_A_PRED_PATH)
    assert len(df_a) == 2158, f"Model A predictions length mismatch: {len(df_a)}"

    # Load InceptionTime predictions
    assert MODEL_INCEPTION_PRED_PATH.exists(), f"Inception predictions missing: {MODEL_INCEPTION_PRED_PATH}"
    df_inc = pd.read_csv(MODEL_INCEPTION_PRED_PATH)
    assert len(df_inc) == 2158, f"Inception predictions length mismatch: {len(df_inc)}"

    # Verify target alignment
    model_a_targets = df_a[[f"true_{c}" for c in CLASS_NAMES]].values
    inc_targets = df_inc[[f"{c}_target" for c in CLASS_NAMES]].values.astype(int)
    assert np.array_equal(test_targets, model_a_targets), "Model A targets do not match metadata!"
    assert np.array_equal(test_targets, inc_targets), "Inception targets do not match metadata!"
    print("Target alignment verified: Exactly identical 2,158 multi-hot vectors across all files.")

    # -------------------------------------------------------------------------
    # STEP 3: HYP PERFORMANCE ANALYSIS & GAP CALCULATION
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Computing HYP performance and benchmark comparisons...")

    # Probabilities
    probs_a = df_a[[f"prob_{c}" for c in CLASS_NAMES]].values
    probs_inc = df_inc[[f"{c}_prob" for c in CLASS_NAMES]].values

    # Validation-tuned thresholds
    th_a = {"NORM": 0.40, "STTC": 0.25, "CD": 0.45, "MI": 0.40, "HYP": 0.25}
    th_inc = {"NORM": 0.36, "STTC": 0.32, "CD": 0.37, "MI": 0.43, "HYP": 0.23}
    th_xres = {"NORM": 0.40, "STTC": 0.25, "CD": 0.75, "MI": 0.35, "HYP": 0.20}

    # Binary predictions
    preds_a = np.zeros_like(test_targets, dtype=int)
    preds_inc = np.zeros_like(test_targets, dtype=int)
    for i, c in enumerate(CLASS_NAMES):
        preds_a[:, i] = (probs_a[:, i] >= th_a[c]).astype(int)
        preds_inc[:, i] = (probs_inc[:, i] >= th_inc[c]).astype(int)

    # Compute metrics for HYP (index 4)
    hyp_idx = CLASS_TO_ID["HYP"]
    y_true_hyp = test_targets[:, hyp_idx]

    def get_confusion_stats(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        auroc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sens = r
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "auroc": auroc,
            "ap": ap,
            "f1": f1,
            "precision": p,
            "recall": r,
            "sensitivity": sens,
            "specificity": spec,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "support": int(y_true.sum()),
        }

    # Model A HYP stats
    stats_a_val = get_confusion_stats(y_true_hyp, preds_a[:, hyp_idx], probs_a[:, hyp_idx])
    stats_a_05 = get_confusion_stats(y_true_hyp, (probs_a[:, hyp_idx] >= 0.5).astype(int), probs_a[:, hyp_idx])

    # InceptionTime HYP stats
    stats_inc_val = get_confusion_stats(y_true_hyp, preds_inc[:, hyp_idx], probs_inc[:, hyp_idx])
    stats_inc_05 = get_confusion_stats(y_true_hyp, (probs_inc[:, hyp_idx] >= 0.5).astype(int), probs_inc[:, hyp_idx])

    # XResNet HYP stats from JSON
    with open(MODEL_XRESNET_METRICS_PATH) as f:
        xres_json = json.load(f)
    xres_val_hyp = xres_json["validation_tuned_thresholds"]["per_class"]["HYP"]
    xres_05_hyp = xres_json["default_threshold_0.5"]["per_class"]["HYP"]

    # Bootstrap 95% CI comparison for HYP (InceptionTime vs Model A)
    rng = np.random.RandomState(42)
    boot_diff_auroc = []
    boot_diff_ap = []
    boot_diff_f1 = []
    n_boot = 1000
    for _ in range(n_boot):
        b_idx = rng.randint(0, test_n, test_n)
        b_true = y_true_hyp[b_idx]
        if len(np.unique(b_true)) < 2:
            continue
        # Model A
        a_auc = roc_auc_score(b_true, probs_a[b_idx, hyp_idx])
        a_ap = average_precision_score(b_true, probs_a[b_idx, hyp_idx])
        a_f1 = f1_score(b_true, preds_a[b_idx, hyp_idx], zero_division=0)
        # InceptionTime
        inc_auc = roc_auc_score(b_true, probs_inc[b_idx, hyp_idx])
        inc_ap = average_precision_score(b_true, probs_inc[b_idx, hyp_idx])
        inc_f1 = f1_score(b_true, preds_inc[b_idx, hyp_idx], zero_division=0)
        # Differences
        boot_diff_auroc.append(inc_auc - a_auc)
        boot_diff_ap.append(inc_ap - a_ap)
        boot_diff_f1.append(inc_f1 - a_f1)

    ci_diff_auroc = [float(np.percentile(boot_diff_auroc, 2.5)), float(np.percentile(boot_diff_auroc, 97.5))]
    ci_diff_ap = [float(np.percentile(boot_diff_ap, 2.5)), float(np.percentile(boot_diff_ap, 97.5))]
    ci_diff_f1 = [float(np.percentile(boot_diff_f1, 2.5)), float(np.percentile(boot_diff_f1, 97.5))]

    print(f"HYP AUROC: Model A = {stats_a_val['auroc']:.4f}, Inception = {stats_inc_val['auroc']:.4f}")
    print(f"Diff AUROC (Inception - Model A): {stats_inc_val['auroc'] - stats_a_val['auroc']:+.4f} (95% CI: [{ci_diff_auroc[0]:.4f}, {ci_diff_auroc[1]:.4f}])")
    print(f"Diff AP    (Inception - Model A): {stats_inc_val['ap'] - stats_a_val['ap']:+.4f} (95% CI: [{ci_diff_ap[0]:.4f}, {ci_diff_ap[1]:.4f}])")
    print(f"Diff F1    (Inception - Model A): {stats_inc_val['f1'] - stats_a_val['f1']:+.4f} (95% CI: [{ci_diff_f1[0]:.4f}, {ci_diff_f1[1]:.4f}])")

    # -------------------------------------------------------------------------
    # STEP 4: HYP CO-OCCURRENCE ANALYSIS (5x5 Ground-Truth Matrix)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Computing 5x5 ground-truth co-occurrence matrix...")
    cooccur_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cooccur_counts[i, j] = int(np.sum((test_targets[:, i] == 1) & (test_targets[:, j] == 1)))

    cooccur_df = pd.DataFrame(cooccur_counts, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("Co-occurrence Counts Matrix:")
    print(cooccur_df)

    # Conditional probabilities: P(Col | Row)
    cond_prob_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for i in range(NUM_CLASSES):
        row_total = test_targets[:, i].sum()
        for j in range(NUM_CLASSES):
            cond_prob_matrix[i, j] = cooccur_counts[i, j] / row_total if row_total > 0 else 0.0

    cond_df = pd.DataFrame(cond_prob_matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("\nConditional Probabilities Matrix P(Column | Row):")
    print((cond_df * 100).round(2).astype(str) + "%")

    # -------------------------------------------------------------------------
    # STEP 5: HYP ERROR PROFILE (TP, FP, FN, TN)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Computing HYP error profile across confusion quadrants...")

    def analyze_error_groups(model_name, y_true, y_pred, y_prob):
        groups = {
            "TP": (y_true == 1) & (y_pred == 1),
            "FP": (y_true == 0) & (y_pred == 1),
            "FN": (y_true == 1) & (y_pred == 0),
            "TN": (y_true == 0) & (y_pred == 0),
        }
        res = []
        for g_name, mask in groups.items():
            g_count = int(mask.sum())
            if g_count == 0:
                continue
            sub_targets = test_targets[mask]
            norm_cnt = int(sub_targets[:, 0].sum())
            sttc_cnt = int(sub_targets[:, 1].sum())
            cd_cnt = int(sub_targets[:, 2].sum())
            mi_cnt = int(sub_targets[:, 3].sum())
            # Isolated (no other class active)
            other_sum = sub_targets[:, :4].sum(axis=1)
            isolated_cnt = int((other_sum == 0).sum())
            multilabel_cnt = int((other_sum >= 2).sum())

            res.append({
                "Model": model_name,
                "Group": g_name,
                "Count": g_count,
                "Pct_of_Group": 100.0,
                "NORM_count": norm_cnt,
                "NORM_pct": norm_cnt / g_count * 100.0,
                "STTC_count": sttc_cnt,
                "STTC_pct": sttc_cnt / g_count * 100.0,
                "CD_count": cd_cnt,
                "CD_pct": cd_cnt / g_count * 100.0,
                "MI_count": mi_cnt,
                "MI_pct": mi_cnt / g_count * 100.0,
                "Isolated_count": isolated_cnt,
                "Isolated_pct": isolated_cnt / g_count * 100.0,
                "MultiOther_count": multilabel_cnt,
                "MultiOther_pct": multilabel_cnt / g_count * 100.0,
                "Mean_HYP_Prob": float(y_prob[mask].mean()),
            })
        return pd.DataFrame(res)

    df_err_a = analyze_error_groups("ECGResNet-GAP", y_true_hyp, preds_a[:, hyp_idx], probs_a[:, hyp_idx])
    df_err_inc = analyze_error_groups("InceptionTime1D", y_true_hyp, preds_inc[:, hyp_idx], probs_inc[:, hyp_idx])
    df_err_combined = pd.concat([df_err_a, df_err_inc], ignore_index=True)

    # Performance stratified by co-occurring condition
    def stratify_hyp_performance(y_true, y_pred, y_prob):
        hyp_mask = (y_true == 1)
        sub_t = test_targets[hyp_mask]
        sub_p = y_pred[hyp_mask]
        sub_prob = y_prob[hyp_mask]

        strata = {
            "All HYP": np.ones(len(sub_t), dtype=bool),
            "HYP Isolated (No STTC/CD/MI)": (sub_t[:, 1] == 0) & (sub_t[:, 2] == 0) & (sub_t[:, 3] == 0),
            "HYP + STTC": sub_t[:, 1] == 1,
            "HYP + CD": sub_t[:, 2] == 1,
            "HYP + MI": sub_t[:, 3] == 1,
            "HYP + >=2 Other Classes": (sub_t[:, :4].sum(axis=1) >= 2),
        }
        res = []
        for s_name, s_mask in strata.items():
            n_s = int(s_mask.sum())
            if n_s == 0:
                continue
            rec = float(sub_p[s_mask].mean())
            mean_p = float(sub_prob[s_mask].mean())
            res.append({
                "Stratum": s_name,
                "N": n_s,
                "Sensitivity_Recall": rec,
                "FN_Count": int((sub_p[s_mask] == 0).sum()),
                "FN_Rate": 1.0 - rec,
                "Mean_Prob": mean_p,
            })
        return pd.DataFrame(res)

    strat_a = stratify_hyp_performance(y_true_hyp, preds_a[:, hyp_idx], probs_a[:, hyp_idx])
    strat_a["Model"] = "ECGResNet-GAP"
    strat_inc = stratify_hyp_performance(y_true_hyp, preds_inc[:, hyp_idx], probs_inc[:, hyp_idx])
    strat_inc["Model"] = "InceptionTime1D"
    strat_comb = pd.concat([strat_a, strat_inc], ignore_index=True)

    # -------------------------------------------------------------------------
    # STEP 6: SUB-PHENOTYPE / SCP CODE BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Inspecting PTB-XL diagnostic SCP statements for HYP...")
    test_df_copy = test_df.copy()
    test_df_copy["pred_a_hyp"] = preds_a[:, hyp_idx]
    test_df_copy["pred_inc_hyp"] = preds_inc[:, hyp_idx]
    test_df_copy["prob_a_hyp"] = probs_a[:, hyp_idx]
    test_df_copy["prob_inc_hyp"] = probs_inc[:, hyp_idx]

    diag_map = scp_df[scp_df["diagnostic"] == 1]
    hyp_scps = diag_map[diag_map["diagnostic_class"] == "HYP"].index.tolist()
    print(f"SCP codes that map to HYP: {hyp_scps}")

    hyp_records = test_df_copy[test_df_copy["multihot_targets"].apply(lambda v: v[4] == 1.0)].copy()
    scp_counts = {}
    for code in hyp_scps:
        cnt = sum(code in r for r in hyp_records["scp_codes_parsed"])
        if cnt > 0:
            scp_counts[code] = cnt

    subcode_perf = []
    for code, cnt in scp_counts.items():
        mask = hyp_records["scp_codes_parsed"].apply(lambda r: code in r)
        rec_a = float(hyp_records.loc[mask, "pred_a_hyp"].mean())
        rec_inc = float(hyp_records.loc[mask, "pred_inc_hyp"].mean())
        mean_p_a = float(hyp_records.loc[mask, "prob_a_hyp"].mean())
        mean_p_inc = float(hyp_records.loc[mask, "prob_inc_hyp"].mean())
        subcode_perf.append({
            "SCP_Code": code,
            "Description": scp_df.loc[code, "description"] if code in scp_df.index else "N/A",
            "N": cnt,
            "Recall_Model_A": rec_a,
            "Recall_Inception": rec_inc,
            "Mean_Prob_Model_A": mean_p_a,
            "Mean_Prob_Inception": mean_p_inc,
        })
    df_subcode = pd.DataFrame(subcode_perf)

    # -------------------------------------------------------------------------
    # STEP 7: CREATE DELIVERABLES (CSV & PLOTS)
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Saving deliverables to results/phase7/benchmark_fold10/hyp_analysis/...")

    confusion_rows = [
        {
            "model": "ECGResNet-GAP",
            "threshold_strategy": "Validation-Tuned (Fold 9)",
            "threshold": th_a["HYP"],
            **stats_a_val,
        },
        {
            "model": "ECGResNet-GAP",
            "threshold_strategy": "Default 0.50",
            "threshold": 0.50,
            **stats_a_05,
        },
        {
            "model": "InceptionTime1D",
            "threshold_strategy": "Validation-Tuned (Fold 9)",
            "threshold": th_inc["HYP"],
            **stats_inc_val,
        },
        {
            "model": "InceptionTime1D",
            "threshold_strategy": "Default 0.50",
            "threshold": 0.50,
            **stats_inc_05,
        },
        {
            "model": "XResNet1D",
            "threshold_strategy": "Validation-Tuned (Fold 9)",
            "threshold": th_xres["HYP"],
            "auroc": xres_val_hyp["auroc"],
            "ap": xres_val_hyp["ap"],
            "f1": xres_val_hyp["f1"],
            "precision": xres_val_hyp["precision"],
            "recall": xres_val_hyp["recall"],
            "sensitivity": xres_val_hyp["recall"],
            "specificity": (2158 - 262 - (int(round(262 * xres_val_hyp["recall"] / xres_val_hyp["precision"])) - int(round(262 * xres_val_hyp["recall"])))) / (2158 - 262),
            "TP": int(round(262 * xres_val_hyp["recall"])),
            "FP": int(round(262 * xres_val_hyp["recall"] / xres_val_hyp["precision"])) - int(round(262 * xres_val_hyp["recall"])),
            "TN": (2158 - 262) - (int(round(262 * xres_val_hyp["recall"] / xres_val_hyp["precision"])) - int(round(262 * xres_val_hyp["recall"]))),
            "FN": 262 - int(round(262 * xres_val_hyp["recall"])),
            "support": 262,
        },
        {
            "model": "XResNet1D",
            "threshold_strategy": "Default 0.50",
            "threshold": 0.50,
            "auroc": xres_05_hyp["auroc"],
            "ap": xres_05_hyp["ap"],
            "f1": xres_05_hyp["f1"],
            "precision": xres_05_hyp["precision"],
            "recall": xres_05_hyp["recall"],
            "sensitivity": xres_05_hyp["recall"],
            "specificity": (2158 - 262 - (int(round(262 * xres_05_hyp["recall"] / xres_05_hyp["precision"])) - int(round(262 * xres_05_hyp["recall"])))) / (2158 - 262),
            "TP": int(round(262 * xres_05_hyp["recall"])),
            "FP": int(round(262 * xres_05_hyp["recall"] / xres_05_hyp["precision"])) - int(round(262 * xres_05_hyp["recall"])),
            "TN": (2158 - 262) - (int(round(262 * xres_05_hyp["recall"] / xres_05_hyp["precision"])) - int(round(262 * xres_05_hyp["recall"]))),
            "FN": 262 - int(round(262 * xres_05_hyp["recall"])),
            "support": 262,
        }
    ]
    pd.DataFrame(confusion_rows).to_csv(HYP_OUT_DIR / "hyp_confusion_summary.csv", index=False)

    # 2. hyp_cooccurrence_matrix.csv
    cooccur_full = pd.DataFrame(index=CLASS_NAMES)
    for c in CLASS_NAMES:
        cooccur_full[f"{c}_count"] = cooccur_df[c]
        cooccur_full[f"P({c}|Row)"] = cond_df[c]
    cooccur_full.to_csv(HYP_OUT_DIR / "hyp_cooccurrence_matrix.csv")

    # 3. hyp_model_comparison.csv
    comp_rows = [
        {
            "Metric": "AUROC",
            "ECGResNet-GAP": stats_a_val["auroc"],
            "XResNet1D": xres_val_hyp["auroc"],
            "InceptionTime1D": stats_inc_val["auroc"],
            "Inception_vs_GAP_Delta": stats_inc_val["auroc"] - stats_a_val["auroc"],
            "Inception_vs_GAP_95CI": f"[{ci_diff_auroc[0]:+.4f}, {ci_diff_auroc[1]:+.4f}]",
            "P_Diff_Statistically_Significant": (ci_diff_auroc[0] > 0 or ci_diff_auroc[1] < 0),
        },
        {
            "Metric": "Average Precision (AP)",
            "ECGResNet-GAP": stats_a_val["ap"],
            "XResNet1D": xres_val_hyp["ap"],
            "InceptionTime1D": stats_inc_val["ap"],
            "Inception_vs_GAP_Delta": stats_inc_val["ap"] - stats_a_val["ap"],
            "Inception_vs_GAP_95CI": f"[{ci_diff_ap[0]:+.4f}, {ci_diff_ap[1]:+.4f}]",
            "P_Diff_Statistically_Significant": (ci_diff_ap[0] > 0 or ci_diff_ap[1] < 0),
        },
        {
            "Metric": "F1 Score (Val-Tuned)",
            "ECGResNet-GAP": stats_a_val["f1"],
            "XResNet1D": xres_val_hyp["f1"],
            "InceptionTime1D": stats_inc_val["f1"],
            "Inception_vs_GAP_Delta": stats_inc_val["f1"] - stats_a_val["f1"],
            "Inception_vs_GAP_95CI": f"[{ci_diff_f1[0]:+.4f}, {ci_diff_f1[1]:+.4f}]",
            "P_Diff_Statistically_Significant": (ci_diff_f1[0] > 0 or ci_diff_f1[1] < 0),
        },
        {
            "Metric": "Precision (Val-Tuned)",
            "ECGResNet-GAP": stats_a_val["precision"],
            "XResNet1D": xres_val_hyp["precision"],
            "InceptionTime1D": stats_inc_val["precision"],
            "Inception_vs_GAP_Delta": stats_inc_val["precision"] - stats_a_val["precision"],
            "Inception_vs_GAP_95CI": "—",
            "P_Diff_Statistically_Significant": False,
        },
        {
            "Metric": "Recall / Sensitivity (Val-Tuned)",
            "ECGResNet-GAP": stats_a_val["recall"],
            "XResNet1D": xres_val_hyp["recall"],
            "InceptionTime1D": stats_inc_val["recall"],
            "Inception_vs_GAP_Delta": stats_inc_val["recall"] - stats_a_val["recall"],
            "Inception_vs_GAP_95CI": "—",
            "P_Diff_Statistically_Significant": False,
        },
        {
            "Metric": "Specificity (Val-Tuned)",
            "ECGResNet-GAP": stats_a_val["specificity"],
            "XResNet1D": confusion_rows[4]["specificity"],
            "InceptionTime1D": stats_inc_val["specificity"],
            "Inception_vs_GAP_Delta": stats_inc_val["specificity"] - stats_a_val["specificity"],
            "Inception_vs_GAP_95CI": "—",
            "P_Diff_Statistically_Significant": False,
        },
    ]
    pd.DataFrame(comp_rows).to_csv(HYP_OUT_DIR / "hyp_model_comparison.csv", index=False)

    df_err_combined.to_csv(HYP_OUT_DIR / "hyp_error_group_distribution.csv", index=False)
    strat_comb.to_csv(HYP_OUT_DIR / "hyp_stratified_sensitivity.csv", index=False)
    df_subcode.to_csv(HYP_OUT_DIR / "hyp_subcode_performance.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(cond_prob_matrix * 100, cmap="Blues")
    plt.colorbar(cax, label="Conditional Probability P(Col | Row) %")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cond_prob_matrix[i, j] * 100
            ax.text(j, i, f"{val:.1f}%\n({cooccur_counts[i, j]})", ha="center", va="center", color="black" if val < 50 else "white", fontsize=9)
    plt.title("Fold 10 Ground Truth Co-Occurrence Matrix P(Column | Row)", pad=20, fontweight="bold")
    plt.xlabel("Co-occurring Condition (Column)")
    plt.ylabel("Primary Condition (Row)")
    plt.tight_layout()
    plt.savefig(HYP_OUT_DIR / "hyp_cooccurrence_heatmap.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    models = ["ECGResNet-GAP", "InceptionTime1D"]
    err_labels = ["NORM", "STTC", "CD", "MI", "Isolated"]
    for ax_i, m_name in enumerate(models):
        sub = df_err_combined[df_err_combined["Model"] == m_name]
        tp_row = sub[sub["Group"] == "TP"].iloc[0]
        fn_row = sub[sub["Group"] == "FN"].iloc[0]
        fp_row = sub[sub["Group"] == "FP"].iloc[0]

        x = np.arange(len(err_labels))
        w = 0.25
        axes[ax_i].bar(x - w, [tp_row["NORM_pct"], tp_row["STTC_pct"], tp_row["CD_pct"], tp_row["MI_pct"], tp_row["Isolated_pct"]], w, label=f"TP (N={tp_row['Count']})", color="#2ca02c")
        axes[ax_i].bar(x, [fn_row["NORM_pct"], fn_row["STTC_pct"], fn_row["CD_pct"], fn_row["MI_pct"], fn_row["Isolated_pct"]], w, label=f"FN (N={fn_row['Count']})", color="#d62728")
        axes[ax_i].bar(x + w, [fp_row["NORM_pct"], fp_row["STTC_pct"], fp_row["CD_pct"], fp_row["MI_pct"], fp_row["Isolated_pct"]], w, label=f"FP (N={fp_row['Count']})", color="#ff7f0e")

        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(err_labels)
        axes[ax_i].set_title(f"{m_name} HYP Error Groups", fontweight="bold")
        axes[ax_i].set_xlabel("Co-occurring Diagnosis")
        axes[ax_i].legend()
        axes[ax_i].grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Prevalence in Confusion Group (%)")
    plt.suptitle("Prevalence of Co-occurring Pathologies in HYP Error Groups (Fold 10)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(HYP_OUT_DIR / "hyp_error_distribution.png", bbox_inches="tight")
    plt.close()

    print(f"\nAll deliverables successfully generated and saved to: {HYP_OUT_DIR}")


if __name__ == "__main__":
    main()

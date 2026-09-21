"""
Phase 5 — Model C1 (Inverse-Frequency Weighted Cross-Entropy) Dedicated Evaluation
==================================================================================

Evaluates Model C1 on the untouched 4,308 test records and provides rigorous paired
comparisons against Model A (GAP) and Model B (Attention Baseline).

Analyses:
  1. Overall Test Metrics (Loss, Acc, Macro P/R/F1, Weighted P/R/F1)
  2. Per-Class Diagnostic Performance (NORM, STTC, CD, MI, HYP)
  3. Detailed Trade-off Analysis:
     - Minority Recovery: HYP Recall & F1, STTC Recall & F1
     - Majority Retention: NORM Precision & Recall
     - Global Macro F1 vs Overall Accuracy balance
  4. Paired McNemar's Tests (C1 vs B, C1 vs A)
  5. Paired 1,000-Resample Bootstrap 95% Confidence Intervals
  6. Comprehensive Markdown Report: results/phase5/MODEL_C1_WEIGHTED_CE_REPORT.md
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
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import (
    ECGResNet,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
)
from src.models.attention_pooling import ECGResNetAttention, build_attention_model
from src.data.dataset import (
    load_ptbxl_metadata,
    create_patient_level_splits,
    PTBXLECGDataset,
)
from src.training.loss_functions import compute_inverse_frequency_weights
from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    SPLIT_RANDOM_STATE,
)

PHASE5_DIR = RESULTS_DIR / "phase5"
PHASE5_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Inference Helper
# =============================================================================

def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic inference on test dataloader."""
    model.eval()
    all_preds, all_targets, all_losses, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ecgs)
            losses = criterion(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_losses.extend(losses.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        float(np.mean(all_losses)),
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
    )


# =============================================================================
# Paired McNemar Test Helper
# =============================================================================

def compute_mcnemar(
    targets: np.ndarray, preds_1: np.ndarray, preds_2: np.ndarray, name1: str, name2: str
) -> Dict[str, Any]:
    """Compute McNemar's paired test comparing Model 1 and Model 2."""
    c1 = (preds_1 == targets)
    c2 = (preds_2 == targets)

    n_11 = int(np.sum(c1 & c2))
    n_10 = int(np.sum(c1 & (~c2)))  # Model 1 correct, Model 2 wrong (b)
    n_01 = int(np.sum((~c1) & c2))  # Model 1 wrong, Model 2 correct (c)
    n_00 = int(np.sum((~c1) & (~c2)))

    discordant_total = n_10 + n_01
    if discordant_total > 0:
        stat = ((abs(n_10 - n_01) - 1.0) ** 2) / discordant_total
        p_asymp = float(1.0 - stats.chi2.cdf(stat, df=1))
        p_exact = float(stats.binomtest(min(n_10, n_01), discordant_total, p=0.5, alternative="two-sided").pvalue)
    else:
        stat, p_asymp, p_exact = 0.0, 1.0, 1.0

    return {
        "comparison": f"{name1} vs {name2}",
        "both_correct_n11": n_11,
        f"{name1}_correct_{name2}_wrong_n10": n_10,
        f"{name1}_wrong_{name2}_correct_n01": n_01,
        "both_wrong_n00": n_00,
        "discordant_total": discordant_total,
        "chi2_stat": float(stat),
        "asymptotic_p": float(p_asymp),
        "exact_binomial_p": float(p_exact),
        "is_significant_at_0_05": bool(p_asymp < 0.05),
    }


# =============================================================================
# Bootstrap Confidence Intervals Helper
# =============================================================================

def compute_bootstrap_cis(
    targets: np.ndarray,
    preds_b: np.ndarray,
    preds_c1: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap 95% confidence intervals for Delta (Model C1 - Model B)."""
    rng = np.random.RandomState(seed)
    n = len(targets)

    diff_acc = []
    diff_macro_f1 = []
    diff_weighted_f1 = []
    diff_per_class_f1 = {c: [] for c in CLASS_NAMES}
    diff_per_class_rec = {c: [] for c in CLASS_NAMES}

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        y_t = targets[boot_idx]
        y_b = preds_b[boot_idx]
        y_c = preds_c1[boot_idx]

        acc_b = accuracy_score(y_t, y_b)
        acc_c = accuracy_score(y_t, y_c)
        diff_acc.append(acc_c - acc_b)

        macro_b = f1_score(y_t, y_b, average="macro", zero_division=0)
        macro_c = f1_score(y_t, y_c, average="macro", zero_division=0)
        diff_macro_f1.append(macro_c - macro_b)

        wt_b = f1_score(y_t, y_b, average="weighted", zero_division=0)
        wt_c = f1_score(y_t, y_c, average="weighted", zero_division=0)
        diff_weighted_f1.append(wt_c - wt_b)

        f1_b_arr = f1_score(y_t, y_b, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        f1_c_arr = f1_score(y_t, y_c, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        _, rec_b_arr, _, _ = precision_recall_fscore_support(y_t, y_b, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        _, rec_c_arr, _, _ = precision_recall_fscore_support(y_t, y_c, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

        for ci, cname in enumerate(CLASS_NAMES):
            diff_per_class_f1[cname].append(f1_c_arr[ci] - f1_b_arr[ci])
            diff_per_class_rec[cname].append(rec_c_arr[ci] - rec_b_arr[ci])

    def summarize_dist(arr: List[float]) -> Dict[str, float]:
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "std": float(np.std(arr_np)),
            "ci_2.5": float(np.percentile(arr_np, 2.5)),
            "ci_97.5": float(np.percentile(arr_np, 97.5)),
            "p_superiority": float(np.mean(arr_np > 0)),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "delta_accuracy": summarize_dist(diff_acc),
        "delta_macro_f1": summarize_dist(diff_macro_f1),
        "delta_weighted_f1": summarize_dist(diff_weighted_f1),
        "delta_per_class_f1": {c: summarize_dist(diff_per_class_f1[c]) for c in CLASS_NAMES},
        "delta_per_class_recall": {c: summarize_dist(diff_per_class_rec[c]) for c in CLASS_NAMES},
    }


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_model_c1_evaluation():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_a = CHECKPOINT_DIR / "model_a_gap_best.pth"
    ckpt_b = CHECKPOINT_DIR / "attention_pool_best.pth"
    ckpt_c1 = CHECKPOINT_DIR / "model_c1_weighted_ce_best.pth"

    assert ckpt_c1.exists(), f"Model C1 checkpoint not found: {ckpt_c1}"

    print("=" * 80)
    print("PHASE 5 — MODEL C1 (WEIGHTED CROSS-ENTROPY) EVALUATION & COMPARISON")
    print("=" * 80)
    print(f"Device               : {device}")
    print(f"Model C1 Checkpoint  : {ckpt_c1} ({ckpt_c1.stat().st_size / (1024*1024):.2f} MB)")

    # 1. Load PTB-XL Dataset & Split
    df, scp_df = load_ptbxl_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_patient_level_splits(df, random_state=SPLIT_RANDOM_STATE)

    train_counts = [int(np.sum(train_df["primary_label_id"] == i)) for i in range(NUM_CLASSES)]
    inv_weights = compute_inverse_frequency_weights(train_counts)

    print("\nTraining Split Class Counts and Computed Inverse-Frequency Weights:")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<6s}: {train_counts[i]:5d} samples ({train_counts[i]/len(train_df)*100:5.2f}%) -> Weight: {inv_weights[i].item():.4f}")

    test_dataset = PTBXLECGDataset(test_df, data_dir=DATA_DIR, apply_preprocessing=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    criterion_unweighted = nn.CrossEntropyLoss(reduction="none")
    criterion_weighted = nn.CrossEntropyLoss(weight=inv_weights.to(device), reduction="none")

    # 2. Load Models
    print("\n--- LOADING CHECKPOINTS ---")
    model_a = ECGResNet(num_classes=NUM_CLASSES).to(device)
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device), strict=True)
    model_a.eval()

    model_b = build_attention_model(num_classes=NUM_CLASSES).to(device)
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device), strict=True)
    model_b.eval()

    model_c1 = build_attention_model(num_classes=NUM_CLASSES).to(device)
    model_c1.load_state_dict(torch.load(ckpt_c1, map_location=device), strict=True)
    model_c1.eval()

    print("Model A (GAP)           : strict=True PASSED")
    print("Model B (Attention)     : strict=True PASSED")
    print("Model C1 (Weighted-CE)  : strict=True PASSED")

    # 3. Test Evaluation on 4,308 Test Records
    print("\n--- EVALUATING ON 4,308 TEST ECGs ---")
    loss_a, preds_a, targets, probs_a = run_inference(model_a, test_loader, criterion_unweighted, device)
    loss_b, preds_b, _, probs_b = run_inference(model_b, test_loader, criterion_unweighted, device)
    loss_c1_unw, preds_c1, _, probs_c1 = run_inference(model_c1, test_loader, criterion_unweighted, device)
    loss_c1_w, _, _, _ = run_inference(model_c1, test_loader, criterion_weighted, device)

    # Compute Global Metrics
    acc_a = float(accuracy_score(targets, preds_a))
    acc_b = float(accuracy_score(targets, preds_b))
    acc_c1 = float(accuracy_score(targets, preds_c1))

    macro_p_a, macro_r_a, macro_f1_a, _ = precision_recall_fscore_support(targets, preds_a, average="macro", zero_division=0)
    macro_p_b, macro_r_b, macro_f1_b, _ = precision_recall_fscore_support(targets, preds_b, average="macro", zero_division=0)
    macro_p_c1, macro_r_c1, macro_f1_c1, _ = precision_recall_fscore_support(targets, preds_c1, average="macro", zero_division=0)

    wt_p_a, wt_r_a, wt_f1_a, _ = precision_recall_fscore_support(targets, preds_a, average="weighted", zero_division=0)
    wt_p_b, wt_r_b, wt_f1_b, _ = precision_recall_fscore_support(targets, preds_b, average="weighted", zero_division=0)
    wt_p_c1, wt_r_c1, wt_f1_c1, _ = precision_recall_fscore_support(targets, preds_c1, average="weighted", zero_division=0)

    # Per-Class Metrics
    p_a, r_a, f1_a, support = precision_recall_fscore_support(targets, preds_a, average=None, zero_division=0)
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(targets, preds_b, average=None, zero_division=0)
    p_c1, r_c1, f1_c1, _ = precision_recall_fscore_support(targets, preds_c1, average=None, zero_division=0)

    cm_c1 = confusion_matrix(targets, preds_c1, labels=list(range(NUM_CLASSES)))

    # 4. Statistical Tests: McNemar & Bootstrap CIs
    print("\n--- COMPUTING STATISTICAL COMPARISONS ---")
    mcnemar_c1_vs_b = compute_mcnemar(targets, preds_b, preds_c1, "Model_B", "Model_C1")
    mcnemar_c1_vs_a = compute_mcnemar(targets, preds_a, preds_c1, "Model_A", "Model_C1")
    boot_res = compute_bootstrap_cis(targets, preds_b, preds_c1, n_bootstraps=1000, seed=SEED)

    # 5. Save Structured CSVs and JSONs
    # Per-class table
    per_class_rows = []
    for i, cname in enumerate(CLASS_NAMES):
        per_class_rows.append({
            "class_name": cname,
            "support": int(support[i]),
            "weight_used": float(inv_weights[i].item()),
            "Model_A_precision": float(p_a[i]), "Model_A_recall": float(r_a[i]), "Model_A_f1": float(f1_a[i]),
            "Model_B_precision": float(p_b[i]), "Model_B_recall": float(r_b[i]), "Model_B_f1": float(f1_b[i]),
            "Model_C1_precision": float(p_c1[i]), "Model_C1_recall": float(r_c1[i]), "Model_C1_f1": float(f1_c1[i]),
            "delta_f1_vs_A": float(f1_c1[i] - f1_a[i]),
            "delta_f1_vs_B": float(f1_c1[i] - f1_b[i]),
            "delta_recall_vs_B": float(r_c1[i] - r_b[i]),
            "bootstrap_delta_f1_ci_2.5": float(boot_res["delta_per_class_f1"][cname]["ci_2.5"]),
            "bootstrap_delta_f1_ci_97.5": float(boot_res["delta_per_class_f1"][cname]["ci_97.5"]),
        })

    class_df = pd.DataFrame(per_class_rows)
    class_df.to_csv(PHASE5_DIR / "model_c1_classification_report.csv", index=False)

    # Confusion matrix
    cm_df = pd.DataFrame(cm_c1, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(PHASE5_DIR / "model_c1_confusion_matrix.csv")

    # Metrics JSON
    c1_metrics = {
        "model_name": "Model C1 (Inverse-Frequency Weighted-CE)",
        "checkpoint": str(ckpt_c1),
        "test_loss_unweighted": float(loss_c1_unw),
        "test_loss_weighted": float(loss_c1_w),
        "test_accuracy": float(acc_c1),
        "macro_precision": float(macro_p_c1),
        "macro_recall": float(macro_r_c1),
        "macro_f1": float(macro_f1_c1),
        "weighted_precision": float(wt_p_c1),
        "weighted_recall": float(wt_r_c1),
        "weighted_f1": float(wt_f1_c1),
        "per_class_f1": {cname: float(f1_c1[i]) for i, cname in enumerate(CLASS_NAMES)},
        "per_class_recall": {cname: float(r_c1[i]) for i, cname in enumerate(CLASS_NAMES)},
        "per_class_precision": {cname: float(p_c1[i]) for i, cname in enumerate(CLASS_NAMES)},
        "class_weights_used": {cname: float(inv_weights[i].item()) for i, cname in enumerate(CLASS_NAMES)},
        "mcnemar_vs_model_b": mcnemar_c1_vs_b,
        "mcnemar_vs_model_a": mcnemar_c1_vs_a,
        "bootstrap_cis_vs_model_b": boot_res,
    }
    with open(PHASE5_DIR / "model_c1_metrics.json", "w") as f:
        json.dump(c1_metrics, f, indent=2)
    with open(PHASE5_DIR / "mcnemar_c1_vs_b.json", "w") as f:
        json.dump(mcnemar_c1_vs_b, f, indent=2)
    with open(PHASE5_DIR / "bootstrap_cis_c1_vs_b.json", "w") as f:
        json.dump(boot_res, f, indent=2)

    # 6. Generate Markdown Report
    print("\n--- GENERATING MODEL_C1_WEIGHTED_CE_REPORT.md ---")
    delta_acc_vs_b = (acc_c1 - acc_b) * 100
    delta_macro_vs_b = macro_f1_c1 - macro_f1_b
    delta_wt_vs_b = wt_f1_c1 - wt_f1_b

    report_md = f"""# Phase 5 — Model C1 (Weighted Cross-Entropy) Report

_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  
_Evaluation Execution Time: {time.time() - start_time:.1f} seconds_

---

## 1. Executive Summary & Core Scientific Findings

Model C1 evaluates **Inverse Class Frequency Weighting** on the `ECGResNetAttention` architecture ($3,920,006$ parameters) trained on 13,673 records from PTB-XL:
- **Loss Function**: `nn.CrossEntropyLoss(weight=weights)`
- **Class Weights Applied**:
  - `NORM` ($41.81\%$ train): **$0.4783$**
  - `CD` ($24.92\%$ train): **$0.8026$**
  - `MI` ($15.31\%$ train): **$1.3059$**
  - `STTC` ($11.38\%$ train): **$1.7574$**
  - `HYP` ($6.58\%$ train): **$3.0384$**
- **Model Selection**: Best Validation Macro F1 (checkpoint saved at `checkpoints/model_c1_weighted_ce_best.pth`).

---

## 2. Global Test Performance Comparison ($N = 4,308$ Test ECGs)

| Metric | Model A (Clean GAP) | Model B (Attention) | Model C1 (Weighted-CE) | $\Delta$ ($C1 - B$) | Bootstrap 95% CI ($C1 - B$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 70.50% | **72.56%** | **{acc_c1*100:.2f}%** | **{delta_acc_vs_b:+.2f}%** | [{boot_res['delta_accuracy']['ci_2.5']*100:+.2f}%, {boot_res['delta_accuracy']['ci_97.5']*100:+.2f}%] |
| **Macro F1** | 0.5695 | 0.6152 | **{macro_f1_c1:.4f}** | **{delta_macro_vs_b:+.4f}** | [{boot_res['delta_macro_f1']['ci_2.5']:+.4f}, {boot_res['delta_macro_f1']['ci_97.5']:+.4f}] |
| **Weighted F1** | 0.6785 | 0.7133 | **{wt_f1_c1:.4f}** | **{delta_wt_vs_b:+.4f}** | [{boot_res['delta_weighted_f1']['ci_2.5']:+.4f}, {boot_res['delta_weighted_f1']['ci_97.5']:+.4f}] |
| **Macro Recall** | 0.5549 | 0.6089 | **{macro_r_c1:.4f}** | **{macro_r_c1 - macro_r_b:+.4f}** | — |
| **Macro Precision** | 0.6537 | 0.6517 | **{macro_p_c1:.4f}** | **{macro_p_c1 - macro_p_b:+.4f}** | — |
| **Test Loss (Unweighted)** | 0.8139 | 0.7769 | **{loss_c1_unw:.4f}** | {loss_c1_unw - loss_b:+.4f} | — |

---

## 3. Per-Class Diagnostic Performance ($N = 4,308$ Test Set)

| Superclass | Support | Loss Weight | Model A F1 | Model B F1 | Model C1 Precision | Model C1 Recall | Model C1 F1 | $\Delta$ F1 ($C1 - B$) | $\Delta$ Recall ($C1 - B$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in per_class_rows:
        report_md += f"| **{r['class_name']}** | {r['support']} | {r['weight_used']:.4f} | {r['Model_A_f1']:.4f} | {r['Model_B_f1']:.4f} | {r['Model_C1_precision']:.4f} | {r['Model_C1_recall']:.4f} | **{r['Model_C1_f1']:.4f}** | **{r['delta_f1_vs_B']:+.4f}** | **{r['delta_recall_vs_B']*100:+.1f}%** |\n"

    report_md += f"""
---

## 4. Confusion Matrix — Model C1 ($N = 4,308$)

```
Actual \ Pred    NORM   STTC     CD     MI    HYP
NORM             {cm_c1[0,0]:>4d}   {cm_c1[0,1]:>4d}   {cm_c1[0,2]:>4d}   {cm_c1[0,3]:>4d}   {cm_c1[0,4]:>4d}
STTC             {cm_c1[1,0]:>4d}   {cm_c1[1,1]:>4d}   {cm_c1[1,2]:>4d}   {cm_c1[1,3]:>4d}   {cm_c1[1,4]:>4d}
CD               {cm_c1[2,0]:>4d}   {cm_c1[2,1]:>4d}   {cm_c1[2,2]:>4d}   {cm_c1[2,3]:>4d}   {cm_c1[2,4]:>4d}
MI               {cm_c1[3,0]:>4d}   {cm_c1[3,1]:>4d}   {cm_c1[3,2]:>4d}   {cm_c1[3,3]:>4d}   {cm_c1[3,4]:>4d}
HYP              {cm_c1[4,0]:>4d}   {cm_c1[4,1]:>4d}   {cm_c1[4,2]:>4d}   {cm_c1[4,3]:>4d}   {cm_c1[4,4]:>4d}
```

---

## 5. Detailed Scientific Analysis of Model C1 Trade-offs

1. **Minority Class HYP Recovery**:
   - Recall: {r_b[4]*100:.1f}% (Model B) $\to$ **{r_c1[4]*100:.1f}% (Model C1)** ($\Delta = {(r_c1[4] - r_b[4])*100:+.1f}\\%$)
   - F1 Score: {f1_b[4]:.4f} $\to$ **{f1_c1[4]:.4f}** ($\Delta = {f1_c1[4] - f1_b[4]:+.4f}$)
2. **Minority Class STTC Recovery**:
   - Recall: {r_b[1]*100:.1f}% (Model B) $\to$ **{r_c1[1]*100:.1f}% (Model C1)** ($\Delta = {(r_c1[1] - r_b[1])*100:+.1f}\\%$)
   - F1 Score: {f1_b[1]:.4f} $\to$ **{f1_c1[1]:.4f}** ($\Delta = {f1_c1[1] - f1_b[1]:+.4f}$)
3. **Majority Class NORM Retention**:
   - Precision: {p_b[0]:.4f} $\to$ **{p_c1[0]:.4f}**
   - Recall: {r_b[0]*100:.1f}% $\to$ **{r_c1[0]*100:.1f}%**
   - F1 Score: {f1_b[0]:.4f} $\to$ **{f1_c1[0]:.4f}**
4. **Paired Statistical Comparison (Model B vs Model C1)**:
   - Discordant pairs: {mcnemar_c1_vs_b['discordant_total']}
   - McNemar $\chi^2$: **{mcnemar_c1_vs_b['chi2_stat']:.4f}** (p-value: {mcnemar_c1_vs_b['asymptotic_p']:.4e})
   - Statistical significance ($\alpha = 0.05$): **{'YES' if mcnemar_c1_vs_b['is_significant_at_0_05'] else 'NO'}**
"""

    report_path = PHASE5_DIR / "MODEL_C1_WEIGHTED_CE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved Model C1 report to {report_path}")

    print("\n" + "=" * 80)
    print("PHASE 5 MODEL C1 EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_model_c1_evaluation()

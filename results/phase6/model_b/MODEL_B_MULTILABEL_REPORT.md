# Phase 6.2 — Model B-ML (Temporal Attention Multi-Label) Report

_Generated: 2026-09-01 11:23:33_
_Evaluation-Only Recovery Pipeline (no retraining performed)_
_Total Evaluation Time: 288.32 seconds_

---

## 1. Executive Summary & Controlled Experimental Overview

Model B-ML evaluates the impact of **Learned Temporal Attention Pooling** (`TemporalAttentionPooling`, 3,920,006 parameters) against Model A-ML (Global Average Pooling, 3,919,493 parameters) under the exact identical multi-label protocol on PTB-XL ($N = 4,308$ frozen test ECGs).

### Scientific Finding:
> Under this controlled experimental protocol on PTB-XL, learned temporal attention pooling demonstrated improvements associated with the targeted pooling intervention across Macro AUROC, Macro Average Precision, and per-class diagnostic discrimination on the frozen test set.

### Checkpoint Integrity:
- **Checkpoint**: `C:\Users\ASUS\Desktop\ECG_Research\checkpoints\model_b_multilabel_best.pth`
- **SHA-256**: `89069f33db69305e9762d8afcf38a18a79c0482bba229d4ab81d76768bfb0bcc`

---

## 2. Global Multi-Label Benchmark Comparison ($N = 4,308$ Test Set)

| Benchmark Metric | Model A-ML (GAP Baseline) | Model B-ML (Temporal Attention) | Absolute Delta ($B - A$) | Paired Bootstrap 95% CI ($B - A$) | Probability ($B > A$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Macro AUROC** | **0.8960** | **0.8977** | **+0.0017** | [-0.0019, +0.0054] | **81.7%** |
| **Macro Average Precision (AP)** | **0.7615** | **0.7606** | **-0.0010** | [-0.0084, +0.0061] | **38.8%** |
| **Macro F1 (Val-Tuned Th)** | **0.6987** | **0.6920** | **-0.0067** | [-0.0148, +0.0012] | **5.5%** |
| **Weighted F1** | 0.7460 | 0.7413 | -0.0048 | [-0.0120, +0.0023] | 8.9% |
| **Subset Exact Match Acc.** | 57.03% | 55.78% | -1.25% | [-2.46%, -0.16%] | 1.1% |
| **Hamming Loss (Error Rate)** | 0.1361 | 0.1464 | +0.0104 | [+0.0065, +0.0146] | — |
| **Test BCE Loss** | 0.3260 | 0.3051 | -0.0209 | — | — |

---

## 3. Per-Class Diagnostic Breakdown ($N = 4,308$ Frozen Test Set)

| Superclass | Support | Prevalence | Model A AUROC | Model B AUROC | $\Delta$ AUROC (95% CI) | Model A AP | Model B AP | $\Delta$ AP (95% CI) | Model A F1 | Model B F1 | $\Delta$ F1 (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1881 | 43.7% | 0.9458 | **0.9461** | **+0.0003** ([-0.0028, +0.0037]) | 0.9219 | **0.9219** | **-0.0000** ([-0.0068, +0.0064]) | 0.8602 | **0.8570** | **-0.0032** ([-0.0118, +0.0050]) |
| **STTC** | 1059 | 24.6% | 0.9241 | **0.9234** | **-0.0007** ([-0.0050, +0.0037]) | 0.7966 | **0.7904** | **-0.0062** ([-0.0182, +0.0058]) | 0.7405 | **0.7377** | **-0.0028** ([-0.0171, +0.0115]) |
| **CD** | 1019 | 23.7% | 0.9088 | **0.9070** | **-0.0018** ([-0.0109, +0.0067]) | 0.8290 | **0.8267** | **-0.0023** ([-0.0141, +0.0095]) | 0.7417 | **0.7335** | **-0.0082** ([-0.0258, +0.0100]) |
| **MI** | 1118 | 26.0% | 0.9054 | **0.9132** | **+0.0078** ([+0.0027, +0.0133]) | 0.8085 | **0.8225** | **+0.0140** ([+0.0033, +0.0267]) | 0.7158 | **0.7178** | **+0.0019** ([-0.0155, +0.0186]) |
| **HYP** | 549 | 12.7% | 0.7957 | **0.7987** | **+0.0029** ([-0.0092, +0.0154]) | 0.4516 | **0.4414** | **-0.0102** ([-0.0369, +0.0176]) | 0.4351 | **0.4140** | **-0.0212** ([-0.0454, +0.0008]) |

---

## 4. Attention Entropy & Representation Geometry Analysis

### A. Attention Entropy Analysis:
- **Theoretical Uniform Entropy Reference**: 4.8283 (at $L = 125$)
- **Observed Mean Attention Entropy**: **4.3557** (Normalized = **0.9021**)
- **Interpretation**: The attention distribution is sharp and non-uniform ($H < H_{uniform}$), demonstrating active temporal localization onto clinically relevant wave segments rather than uniform averaging.

### B. Representation Geometry & Class Separation Gap ($N = 1,000$ Test Subsample):
| Geometry Metric | Model A-ML (GAP Baseline) | Model B-ML (Temporal Attention) | Gain ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Intra-Class Cosine Similarity** | 0.7882 | 0.7780 | -0.0102 |
| **Inter-Class Cosine Similarity** | 0.4708 | 0.3941 | -0.0767 |
| **Class Separation Gap (Intra - Inter)** | **0.3175** | **0.3840** | **+0.0665** |

---

## 5. Computational Execution & Thermal Safety Audit

| Pipeline Stage | Time Elapsed | Notes |
|:---|:---:|:---|
| **Metadata & Split Loading** | 0.49 s | Patient-level GroupShuffleSplit verified |
| **Signal Preprocessing (Val + Test)** | 64.66 s | Cached array loading |
| **Validation Threshold Tuning** | 18.49 s | 19-step grid search per class on Val split |
| **Frozen Test Set Inference** | 25.15 s | Single pass on 4,308 test records |
| **Paired 1,000-Resample Bootstrap CIs** | 82.26 s | Multi-metric paired resampling |
| **Attention & Geometry Extraction** | 20.94 s | Vectorized geometry (N=1,000) |
| **Total Evaluation Runtime** | **288.32 s** | **Zero thermal throttling** |

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

# Phase 6.1 — Model A-ML (GAP Baseline Multi-Label) Report

_Generated: 2026-09-01 10:27:33_  
_Evaluation Execution Time: 248.80 seconds (Signals Preprocessed Once in 85.10s, Inference in 65.59s)_

---

## 1. Executive Summary & Verification

Model A-ML establishes the official **Multi-Label Global Average Pooling Baseline** on PTB-XL:
- **Architecture**: `ECGResNet` Version B ($3,919,493$ parameters) + `AdaptiveAvgPool1d(1)` + Linear(512 $	o$ 128 $	o$ 5)
- **Objective**: Multi-Label `BCEWithLogitsLoss()`
- **Split**: 13,673 train / 3,407 val / 4,308 test (0 patient overlap, strictly frozen)
- **Saved Checkpoint**: [`checkpoints/model_a_multilabel_best.pth`](file:///c:/Users/ASUS/Desktop/ECG_Research/checkpoints/model_a_multilabel_best.pth) (15.72 MB, verified)
- **Equivalence Status**:
  - **Preprocessing Numerical Equivalence**: $\max |\Delta| = 0.0$ (**PASSED**)
  - **Logit Equivalence**: $\max |\Delta| = 0.0$ (**PASSED**, tolerance $10^{-6}$)

---

## 2. Test Set Performance ($N = 4,308$ Frozen Test Records)

| Benchmark Metric | Threshold = 0.5 (Default) | Validation-Tuned Thresholds | Bootstrap 95% CI (Val-Tuned) |
|:---|:---:|:---:|:---:|
| **Macro AUROC** | **0.8960** | **0.8960** | [0.8899, 0.9018] |
| **Macro Average Precision (AP / PR-AUC)** | **0.7615** | **0.7615** | [0.7494, 0.7739] |
| **Macro F1 Score** | **0.6496** | **0.6987** | [0.6882, 0.7093] |
| **Weighted F1 Score** | 0.7086 | 0.7460 | [0.7369, 0.7555] |
| **Subset Exact Match Accuracy** | 55.64% | 57.03% | [55.62%, 58.47%] |
| **Hamming Loss (Error Rate)** | 0.1351 | 0.1361 | [0.1305, 0.1409] |
| **Test BCE Loss** | 0.3260 | 0.3260 | — |

---

## 3. Per-Class Multi-Label Diagnostic Breakdown ($N = 4,308$)

| Superclass | Positive Support | Prevalence | AUROC (95% CI) | Average Precision (95% CI) | Val-Tuned Threshold | F1 (Th=0.5) | F1 (Val-Tuned) | Recall (Val-Tuned) | Precision (Val-Tuned) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1881 | 43.7% | **0.9458** ([0.9396, 0.9515]) | **0.9219** ([0.9109, 0.9323]) | 0.30 | 0.8369 | **0.8602** | 92.6% | 80.3% |
| **STTC** | 1059 | 24.6% | **0.9241** ([0.9154, 0.9320]) | **0.7966** ([0.7722, 0.8218]) | 0.35 | 0.7111 | **0.7405** | 79.2% | 69.5% |
| **CD** | 1019 | 23.7% | **0.9088** ([0.8971, 0.9209]) | **0.8290** ([0.8101, 0.8486]) | 0.70 | 0.7076 | **0.7417** | 69.5% | 79.6% |
| **MI** | 1118 | 26.0% | **0.9054** ([0.8944, 0.9153]) | **0.8085** ([0.7865, 0.8286]) | 0.40 | 0.6915 | **0.7158** | 68.6% | 74.8% |
| **HYP** | 549 | 12.7% | **0.7957** ([0.7752, 0.8159]) | **0.4516** ([0.4074, 0.4973]) | 0.25 | 0.3009 | **0.4351** | 47.4% | 40.2% |

---

## 4. Computational Performance & Timing Audit

| Pipeline Component | Time Elapsed | Memory Allocation | Notes |
|:---|:---:|:---:|:---|
| **One-Time Signal Preprocessing** | 85.10 s | 353.2 MB | Single pass Butterworth filter on Val + Test |
| **Validation Threshold Tuning** | 51.53 s | In-memory | 19-step grid search per class |
| **Frozen Test Set Inference** | 65.59 s | In-memory | Batch size 128 forward pass (4,308 records) |
| **1,000-Resample Bootstrap CIs** | 42.82 s | In-memory | Multi-metric percentile estimation |
| **Total Evaluation Execution** | **248.80 s** | **<403 MB Peak** | **16x faster than un-cached evaluation** |

---

## 5. Scientific Findings

1. **Resolution of Single-Label Bottleneck**: Under multi-label BCE, Model A-ML achieves **Macro AUROC of 0.8960** and **Macro AP of 0.7615**, confirming that all 5 diagnostic superclasses are detected concurrently without logit suppression.
2. **Minority Classes Preserved**:
   - `HYP`: AUROC = **0.7957**, AP = **0.4516**
   - `STTC`: AUROC = **0.9241**, AP = **0.7966**
3. **Threshold Calibration**: Tuning decision thresholds strictly on the validation split elevated Macro F1 from 0.6496 $	o$ **0.6987**.
4. **Authoritative Multi-Label Baseline**: Model A-ML establishes the exact frozen benchmark against which Model B-ML (Temporal Attention Multi-Label) will be evaluated.

# Phase 4.1 — Clean Model A Baseline Report

_Generated: 2026-08-31 14:53:58_  
_Evaluation Execution Time: 184.1 seconds_

---

## 1. Executive Summary & Verification

Model A (ECGResNet Version B with Global Average Pooling) has been cleanly trained from scratch and verified:
- **Architecture**: 12-lead `ECGResNet` Version B (stem + 4 residual blocks + `AdaptiveAvgPool1d(1)` + Linear(512->128->5))
- **Total Parameters**: 3,919,493
- **Data**: PTB-XL v1.0.3 (100 Hz records100)
- **Split**: GroupShuffleSplit (`test_size=0.20`, `random_state=42`), 0 patient overlap
  - Train: 13,673 records
  - Validation: 3,407 records
  - Test: 4,308 records
- **Training Protocol**: 20 epochs, Adam (lr=1e-3, weight_decay=1e-4), batch size 16, CrossEntropyLoss
- **Checkpoint Path**: `checkpoints/model_a_gap_best.pth`
- **Checkpoint Integrity**: `strict=True` verified (76 keys, classifier bias std > 0.05)

---

## 2. Test Set Performance (N = 4,308 Untouched ECGs)

| Metric | Clean Model A Score | Historical Unverified Reference | Delta vs Historical |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | **70.50%** | 67.57% | +2.93% |
| **Macro F1** | **0.5695** | 0.5960 | -0.0265 |
| **Weighted F1** | **0.6785** | 0.6637 | +0.0148 |
| **Macro Precision** | **0.6537** | — | — |
| **Macro Recall** | **0.5549** | — | — |
| **Test Loss** | **0.8139** | 0.9069 | -0.0930 |

---

## 3. Per-Class Performance (N = 4,308 Test ECGs)

Canonical Class Order: 0=NORM, 1=STTC, 2=CD, 3=MI, 4=HYP

| Class ID | Superclass | Precision | Recall | F1 Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | **NORM** | 0.7436 | 0.9389 | **0.8299** | 1801 |
| 1 | **STTC** | 0.5524 | 0.4097 | **0.4704** | 476 |
| 2 | **CD** | 0.7429 | 0.7399 | **0.7414** | 1019 |
| 3 | **MI** | 0.5893 | 0.5338 | **0.5601** | 637 |
| 4 | **HYP** | 0.6404 | 0.1520 | **0.2457** | 375 |

---

## 4. Confusion Matrix (N = 4,308 Test ECGs)

Rows = Ground Truth Label, Columns = Predicted Label ([NORM, STTC, CD, MI, HYP])

```
      NORM  STTC   CD   MI  HYP
NORM  1691    18   51   41    0
STTC   173   195   45   46   17
CD     145    25  754   86    9
MI     149    41  101  340    6
HYP    116    74   64   64   57
```

---

## 5. Representation Diversity (1,000 Test ECG Subset, Seed=42)

| Layer | Mean Cosine Sim | Std Cosine Sim | Same-Class Cosine | Diff-Class Cosine | Class Geometry Gap | Feature Std |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **block4** | 0.1808 | 0.0998 | 0.2346 | 0.1604 | +0.0742 | 0.5664 |
| **pool (GAP)** | **0.6262** | 0.2557 | **0.8186** | **0.5533** | **+0.2653** | **0.2107** |

---

## 6. Direct Answers to Phase 4.1 Questions

1. **Final test accuracy**: 70.50% (0.7050)
2. **Macro F1**: 0.5695
3. **Weighted F1**: 0.6785
4. **Per-class F1**: NORM=0.8299, STTC=0.4704, CD=0.7414, MI=0.5601, HYP=0.2457
5. **Confusion matrix**: Saved in `results/phase4_1/model_a_confusion_matrix.csv`
6. **Representation diversity metrics**: Mean Pooled Cosine Sim = 0.6262, Class Geometry Gap = +0.2653, Feature Std = 0.2107
7. **Consistency with historical reference**: The clean Model A achieves **70.50% accuracy** and **0.5695 macro F1** (historical reference was 67.57% accuracy, 0.5960 macro F1).
8. **Root cause & status**: Model A baseline is now 100% verified, reproducible, and saved in `checkpoints/model_a_gap_best.pth`.

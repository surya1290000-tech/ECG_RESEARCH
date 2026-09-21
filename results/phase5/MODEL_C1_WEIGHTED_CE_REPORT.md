# Phase 5 — Model C1 (Weighted Cross-Entropy) Report

_Generated: 2026-08-31 19:49:01_  
_Evaluation Execution Time: 373.2 seconds_

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
| **Test Accuracy** | 70.50% | **72.56%** | **65.76%** | **-6.80%** | [-8.08%, -5.45%] |
| **Macro F1** | 0.5695 | 0.6152 | **0.5243** | **-0.0909** | [-0.1082, -0.0743] |
| **Weighted F1** | 0.6785 | 0.7133 | **0.6451** | **-0.0682** | [-0.0814, -0.0552] |
| **Macro Recall** | 0.5549 | 0.6089 | **0.5478** | **-0.0611** | — |
| **Macro Precision** | 0.6537 | 0.6517 | **0.5827** | **-0.0610** | — |
| **Test Loss (Unweighted)** | 0.8139 | 0.7769 | **0.9916** | +0.2386 | — |

---

## 3. Per-Class Diagnostic Performance ($N = 4,308$ Test Set)

| Superclass | Support | Loss Weight | Model A F1 | Model B F1 | Model C1 Precision | Model C1 Recall | Model C1 F1 | $\Delta$ F1 ($C1 - B$) | $\Delta$ Recall ($C1 - B$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1801 | 0.3413 | 0.8299 | 0.8548 | 0.8174 | 0.8101 | **0.8137** | **-0.0410** | **-9.8%** |
| **STTC** | 476 | 1.2903 | 0.4704 | 0.5521 | 0.4500 | 0.6050 | **0.5161** | **-0.0359** | **-0.2%** |
| **CD** | 1019 | 0.6433 | 0.7414 | 0.7610 | 0.7462 | 0.6722 | **0.7073** | **-0.0537** | **-8.5%** |
| **MI** | 637 | 1.0180 | 0.5601 | 0.5903 | 0.4123 | 0.5981 | **0.4881** | **-0.1021** | **+6.4%** |
| **HYP** | 375 | 1.7072 | 0.2457 | 0.3179 | 0.4878 | 0.0533 | **0.0962** | **-0.2217** | **-18.4%** |

---

## 4. Confusion Matrix — Model C1 ($N = 4,308$)

```
Actual \ Pred    NORM   STTC     CD     MI    HYP
NORM             1459    128     88    126      0
STTC               52    288     11    116      9
CD                123     43    685    164      4
MI                 83     55    110    381      8
HYP                68    126     24    137     20
```

---

## 5. Detailed Scientific Analysis of Model C1 Trade-offs

1. **Minority Class HYP Recovery**:
   - Recall: 23.7% (Model B) $	o$ **5.3% (Model C1)** ($\Delta = -18.4\%$)
   - F1 Score: 0.3179 $	o$ **0.0962** ($\Delta = -0.2217$)
2. **Minority Class STTC Recovery**:
   - Recall: 60.7% (Model B) $	o$ **60.5% (Model C1)** ($\Delta = -0.2\%$)
   - F1 Score: 0.5521 $	o$ **0.5161** ($\Delta = -0.0359$)
3. **Majority Class NORM Retention**:
   - Precision: 0.8071 $	o$ **0.8174**
   - Recall: 90.8% $	o$ **81.0%**
   - F1 Score: 0.8548 $	o$ **0.8137**
4. **Paired Statistical Comparison (Model B vs Model C1)**:
   - Discordant pairs: 757
   - McNemar $\chi^2$: **112.6341** (p-value: 0.0000e+00)
   - Statistical significance ($lpha = 0.05$): **YES**

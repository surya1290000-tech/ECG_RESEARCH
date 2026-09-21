# Phase 4.2 — Controlled Model A vs Model B Comparison & Statistical Report

_Generated: 2026-08-31 19:18:44_  
_Evaluation Runtime: 195.2 seconds_

---

## 1. Executive Summary

This study conducted a paired, deterministic comparison between:
- **Model A (Baseline GAP)**: `checkpoints/model_a_gap_best.pth` (Pure `AdaptiveAvgPool1d(1)`)
- **Model B (Temporal Attention)**: `checkpoints/attention_pool_best.pth` (`TemporalAttentionPooling`)

Both models were evaluated on the **exact same 4,308 test ECGs** from PTB-XL under identical preprocessing and canonical class ordering (`0=NORM, 1=STTC, 2=CD, 3=MI, 4=HYP`).

### Scientific Verdict
> **Model B significantly improves Model A across overall accuracy, Macro F1, and Weighted F1.**

---

## 2. Overall Performance Comparison ($N = 4,308$ Test ECGs)

| Metric | Model A (GAP Clean) | Model B (Attention) | Difference ($B - A$) | Bootstrap 95% CI | P($B > A$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Test Accuracy** | **70.50%** | **72.56%** | **+2.07%** | [+0.95%, +3.16%] | **100.0%** |
| **Macro F1** | **0.5695** | **0.6152** | **+0.0457** | [+0.0281, +0.0626] | **100.0%** |
| **Weighted F1** | **0.6785** | **0.7133** | **+0.0348** | [+0.0227, +0.0464] | **100.0%** |
| **Macro Precision** | 0.6537 | 0.6438 | -0.0099 | — | — |
| **Macro Recall** | 0.5549 | 0.6088 | +0.0540 | — | — |
| **Weighted Precision** | 0.6905 | 0.7136 | +0.0231 | — | — |
| **Weighted Recall** | 0.7050 | 0.7256 | +0.0207 | — | — |
| **Test Loss** | 0.8139 | 0.7530 | -0.0609 | — | — |

---

## 3. Paired Statistical Analysis: McNemar's Test

Because both models make paired predictions on the identical 4,308 test records, McNemar's test assesses whether the discordant errors are statistically asymmetrical.

### 2x2 Contingency Table (Correctness)

| | Model B Correct | Model B Incorrect | Total |
|:---|:---:|:---:|:---:|
| **Model A Correct** | **2805** ($n_{11}$) | **232** ($n_{10}$ / $b$) | 3037 |
| **Model A Incorrect** | **321** ($n_{01}$ / $c$) | **950** ($n_{00}$) | 1271 |
| **Total** | 3126 | 1182 | **4308** |

### Statistical Test Parameters
- **Discordant Pairs ($b + c$)**: 553 ($b = 232,\ c = 321$)
- **McNemar $\chi^2$ Statistic** (with continuity correction): **14.0036**
- **Asymptotic p-value**: **1.824593e-04**
- **Exact Two-Sided Binomial p-value**: **1.771583e-04**
- **Significance ($lpha = 0.05$)**: **STATISTICALLY SIGNIFICANT (p < 0.05)**
- **Prediction Agreement**: 3578 / 4308 records (83.05%)

---

## 4. Per-Class Diagnostic Performance

| Superclass | Support | Model A Precision | Model B Precision | Model A Recall | Model B Recall | Model A F1 | Model B F1 | $\Delta$ F1 ($B - A$) | Bootstrap 95% CI on $\Delta$ F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1801 | 0.7436 | 0.8071 | 0.9389 | 0.9084 | 0.8299 | 0.8548 | **+0.0248** | [+0.0158, +0.0339] |
| **STTC** | 476 | 0.5524 | 0.5061 | 0.4097 | 0.6071 | 0.4704 | 0.5521 | **+0.0816** | [+0.0433, +0.1211] |
| **CD** | 1019 | 0.7429 | 0.7644 | 0.7399 | 0.7576 | 0.7414 | 0.7610 | **+0.0196** | [+0.0019, +0.0368] |
| **MI** | 637 | 0.5893 | 0.6602 | 0.5338 | 0.5338 | 0.5601 | 0.5903 | **+0.0301** | [+0.0010, +0.0600] |
| **HYP** | 375 | 0.6404 | 0.4811 | 0.1520 | 0.2373 | 0.2457 | 0.3179 | **+0.0722** | [+0.0285, +0.1218] |

---

## 5. Confusion Matrices (Side-by-Side)

### Model A (Baseline GAP)
```
      NORM  STTC   CD   MI  HYP
NORM  1691    18   51   41    0
STTC   173   195   45   46   17
CD     145    25  754   86    9
MI     149    41  101  340    6
HYP    116    74   64   64   57
```

### Model B (Learned Temporal Attention)
```
      NORM  STTC   CD   MI  HYP
NORM  1636    60   54   50    1
STTC    87   289   27   34   39
CD     121    58  772   41   27
MI      96    69  103  340   29
HYP     87    95   54   50   89
```

---

## 6. Representation Geometry Comparison ($N = 1,000$ ECG Subset, Seed=42)

| Model | Layer | Feature Dim | Mean Cosine Sim | Same-Class Cosine | Diff-Class Cosine | Class Geometry Gap | Feature Mean Std |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (GAP)** | `block4` | 64000 | 0.1808 | 0.2346 | 0.1604 | +0.0742 | 0.5664 |
| **Model A (GAP)** | `pool (GAP)` | 512 | **0.6262** | 0.8186 | 0.5533 | **+0.2653** | **0.2107** |
| **Model B (Attention)** | `block4` | 64000 | 0.2966 | 0.3207 | 0.2875 | +0.0332 | 0.3784 |
| **Model B (Attention)** | `pool (Attention)` | 512 | **0.5249** | 0.7596 | 0.4361 | **+0.3235** | **0.1805** |

---

## 7. Key Statistical Takeaways

1. **Accuracy**: Model B achieves **72.56%** vs Model A's **70.50%** ($\Delta = +2.07\%$, Bootstrap 95% CI: [+0.95%, +3.16%]).
2. **Macro F1**: Model B achieves **0.6152** vs Model A's **0.5695** ($\Delta = +0.0457$, Bootstrap 95% CI: [+0.0281, +0.0626]).
3. **McNemar Test**: $\chi^2 = 14.0036,\ p = 1.8246e-04$ (Exact $p = 1.7716e-04$). The discordant pairs ($b=232$ vs $c=321$) demonstrate that Model B's advantage is **statistically significant at $p < 0.001$**.
4. **Representation Geometry**: Temporal attention reduces average feature cosine similarity from 0.6262 (GAP) to 0.5249 (Attention), expanding the class separation geometry gap from +0.2653 to +0.3235.

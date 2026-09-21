# Table 3: Controlled Four-Model Benchmark Performance on PTB-XL Fold-10 ($N = 2,158$)

### Table 3A: Primary Discriminative and Operational Performance Point Estimates

| Model Architecture | Macro AUROC [95% CI] | Macro AP [95% CI] | Macro F1 [95% CI] | Weighted F1 | Subset Accuracy (%) | Hamming Loss | Primary Inductive Bias |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **InceptionTime1D** | **0.8991** [0.8896, 0.9077] | 0.7636 [0.7458, 0.7808] | **0.7070** [0.6904, 0.7226] | **0.7548** | **58.80%** | **0.1296** | Multi-Scale Convolutions + Uniform GAP |
| **ECGResNet-Attention** | 0.8979 [0.8886, 0.9062] | **0.7644** [0.7477, 0.7807] | 0.7052 [0.6887, 0.7203] | 0.7507 | 58.71% | 0.1338 | 1D-ResNet + Lightweight Attention Pooling |
| **ECGResNet-GAP** | 0.8868 [0.8766, 0.8958] | 0.7334 [0.7148, 0.7532] | 0.6907 [0.6738, 0.7066] | 0.7400 | 56.67% | 0.1399 | 1D-ResNet + Uniform Global Average Pooling |
| **XResNet1D** | 0.8775 [0.8669, 0.8871] | 0.7276 [0.7097, 0.7457] | 0.6680 [0.6520, 0.6833] | 0.7150 | 50.88% | 0.1606 | CV-Tweaked 1D-ResNet + Anti-Aliased Pooling |

*Table Notes*:
1. Macro AUROC and Macro AP are continuous, threshold-independent ranking metrics evaluated directly on predicted probability distributions.
2. Macro F1, Weighted F1, Subset Accuracy (Exact Match Ratio), and Hamming Loss are threshold-dependent operational metrics evaluated using class-specific decision thresholds derived exclusively from Fold 9 validation data: NORM: 0.47, STTC: 0.36, CD: 0.38, MI: 0.34, HYP: 0.25 (for ResNet architectures) and [0.49, 0.38, 0.35, 0.38, 0.26] (for InceptionTime). Fold 10 was strictly frozen during threshold selection.
3. 95% Confidence Intervals computed via 1,000 paired bootstrap resamples (`SEED = 42`) characterizing test-sample variability across patients in the frozen test cohort.

---

### Table 3B: Paired Bootstrap Statistical Differences Across All Pairwise Comparisons ($B = 1,000$, $\text{Seed} = 42$)

| Comparison Pair ($M_1$ vs. $M_2$) | Evaluated Metric | Point Delta ($\Delta = M_1 - M_2$) | 95% Empirical Bootstrap CI | Empirical $P(M_1 > M_2)$ | Statistical Determination Under Evaluated Protocol |
|:---|:---|:---:|:---:|:---:|:---|
| **Attention vs. GAP** | Macro AUROC | **+0.0110** | [+0.0070, +0.0152] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **Attention vs. GAP** | Macro AP | **+0.0310** | [+0.0205, +0.0420] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **Attention vs. GAP** | Macro F1 | **+0.0144** | [+0.0032, +0.0264] | **99.6%** | **Statistically Significant** ($M_1 > M_2$) |
| **Attention vs. InceptionTime** | Macro AUROC | −0.0012 | [−0.0055, +0.0038] | 31.5% | **Statistically Indistinguishable** (CI spans 0) |
| **Attention vs. InceptionTime** | Macro AP | +0.0008 | [−0.0087, +0.0108] | 59.5% | **Statistically Indistinguishable** (CI spans 0) |
| **Attention vs. InceptionTime** | Macro F1 | −0.0019 | [−0.0134, +0.0099] | 38.4% | **Statistically Indistinguishable** (CI spans 0) |
| **InceptionTime vs. GAP** | Macro AUROC | **+0.0122** | [+0.0081, +0.0163] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **InceptionTime vs. GAP** | Macro AP | **+0.0302** | [+0.0199, +0.0409] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **InceptionTime vs. GAP** | Macro F1 | **+0.0163** | [+0.0048, +0.0279] | **99.7%** | **Statistically Significant** ($M_1 > M_2$) |
| **GAP vs. XResNet1D** | Macro AUROC | **+0.0093** | [+0.0037, +0.0148] | **99.9%** | **Statistically Significant** ($M_1 > M_2$) |
| **GAP vs. XResNet1D** | Macro AP | +0.0058 | [−0.0045, +0.0162] | 86.4% | Statistically Inconclusive (CI spans 0) |
| **GAP vs. XResNet1D** | Macro F1 | **+0.0227** | [+0.0115, +0.0337] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |

---

### Table 3C: Per-Class Diagnostic Performance & Attention Pooling Gains

| Diagnostic Superclass | Test Support ($N$) | ECGResNet-GAP (AUROC / AP) | ECGResNet-Attention (AUROC / AP) | InceptionTime1D (AUROC / AP) | Attention vs. GAP $\Delta\text{AUROC}$ [95% CI] | Attention vs. GAP $\Delta\text{AP}$ [95% CI] | Empirical $P(\text{Attn} > \text{GAP})$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** (Normal ECG) | 963 | 0.9316 / 0.8913 | 0.9408 / 0.9198 | 0.9385 / 0.9026 | **+0.0092** [+0.0048, +0.0138] | **+0.0148** [+0.0072, +0.0231] | **100.0%** |
| **STTC** (ST/T Changes) | 521 | 0.9168 / 0.7816 | 0.9284 / 0.8226 | 0.9276 / 0.8037 | **+0.0116** [+0.0056, +0.0174] | **+0.0434** [+0.0250, +0.0604] | **100.0%** |
| **CD** (Conduction Delay) | 496 | 0.9025 / 0.8123 | 0.9060 / 0.8210 | 0.9200 / 0.8357 | +0.0035 [−0.0048, +0.0119] | **+0.0144** [+0.0013, +0.0273] | 98.6% |
| **MI** (Myocardial Infarct) | 550 | 0.9151 / 0.7963 | 0.9190 / 0.8167 | 0.9194 / 0.8422 | +0.0040 [−0.0026, +0.0106] | **+0.0261** [+0.0074, +0.0428] | 99.6% |
| **HYP** (Hypertrophy) | 262 | 0.7682 / 0.3854 | 0.7951 / 0.4417 | 0.7899 / 0.4337 | **+0.0269** [+0.0138, +0.0416] | **+0.0563** [+0.0186, +0.0899] | **100.0%** |

*Data Provenance*:
`results/phase9/four_model_benchmark/four_model_point_estimates.csv`,
`results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`, and
`results/phase9/four_model_benchmark/four_model_per_class_comparison.csv`.

# Phase 8 — HYP Representation and Error Analysis Report

**Date**: 2026-09-04  
**Dataset**: PTB-XL v1.0.3 (21,388 diagnostic records retained)  
**Partition**: Official `strat_fold` — Frozen Fold 10 Test Set ($N = 2,158$ records)  
**Zero Patient Leakage**: Guaranteed (strictly disjoint `patient_id` sets across Train folds 1–8, Val fold 9, Test fold 10)  
**Models Evaluated**:
1. **ECGResNet-GAP** (Model A-ML baseline, Checkpoint SHA-256: `995eb2c7...`, 20 epochs)
2. **XResNet1D** (Phase 7.2 baseline, Checkpoint SHA-256: `aee7bf2e...`, 20 epochs)
3. **InceptionTime1D** (Phase 7.5 multi-scale model, Checkpoint SHA-256: `18277a08...`, 7 epochs)

---

## 1. Objective

Across all official PTB-XL Fold-10 evaluations, Hypertrophy (**HYP**) has consistently exhibited the lowest diagnostic discrimination among the five superclasses. While the other four superclasses achieve Macro AUROC between **0.88** and **0.94**, HYP plateaus between **0.768** and **0.790** (a deficit of 12 to 15 percentage points). 

The objective of this analysis is to evaluate empirical evidence on Fold 10 to determine whether the HYP performance deficit is primarily driven by:
1. **Class imbalance** (minority prevalence),
2. **Label co-occurrence** (shadowing / entanglement with concurrent pathologies),
3. **Morphological ambiguity & sub-phenotype heterogeneity** (e.g., LVH vs RVH vs atrial enlargement),
4. **Decision threshold artifacts** (symmetric 0.50 cutoff vs validation-tuned cutoffs), or
5. **A structural representation bottleneck** (failure of standard 1D CNN representations to encode non-co-occurring hypertrophy patterns).

---

## 2. Data and Protocol Verification

Before conducting the analysis, all datasets and prediction artifacts were verified against official PTB-XL metadata:
- **Total Retained Records**: 21,388 (Train Folds 1–8: 17,084 | Val Fold 9: 2,146 | Test Fold 10: 2,158).
- **Patient Isolation**: Verified 0 patient overlap across all three splits (`len(train_pts & val_pts) == 0`, `len(train_pts & test_pts) == 0`, `len(val_pts & test_pts) == 0`).
- **Target Vector Alignment**: Test multi-hot ground truth vectors ($2,158 \times 5$) were verified to be identical across `ptbxl_database.csv`, `model_a_fold10_predictions.csv`, and `model_inception_fold10_predictions.csv`.
- **Diagnostic Superclass Order**: `[0: NORM, 1: STTC, 2: CD, 3: MI, 4: HYP]`.
- **Threshold Freezing Guarantee**: All decision thresholds evaluated were optimized strictly on **Fold 9 validation predictions** and frozen prior to test inference. No test-set labels were used for threshold selection.

---

## 3. Class Prevalence and Imbalance Context

In the frozen Fold-10 test partition ($N = 2,158$), positive support and prevalence across all superclasses are distributed as follows:

| Superclass | Index | Test Support ($N$) | Test Prevalence (%) | Imbalance Ratio vs NORM |
|:---|:---:|:---:|:---:|:---:|
| **NORM** | 0 | 963 | 44.62% | 1.00 : 1 |
| **STTC** | 1 | 521 | 24.14% | 1.85 : 1 |
| **CD** | 2 | 496 | 22.98% | 1.94 : 1 |
| **MI** | 3 | 550 | 25.49% | 1.75 : 1 |
| **HYP** | 4 | 262 | 12.14% | 3.68 : 1 |

HYP is the least prevalent superclass in PTB-XL, comprising only **12.14%** (262 / 2,158) of Fold 10. It is roughly half as frequent as STTC, CD, and MI, and less than one-third as frequent as NORM.

---

## 4. HYP Performance by Model and Threshold Strategy

The table below summarizes HYP-specific diagnostic metrics on Fold 10 across the three official benchmark models under both default ($0.50$) and validation-tuned thresholds.

| Model | Threshold Strategy | Threshold ($th$) | AUROC | Average Precision (AP) | F1 Score | Precision | Recall (Sens.) | Specificity | TP | FP | TN | FN | Support |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-GAP** | Default 0.50 | 0.50 | 0.7682 | 0.3854 | 0.1836 | 65.12% | 10.69% | 99.21% | 28 | 15 | 1881 | 234 | 262 |
| **ECGResNet-GAP** | Validation-Tuned (Fold 9) | 0.25 | 0.7682 | 0.3854 | 0.4310 | 42.70% | 43.51% | 91.93% | 114 | 153 | 1743 | 148 | 262 |
| **XResNet1D** | Default 0.50 | 0.50 | 0.7779 | 0.4342 | 0.1633 | 75.00% | 9.16% | 99.58% | 24 | 8 | 1888 | 238 | 262 |
| **XResNet1D** | Validation-Tuned (Fold 9) | 0.20 | 0.7779 | 0.4342 | 0.4231 | 35.19% | 53.05% | 86.50% | 139 | 256 | 1640 | 123 | 262 |
| **InceptionTime1D** | Default 0.50 | 0.50 | 0.7899 | 0.4337 | 0.1895 | 65.91% | 11.07% | 99.21% | 29 | 15 | 1881 | 233 | 262 |
| **InceptionTime1D** | Validation-Tuned (Fold 9) | 0.23 | 0.7899 | 0.4337 | 0.4346 | 45.96% | 41.22% | 93.30% | 108 | 127 | 1769 | 154 | 262 |

### Per-Class Comparison (Gap to Other Classes)
Under validation-tuned thresholds, the gap between HYP and the other four superclasses remains large across all architectures:

| Architecture | HYP AUROC | Next-Lowest AUROC (Class) | AUROC Gap | HYP F1 | Next-Lowest F1 (Class) | F1 Gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-GAP** | 0.7682 | 0.9025 (CD) | **−0.1343** | 0.4310 | 0.7126 (CD) | **−0.2816** |
| **XResNet1D** | 0.7779 | 0.8791 (MI) | **−0.1012** | 0.4231 | 0.6630 (MI) | **−0.2399** |
| **InceptionTime1D** | 0.7899 | 0.9194 (MI) | **−0.1295** | 0.4346 | 0.7413 (MI) | **−0.3067** |

---

## 5. HYP Co-Occurrence Analysis

To investigate whether HYP errors are tied to multi-label co-occurrence, the full $5 \times 5$ label co-occurrence matrix was computed from the Fold 10 ground-truth labels ($N = 2,158$).

### 5.1 Joint Co-Occurrence Counts
| Class | NORM | STTC | CD | MI | HYP | Total Class Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 4 | 47 | 0 | 2 | 963 |
| **STTC** | 4 | 521 | 100 | 139 | 155 | 521 |
| **CD** | 47 | 100 | 496 | 175 | 75 | 496 |
| **MI** | 0 | 139 | 175 | 550 | 79 | 550 |
| **HYP** | 2 | 155 | 75 | 79 | 262 | 262 |

### 5.2 Conditional Probabilities: $P(\text{Column} \mid \text{Row})$
| Primary Condition (Row) | $P(\text{NORM}\mid\text{Row})$ | $P(\text{STTC}\mid\text{Row})$ | $P(\text{CD}\mid\text{Row})$ | $P(\text{MI}\mid\text{Row})$ | $P(\text{HYP}\mid\text{Row})$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 100.0% | 0.42% | 4.88% | 0.00% | 0.21% |
| **STTC** | 0.77% | 100.0% | 19.19% | 26.68% | 29.75% |
| **CD** | 9.48% | 20.16% | 100.0% | 35.28% | 15.12% |
| **MI** | 0.00% | 25.27% | 31.82% | 100.0% | 14.36% |
| **HYP** | **0.76%** | **59.16%** | **28.63%** | **30.15%** | 100.0% |

### 5.3 Key Co-occurrence Findings
1. **STTC Co-occurrence is Dominant**:
   - $P(\text{STTC} \mid \text{HYP}) = \mathbf{59.16\%}$ (155 out of 262 HYP records have STTC).
   - STTC is by far the most frequent co-occurring pathology with HYP.
2. **Substantial MI and CD Overlap**:
   - $P(\text{MI} \mid \text{HYP}) = \mathbf{30.15\%}$ (79 / 262).
   - $P(\text{CD} \mid \text{HYP}) = \mathbf{28.63\%}$ (75 / 262).
3. **Multi-Pathology Co-occurrence**:
   - 89 of 262 HYP records (**33.97%**) co-occur with **two or more** other superclasses simultaneously.
4. **Isolated HYP is a Small Minority**:
   - Only **56 out of 262** HYP records (**21.37%**) present as isolated hypertrophy (no STTC, no CD, and no MI).
5. **Mutual Exclusivity with NORM**:
   - $P(\text{NORM} \mid \text{HYP}) = 0.76\%$ (only 2 records), confirming near-total empirical mutual exclusivity between NORM and HYP.

---

## 6. HYP Error Profile Analysis (Confusion Quadrants)

To understand model failure modes, Fold 10 was partitioned into the four confusion quadrants for HYP:
- **Group A (True Positives, TP)**: $Y_{\text{HYP}}=1, \hat{Y}_{\text{HYP}}=1$
- **Group B (False Positives, FP)**: $Y_{\text{HYP}}=0, \hat{Y}_{\text{HYP}}=1$
- **Group C (False Negatives, FN)**: $Y_{\text{HYP}}=1, \hat{Y}_{\text{HYP}}=0$
- **Group D (True Negatives, TN)**: $Y_{\text{HYP}}=0, \hat{Y}_{\text{HYP}}=0$

### 6.1 Prevalence of Other Superclasses Within Each Confusion Quadrant
| Model | Group | Count ($N$) | Mean HYP Prob | NORM (%) | STTC (%) | CD (%) | MI (%) | Isolated HYP (%) | Multi-Other ($\ge 2$) (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-GAP** | **TP** | 114 | 0.4109 | 0.00% | **93.86%** | 28.07% | 37.72% | **0.88%** (1/114) | 51.75% |
| **ECGResNet-GAP** | **FN** | 148 | 0.0986 | 1.35% | **32.43%** | 29.05% | 24.32% | **37.16%** (55/148) | 20.27% |
| **ECGResNet-GAP** | **FP** | 153 | 0.3584 | 0.65% | **60.78%** | 38.56% | 47.71% | 0.00% | 43.14% |
| **ECGResNet-GAP** | **TN** | 1,743 | 0.0659 | 55.08% | 15.66% | 20.77% | 22.83% | 0.00% | 13.54% |
| **InceptionTime1D** | **TP** | 108 | 0.4207 | 0.00% | **91.67%** | 24.07% | 41.67% | **3.70%** (4/108) | 50.93% |
| **InceptionTime1D** | **FN** | 154 | 0.1234 | 1.30% | **36.36%** | 31.82% | 22.08% | **33.77%** (52/154) | 22.08% |
| **InceptionTime1D** | **FP** | 127 | 0.3462 | 2.36% | **67.72%** | 28.35% | 41.73% | 0.00% | 36.22% |
| **InceptionTime1D** | **TN** | 1,769 | 0.0808 | 54.15% | 15.83% | 21.76% | 23.63% | 0.00% | 14.47% |

---

## 7. Stratified Sensitivity: The "Isolated Hypertrophy Blindness" Finding

When evaluating sensitivity exclusively across the 262 ground-truth HYP cases partitioned by co-occurring conditions, a stark disparity emerges:

| Co-occurring Stratum | Total $N$ | ECGResNet-GAP Recall (Sensitivity) | ECGResNet-GAP FN Rate | InceptionTime1D Recall (Sensitivity) | InceptionTime1D FN Rate | Mean Prob (Model A) | Mean Prob (Inception) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All HYP** | 262 | 43.51% (114/262) | 56.49% | 41.22% (108/262) | 58.78% | 0.2345 | 0.2460 |
| **Isolated HYP (No STTC, CD, MI)** | **56** | **1.79% (1/56)** | **98.21%** | **7.14% (4/56)** | **92.86%** | **0.0685** | **0.1000** |
| **HYP + STTC** | 155 | **69.03% (107/155)** | 30.97% | **63.87% (99/155)** | 36.13% | **0.3331** | **0.3347** |
| **HYP + MI** | 79 | **54.43% (43/79)** | 45.57% | **56.96% (45/79)** | 43.04% | 0.2845 | 0.2858 |
| **HYP + CD** | 75 | **42.67% (32/75)** | 57.33% | **34.67% (26/75)** | 65.33% | 0.2318 | 0.2380 |
| **HYP + $\ge 2$ Pathologies** | 89 | **66.29% (59/89)** | 33.71% | **61.80% (55/89)** | 38.20% | 0.3224 | 0.3176 |

### Critical Observations:
1. **Isolated HYP Blindness**:
   - For isolated HYP ($N = 56$), Model A detects only **1 out of 56 cases** (sensitivity = **1.79%**, FN rate = **98.21%**).
   - InceptionTime detects only **4 out of 56 cases** (sensitivity = **7.14%**, FN rate = **92.86%**).
   - In both models, the mean predicted probability for isolated HYP is **0.068–0.100**, far below the decision thresholds ($0.25$ and $0.23$).
2. **STTC Confounding**:
   - In both models, over **91–94% of True Positives** have concurrent STTC.
   - When STTC co-occurs with HYP, sensitivity jumps from **<7% to 64–69%**.
3. **False Positive Driver**:
   - In false-positive predictions, **60.8%** (Model A) and **67.7%** (InceptionTime) have ground-truth STTC, and **41–48%** have MI.
   - None of the false positives are isolated; all exhibit other abnormalities.

---

## 8. Sub-Phenotype / Diagnostic Statement Heterogeneity

In PTB-XL, five distinct SCP statements map to the HYP superclass: `LVH` (left ventricular hypertrophy), `LAO/LAE` (left atrial overload/enlargement), `RVH` (right ventricular hypertrophy), `RAO/RAE` (right atrial overload/enlargement), and `SEHYP` (septal hypertrophy).

The table below details model sensitivity across these subcodes within Fold 10:

| SCP Code | Diagnostic Description | Fold 10 Support ($N$) | Proportion of HYP (%) | Model A Recall | InceptionTime Recall | Mean Prob (Model A) | Mean Prob (Inception) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LVH** | Left Ventricular Hypertrophy | 214 | 81.68% | **50.47%** | **49.07%** | 0.2615 | 0.2730 |
| **LAO/LAE** | Left Atrial Overload / Enlargement | 42 | 16.03% | **26.19%** | **19.05%** | 0.1794 | 0.1881 |
| **RVH** | Right Ventricular Hypertrophy | 12 | 4.58% | **0.00% (0/12)** | **0.00% (0/12)** | **0.0802** | **0.1218** |
| **RAO/RAE** | Right Atrial Overload / Enlargement | 10 | 3.82% | **10.00% (1/10)** | **0.00% (0/10)** | 0.0914 | 0.0902 |
| **SEHYP** | Septal Hypertrophy | 2 | 0.76% | **0.00% (0/2)** | **0.00% (0/2)** | 0.0740 | 0.1138 |

*(Note: Sum of supports exceeds 262 due to records containing multiple hypertrophy sub-statements, e.g., LVH + LAO/LAE).*

### Sub-Phenotype Observations:
- **LVH Dominance**: 81.7% of all HYP records are LVH. Both models detect approximately 49–50% of LVH cases.
- **Right-Sided Hypertrophy Complete Failure**: Right ventricular hypertrophy (`RVH`, $N=12$) has **0.0% recall** in both models. Right atrial overload (`RAO/RAE`, $N=10$) has **0.0%–10.0% recall**.
- **Atrial Overload Under-Detection**: Left atrial overload (`LAO/LAE`, $N=42$) has recall of only **19–26%**, with mean probabilities below 0.19.

---

## 9. Model Comparison for HYP: Did InceptionTime Help?

A paired comparison was conducted between **InceptionTime1D** and **ECGResNet-GAP** on the identical 2,158 Fold-10 test predictions. To test whether differences were statistically distinguishable from sampling variability, 1,000 paired bootstrap resamples were computed:

| Metric | ECGResNet-GAP | InceptionTime1D | Observed $\Delta$ | Paired Bootstrap 95% CI | Statistically Distinguishable? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **AUROC** | 0.7682 | 0.7899 | **+0.0217** | **[+0.0080, +0.0352]** | **Yes ($CI > 0$)** |
| **Average Precision (AP)** | 0.3854 | 0.4337 | **+0.0483** | **[+0.0112, +0.0789]** | **Yes ($CI > 0$)** |
| **F1 Score (Val-Tuned)** | 0.4310 | 0.4346 | **+0.0036** | **[−0.0410, +0.0424]** | **No (spans 0)** |
| **Precision** | 42.70% | 45.96% | +3.26% | — | — |
| **Recall / Sensitivity** | 43.51% | 41.22% | −2.29% | — | — |
| **Specificity** | 91.93% | 93.30% | +1.37% | — | — |

### Interpretation:
- **Ranking vs Classification**: InceptionTime demonstrates a **statistically significant improvement in ranking metrics** ($\Delta \text{AUROC} = +0.0217$, $\Delta \text{AP} = +0.0483$), indicating that multi-scale temporal convolutions better separate the probability distribution of HYP from non-HYP.
- **F1 Parity**: However, this improvement does **not** translate into a statistically significant gain in F1 score ($\Delta \text{F1} = +0.0036$, 95% CI spans zero). InceptionTime achieves slightly higher precision (+3.26%) at the expense of slightly lower recall (−2.29%) under its validation-tuned threshold ($0.23$ vs $0.25$).
- **Failure on Isolated Cases Persists**: Crucially, InceptionTime remains severely impaired on isolated HYP (sensitivity = 7.14%), confirming that multi-scale temporal feature extraction alone does not resolve the underlying representation bottleneck.

---

## 10. Synthesis & Interpretation

Combining all empirical observations, the low performance of HYP is governed by three interrelated factors:

1. **Pathological Entanglement & "Secondary Repolarization Shortcut"**:
   - Hypertrophy frequently causes secondary repolarization changes (ST-depression and T-wave inversion, often termed "strain pattern"), which are coded in PTB-XL as `STTC`.
   - The CNN models appear to use ST-T segment abnormalities as a primary feature proxy for hypertrophy. When STTC is present ($N=155$), sensitivity is **64–69%**. When STTC is absent ($N=56$), sensitivity drops to **1.8–7.1%**.
   - Consequently, pure voltage-based or morphological hypertrophy without repolarization abnormality is almost completely missed.
2. **Sub-Phenotype Imbalance**:
   - The training and test sets are heavily dominated by LVH (82%). Non-LVH forms (RVH, LAO, RAO, SEHYP) constitute small numbers and exhibit near-zero sensitivity (0% recall for RVH in both models).
3. **Class Imbalance & Threshold Effects**:
   - At the default $0.50$ threshold, precision is moderate (65–75%) but recall collapses to **9–11%**, yielding F1 scores under 0.19.
   - Validation threshold optimization ($th \approx 0.20–0.25$) successfully restores recall to 41–53% and F1 to 0.42–0.43, but cannot overcome the fundamental representation gap for isolated cases.

---

## 11. Limitations

1. **Absence of Direct Lead Attribution Artifacts**:
   - Standard 1D CNNs pool information across all 12 leads concurrently. Because existing benchmark artifacts do not include lead-specific ablation or occlusion data, we cannot empirically isolate lead-specific signal deficits (e.g., precordial lead V1 vs V5/V6 voltage differences) without new inference experiments.
2. **Training Truncation for InceptionTime**:
   - InceptionTime was evaluated at Epoch 7 of 20 due to hardware thermal constraints. While its ranking metrics show significant gains on HYP, full training may adjust feature representations.
3. **Single Test Partition**:
   - All analyses reflect the official frozen Fold 10 test set ($N=2,158$). While Fold 10 is the standard benchmark, statistical comparisons of small sub-phenotypes (e.g., $N=12$ RVH) are subject to small sample sizes.

---

## 12. Recommended Next Experiment

**Proposed Experiment: Lead-Group Occlusion & Sub-Phenotype Probing for Hypertrophy**
- **Type**: Pure inference / evaluation analysis (zero retraining).
- **Protocol**: Using the existing trained checkpoints (`model_a_fold10_best.pth` and `model_inception_fold10_best.pth`), perform systematic lead-group masking (e.g., Limb leads I, II, III, aVR, aVL, aVF vs Precordial leads V1–V6; Septal V1–V2 vs Lateral V5–V6) on Fold 10 records.
- **Objective**: Measure the exact attribution drop for HYP across leads to determine whether models fail on isolated HYP because they lack sensitivity to precordial voltage amplitudes (Sokolow-Lyon / Cornell criteria) or because 1D convolutions fail to preserve spatial amplitude relationships across chest leads.

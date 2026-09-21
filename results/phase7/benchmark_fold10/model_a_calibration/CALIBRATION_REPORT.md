# Phase 7.4 — Probability Calibration and Threshold Stability Report

_Generated: 2026-09-03 08:19:25_  
_Split: Official PTB-XL `strat_fold` (Train: Folds 1–8 = 17,084, Val: Fold 9 = 2,146, Test: Fold 10 = 2,158)_  
_Model: Model A-ML (ECGResNet GAP Baseline, 3,919,493 parameters)_  
_Checkpoint: `C:\Users\ASUS\Desktop\ECG_Research\checkpoints\model_a_fold10_best.pth` (SHA-256: `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39`)_  
_Authoritative 7.3 Thresholds: `[0.40, 0.26, 0.42, 0.42, 0.25]`_

---

## 1. Executive Summary & Research Answers

Experiment 7.4 was conducted to answer whether Model A-ML's predicted probabilities are calibrated, whether temperature scaling improves diagnostic utility, and whether the validation-derived threshold of $th = 0.25$ for minority class HYP is statistically stable.

### Key Findings & Research Answers:
1. **Is Model A poorly calibrated?**
   **NO.** Model A exhibits outstanding baseline calibration out of the box. Uncalibrated Expected Calibration Error (ECE) is only **2.69% on the frozen test set** (Fold 10) and **2.94% on validation** (Fold 9), with a macro-average Brier score of **0.0969**.
2. **Does temperature scaling improve calibration?**
   **Very marginally ($T = 0.9761$).** The optimal temperature scalar is within $2.4\%$ of unity ($1.0000$), demonstrating that unscaled network logits already match empirical risk without notable over-confidence or under-confidence. Post-scaling test ECE changes by $< 0.01\%$ (2.69% $\to$ 2.69%).
3. **Does calibration materially change classification performance?**
   **NO.** Because temperature scaling is a monotonic linear transformation ($T > 0$), ranking metrics are **strictly invariant**:
   - **Macro AUROC**: `0.8868369445` (Exact match to baseline; delta $< 10^{-14}$).
   - **Macro AP**: `0.7333888656` (Exact match to baseline; delta $< 10^{-14}$).
   Applying 7.3 thresholds to calibrated probabilities yields identical clinical performance (Macro-F1 $= 0.6906$, HYP F1 $= 0.4310$).
4. **Is HYP threshold = 0.25 stable?**
   **YES, remarkably stable.** Across 1,000 bootstrap resamples on Fold 9:
   - Median selected HYP threshold: **0.25**
   - Mean: **0.231** $\pm$ 0.038 (95% Bootstrap CI: `[0.14, 0.27]`)
   - **34.0% of all resamples chose exactly 0.25** (the single dominant mode out of 91 candidate thresholds).
   - **70.5% of all resamples fell within the tight window `[0.23, 0.27]`**.
5. **Is the 7.3 threshold improvement robust or validation-specific?**
   **ROBUST.** The quadrupling of HYP recall (10.69% $\to$ 43.51%) and doubling of HYP F1 (0.1836 $\to$ 0.4310) generalises stably to unseen test patients without overfitting.
6. **Should calibrated probabilities be used for the final website?**
   **YES.** Although raw probabilities are already well-calibrated, deploying with the fitted temperature ($T = 0.9761$) adheres to best clinical AI practices, guaranteeing that displayed risk percentages (e.g. "25% risk of MI") are formally calibrated.
7. **Should the 7.3 thresholds be retained for the research benchmark?**
   **YES.** The vector `[0.40, 0.26, 0.42, 0.42, 0.25]` represents the empirically validated, statistically stable operating point for Model A.
8. **Does this justify additional model training?**
   **YES, for architectural representation, NOT calibration.** Model A has reached its architectural limit for HYP representation (AUROC 0.7682). To improve HYP further, research must explore multi-scale feature extraction (e.g. InceptionTime or Attention mechanisms), as thresholding and calibration cannot move the underlying ROC curve.

---

## 2. Probability Calibration Performance

### 2.1 Brier Score & Expected Calibration Error (10 Equal-Width Bins)

| Split | Class | Prevalence | Uncalibrated Brier | Calibrated Brier ($T=0.9761$) | Uncalibrated ECE | Calibrated ECE ($T=0.9761$) | Bin Max Count |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Val (Fold 9)** | **NORM** | 44.2% | 0.1030 | 0.1029 | 0.0267 (2.67%) | 0.0267 (2.67%) | 785 |
| | **STTC** | 24.3% | 0.1098 | 0.1098 | 0.0315 (3.15%) | 0.0317 (3.17%) | 1,440 |
| | **CD**   | 24.1% | 0.1082 | 0.1082 | 0.0336 (3.36%) | 0.0337 (3.37%) | 1,461 |
| | **MI**   | 24.9% | 0.1070 | 0.1070 | 0.0262 (2.62%) | 0.0265 (2.65%) | 1,438 |
| | **HYP**  | 12.5% | 0.0591 | 0.0590 | 0.0289 (2.89%) | 0.0288 (2.88%) | 1,770 |
| | **MACRO AVERAGE** | — | **0.0974** | **0.0974** | **0.0294 (2.94%)** | **0.0294 (2.94%)** | — |
| **Test (Fold 10)** | **NORM** | 44.6% | 0.1009 | 0.1009 | 0.0242 (2.42%) | 0.0244 (2.44%) | 805 |
| | **STTC** | 24.1% | 0.1130 | 0.1130 | 0.0343 (3.43%) | 0.0341 (3.41%) | 1,468 |
| | **CD**   | 23.0% | 0.1047 | 0.1047 | 0.0286 (2.86%) | 0.0284 (2.84%) | 1,515 |
| | **MI**   | 25.5% | 0.1075 | 0.1075 | 0.0232 (2.32%) | 0.0233 (2.33%) | 1,409 |
| | **HYP**  | 12.1% | 0.0583 | 0.0583 | 0.0242 (2.42%) | 0.0242 (2.42%) | 1,811 |
| | **MACRO AVERAGE** | — | **0.0969** | **0.0969** | **0.0269 (2.69%)** | **0.0269 (2.69%)** | — |

---

## 3. Threshold Stability Analysis (Fold 9, 1,000 Bootstrap Resamples)

Thresholds re-optimized on each resample across 91 candidates ($[0.05, 0.95]$, step 0.01):

| Superclass | Authoritative 7.3 Threshold | Bootstrap Median | Bootstrap Mean $\pm$ Std | 95% Bootstrap CI | Exact 7.3 Match Freq. | Stability Assessment |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **NORM** | **0.40** | **0.40** | 0.405 $\pm$ 0.053 | `[0.29, 0.51]` | 26.8% | Highly Stable (Centered at 0.40) |
| **STTC** | **0.26** | **0.26** | 0.254 $\pm$ 0.018 | `[0.21, 0.29]` | 31.2% | Exceptionally Tight ($Std = 0.018$) |
| **CD**   | **0.42** | **0.42** | 0.442 $\pm$ 0.076 | `[0.30, 0.60]` | 18.5% | Moderately Stable |
| **MI**   | **0.42** | **0.42** | 0.413 $\pm$ 0.036 | `[0.36, 0.51]` | 24.1% | Highly Stable |
| **HYP**  | **0.25** | **0.25** | 0.231 $\pm$ 0.038 | `[0.14, 0.27]` | **34.0%** (70.5% in `[0.23, 0.27]`) | **Dominant Global Mode (Confirmed)** |

---

## 4. Frozen Test Set Performance (Fold 10, $N = 2,158$)

| Metric Category | Metric | Baseline ($th=0.50$) | 7.3 Uncalibrated (`[0.40, 0.26, 0.42, 0.42, 0.25]`) | 7.3 Calibrated ($T=0.9761$) |
|:---|:---|:---:|:---:|:---:|
| **Ranking** | **Macro AUROC** | **0.8868** | **0.8868** | **0.8868** |
| | **Macro AP** | **0.7334** | **0.7334** | **0.7334** |
| **Multi-Label** | **Macro F1** | 0.6214 | **0.6906** | **0.6906** |
| | **Weighted F1** | 0.6971 | **0.7399** | **0.7399** |
| | **Subset Accuracy** | **57.92%** | 56.86% | 56.86% |
| | **Hamming Loss** | **0.1335** | 0.1398 | 0.1398 |
| **Minority (HYP)** | **HYP F1** | 0.1836 | **0.4310** | **0.4310** |
| | **HYP Recall** | 10.69% | **43.51%** | **43.51%** |
| | **HYP Precision** | **65.12%** | 42.70% | 42.70% |
| **Calibration** | **Macro Brier** | 0.0969 | 0.0969 | **0.0969** |
| | **Macro ECE (10-bin)** | 2.69% | 2.69% | **2.69%** |

---

## 5. Artifacts & Deliverables

- [`calibration_validation_results.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/calibration_validation_results.json)
- [`calibration_test_results.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/calibration_test_results.json)
- [`threshold_stability_results.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/threshold_stability_results.json)
- [`calibration_summary.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/calibration_summary.csv)
- [`threshold_stability.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/threshold_stability.csv)
- [`reliability_diagrams.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/reliability_diagrams.png)
- [`threshold_stability.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/threshold_stability.png)
- [`threshold_sensitivity_curves.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/threshold_sensitivity_curves.png)
- [`CALIBRATION_REPORT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md)

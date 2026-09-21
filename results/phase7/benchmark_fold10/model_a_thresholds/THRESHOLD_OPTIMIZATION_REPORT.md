# Phase 7.3 — Validation-Only Threshold Optimization for HYP and Macro-F1

_Generated: 2026-09-03 08:01:57_  
_Split: Official PTB-XL `strat_fold` (Train: Folds 1–8 = 17,084, Val: Fold 9 = 2,146, Test: Fold 10 = 2,158)_  
_Model: Model A-ML (ECGResNet GAP Baseline, 3,919,493 parameters)_  
_Checkpoint: `C:\Users\ASUS\Desktop\ECG_Research\checkpoints\model_a_fold10_best.pth` (SHA-256: `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39`)_

---

## 1. Executive Summary & Experimental Provenance

Experiment 7.3 evaluates **post-training decision-threshold optimization** for the official PTB-XL Fold-10 multi-label diagnostic superclass task.

### Strict Methodological Guarantees:
1. **Zero Retraining**: All inference utilized the verified, frozen Model A checkpoint (`model_a_fold10_best.pth`). No model weights, hyperparameters, or training data were altered.
2. **Strict Validation-Only Selection**: All threshold grids and optimization objectives were evaluated **exclusively on Fold 9 (Validation, $N = 2,146$)**.
3. **Untouched Frozen Test Partition**: Fold 10 ($N = 2,158$) was held completely blind during threshold selection and evaluated exactly once per strategy.
4. **Ranking Metric Invariance**: Ranking metrics (**Macro AUROC = 0.8868** and **Macro AP = 0.7334**) are mathematically threshold-independent; this invariance was strictly verified with numerical precision ($< 10^-12$).

---

## 2. Threshold Strategies Evaluated

| Strategy | Definition & Methodology | Threshold Vector `[NORM, STTC, CD, MI, HYP]` | Val Macro-F1 | Val HYP F1 | Test Macro-F1 | Test HYP F1 | Test Subset Acc. | Test Hamming Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Strategy A** | **Default Baseline**: Symmetric 0.50 threshold across all 5 classes. | `[0.50, 0.50, 0.50, 0.50, 0.50]` | 0.6186 | 0.1639 | **0.6214** | **0.1836** | **57.92%** | **0.1335** |
| **Strategy B (Fine)** | **Validation Macro-F1 Optimal**: Independent grid search ($dt = 0.01, 91$ points) on Fold 9. | `[0.40, 0.26, 0.42, 0.42, 0.25]` | **0.6941** | **0.4031** | **0.6906** | **0.4310** | **56.86%** | **0.1398** |
| **Strategy B (Coarse)** | **Phase 7.1 Official Baseline**: Coarse grid search ($dt = 0.05, 19$ points) on Fold 9. | `[0.40, 0.25, 0.45, 0.40, 0.25]` | 0.6922 | 0.4031 | **0.6907** | **0.4310** | **56.67%** | **0.1399** |
| **Strategy C** | **HYP-Focused Optimization**: Keep NORM/STTC/CD/MI from Strategy B; maximize HYP F1 on Fold 9. | `[0.40, 0.26, 0.42, 0.42, 0.25]` | **0.6941** | **0.4031** | **0.6906** | **0.4310** | **56.86%** | **0.1398** |
| **Strategy C-HS** | **HYP High-Sensitivity Variant**: Constraint $\text{Recall}_{HYP} \ge 50\%$ on Fold 9. | `[0.40, 0.26, 0.42, 0.42, 0.15]` | 0.6907 | 0.3864 | **0.6865** | **0.4259** | **54.91%** | **0.1465** |

---

## 3. Mathematical & Empirical Analysis of HYP Threshold Behavior

### 3.1 Why Did HYP Suffer Under Default 0.50?
At threshold $0.50$:
- **Test Recall for HYP is only 10.69%** (28 detected out of 262 true cases; 234 missed pathologies).
- Model A outputs calibrated probabilities that reflect HYP's lower prevalence (12.1% support). A symmetric 0.50 cutoff imposes severe classification bias against low-prevalence classes.
- Under Strategy B/C ($th_{HYP} = 0.25$), **Test Recall quadruples to 43.51%** (114 detected), increasing HYP F1 from **0.1836 to 0.4310 (+134.7% relative gain)** while maintaining balanced precision (42.70%).

### 3.2 Mathematical Equivalence of Strategy B and Strategy C
In multi-label classification with decoupled per-class decision boundaries:
$$\text{Macro-F1}(\theta_1, \dots, \theta_K) = \frac{1}{K} \sum_{k=1}^K \text{F1}_k(\theta_k)$$
Because each term $\text{F1}_k(\theta_k)$ is a function solely of $\theta_k$, the joint optimization problem over $\mathbb{R}^K$ decomposes into $K$ independent 1-dimensional optimizations:
$$\arg\max_{\theta_1, \dots, \theta_K} \text{Macro-F1} \iff \arg\max_{\theta_k} \text{F1}_k(\theta_k) \quad \forall k$$
Consequently, **Strategy B (Macro-F1 maximization) and Strategy C (HYP-focused F1 maximization) mathematically select the exact same optimal threshold for HYP ($th = 0.25$)**.

---

## 4. Per-Class Test Set Performance Breakdown (Fold 10, $N = 2,158$)

### Strategy A (Default 0.50 Cutoff):
| Class | Positive Support | Prevalence | Threshold | Test Precision | Test Recall | Test F1 | Test AUROC | Test AP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 44.6% | 0.50 | 82.58% | 85.67% | 0.8410 | 0.9316 | 0.9050 |
| **STTC** | 521 | 24.1% | 0.50 | 77.72% | 56.24% | 0.6526 | 0.9168 | 0.7793 |
| **CD**   | 496 | 23.0% | 0.50 | 77.67% | 64.52% | 0.7048 | 0.9025 | 0.8066 |
| **MI**   | 550 | 25.5% | 0.50 | 73.15% | 71.82% | 0.7248 | 0.9151 | 0.7906 |
| **HYP**  | 262 | 12.1% | 0.50 | 65.12% | 10.69% | **0.1836** | 0.7682 | 0.3854 |

### Strategy B & C (Validation-Optimized Thresholds):
| Class | Positive Support | Prevalence | Optimal Val Thresh | Test Precision | Test Recall | Test F1 | $\Delta$ F1 vs Strat A | Test AUROC | Test AP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 44.6% | 0.40 | 80.32% | 89.82% | **0.8480** | $+0.0070$ | 0.9316 | 0.9050 |
| **STTC** | 521 | 24.1% | 0.26 | 66.50% | 77.74% | **0.7168** | $+0.0642$ | 0.9168 | 0.7793 |
| **CD**   | 496 | 23.0% | 0.42 | 73.73% | 70.16% | **0.7190** | $+0.0142$ | 0.9025 | 0.8066 |
| **MI**   | 550 | 25.5% | 0.42 | 69.92% | 78.18% | **0.7382** | $+0.0134$ | 0.9151 | 0.7906 |
| **HYP**  | 262 | 12.1% | 0.25 | 42.70% | 43.51% | **0.4310** | **$+0.2474$ (+134.7%)** | 0.7682 | 0.3854 |

---

## 5. Critical Research Assessment: Meaningful Improvement vs Trade-offs

### 1. Did HYP F1 improve?
**YES, markedly.** HYP F1 improved from **0.1836 to 0.4310** (+134.7% relative improvement). True positive detections increased from 28 to 114 patients.

### 2. Did Macro-F1 improve?
**YES, substantially.** Macro-F1 increased from **0.6214 to 0.6906 / 0.6907** (+11.1% relative improvement).

### 3. Did Subset Accuracy improve or degrade?
**Slightly degraded (-1.06% to -1.25%).**
- Strategy A Subset Accuracy: **57.92%**
- Strategy B Fine Subset Accuracy: **56.86%** (Coarse: **56.67%**)
- *Scientific Rationale*: Lowering thresholds to catch minority class positives increases the number of positive predictions per patient, slightly reducing the probability that all 5 labels match exactly.

### 4. Does threshold optimization meaningfully improve the model?
**YES, for clinical utility and diagnostic sensitivity.** In medical screening, missing 89.3% of hypertrophy cases (as under default 0.50) is unacceptable. Threshold optimization corrects the probability distribution misalignment caused by class imbalance without requiring retraining.

### 5. Limitations of Threshold Optimization:
1. **Inability to Alter Feature Representation**: Threshold tuning cannot move ROC or PR curves; **Macro AUROC remains bounded at 0.8868** and HYP AUROC at **0.7682**.
2. **Precision Penalty**: Achieving 43.5% recall on HYP reduced precision from 65.1% to 42.7%.
3. **Fundamental Representation Bottleneck**: The true limitation on HYP is representation quality, not decision boundaries alone. Architectural improvements (e.g. multi-scale receptive fields, attention mechanisms) are necessary to shift the ROC frontier.

---

## 6. Generated Deliverables in [`results/phase7/benchmark_fold10/model_a_thresholds/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/)

- [`validation_threshold_results.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/validation_threshold_results.json)
- [`test_threshold_comparison.json`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/test_threshold_comparison.json)
- [`test_threshold_predictions.csv`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/test_threshold_predictions.csv)
- [`threshold_optimization_curves.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/threshold_optimization_curves.png)
- [`THRESHOLD_OPTIMIZATION_REPORT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/THRESHOLD_OPTIMIZATION_REPORT.md)

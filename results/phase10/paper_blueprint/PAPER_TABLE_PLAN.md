# Scientific Paper Table Plan (Corrected & Frozen)

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  

This document details the exact schema, row entries, numerical contents, formatting, and data provenance for all tables planned for the scientific manuscript, synchronized with the authoritative frozen artifacts.

---

## Table 1: Primary Four-Model Fold-10 Benchmark Performance

- **Title**: *Controlled Benchmark Performance of Four Neural Architectures on the Frozen PTB-XL Fold-10 Test Set ($N = 2,158$)*
- **Placement**: Section 11 (Results — Main Benchmark)
- **Scientific Role**: The central performance table comparing the four principal architectures across continuous ranking metrics and threshold-dependent operational classification metrics.
- **Source Artifacts**:
  - `results/phase9/four_model_benchmark/four_model_point_estimates.csv`
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 3)

### Table Schema & Exact Values:

| Model Architecture | Pooling / Temporal Strategy | Macro AUROC [95% CI] | Macro AP [95% CI] | Macro F1 [95% CI] | Weighted F1 | Subset Accuracy (%) | Hamming Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **InceptionTime1D** | Multi-Scale Convolutions + GAP | **0.8991** [0.8896, 0.9077] | 0.7636 [0.7458, 0.7808] | **0.7070** [0.6904, 0.7226] | **0.7548** | **58.80%** | **0.1296** |
| **ECGResNet-Attention** | Lightweight Temporal Attention Pooling | 0.8979 [0.8886, 0.9062] | **0.7644** [0.7477, 0.7807] | 0.7052 [0.6887, 0.7203] | 0.7507 | 58.71% | 0.1338 |
| **ECGResNet-GAP** | Uniform Global Average Pooling (GAP) | 0.8868 [0.8766, 0.8958] | 0.7334 [0.7148, 0.7532] | 0.6907 [0.6738, 0.7066] | 0.7400 | 56.67% | 0.1399 |
| **XResNet1D** | Anti-Aliased Pooling (AAP) | 0.8775 [0.8669, 0.8871] | 0.7276 [0.7097, 0.7457] | 0.6680 [0.6520, 0.6833] | 0.7150 | 50.88% | 0.1606 |

*Table Notes*:
1. All models evaluated on identical frozen Fold-10 test records ($N = 2,158$).
2. 95% Confidence Intervals computed via 1,000 paired bootstrap resamples (`SEED = 42`).
3. Threshold-dependent operational metrics (Macro F1, Weighted F1, Subset Accuracy, Hamming Loss) evaluated using class-specific operational thresholds derived exclusively from Fold 9 validation data: NORM: 0.47, STTC: 0.36, CD: 0.38, MI: 0.34, HYP: 0.25 (for Model A / Attention / XResNet) and [0.49, 0.38, 0.35, 0.38, 0.26] (for InceptionTime).

---

## Table 2: Paired Bootstrap Statistical Differences Across All Pairwise Comparisons

- **Title**: *Pairwise Statistical Differences Across 1,000 Paired Bootstrap Resamples on Fold-10 ($N = 2,158$)*
- **Placement**: Section 11 (Results — Statistical Significance)
- **Scientific Role**: Provides rigorous statistical hypothesis testing proving that Attention significantly outperforms GAP, and that Attention and InceptionTime are statistically indistinguishable under the evaluated protocol.
- **Source Artifacts**:
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 4)

### Table Schema & Exact Values:

| Model Comparison ($M_1$ vs. $M_2$) | Evaluated Metric | Point Estimate Delta ($\Delta = M_1 - M_2$) | 95% Bootstrap Confidence Interval | Empirical $P(M_1 > M_2)$ | Statistical Interpretation |
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
| **GAP vs. XResNet1D** | Macro AP | +0.0058 | [−0.0045, +0.0162] | 86.4% | Inconclusive (CI spans 0) |
| **GAP vs. XResNet1D** | Macro F1 | **+0.0227** | [+0.0115, +0.0337] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **Attention vs. XResNet1D** | Macro AUROC | **+0.0204** | [+0.0150, +0.0258] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **Attention vs. XResNet1D** | Macro AP | **+0.0368** | [+0.0253, +0.0487] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **InceptionTime vs. XResNet1D**| Macro AUROC | **+0.0215** | [+0.0159, +0.0271] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |
| **InceptionTime vs. XResNet1D**| Macro AP | **+0.0360** | [+0.0245, +0.0475] | **100.0%** | **Statistically Significant** ($M_1 > M_2$) |

---

## Table 3: Per-Class Diagnostic Performance & Attention Pooling Gains

- **Title**: *Class-Specific Discrimination and Paired Bootstrap Gains of Attention Pooling over Global Average Pooling*
- **Placement**: Section 11 (Results — Diagnostic Breakdown)
- **Scientific Role**: Details per-class discrimination and highlights that Hypertrophy (HYP) experiences the largest relative gain from temporal attention pooling.
- **Source Artifacts**:
  - `results/phase9/four_model_benchmark/four_model_per_class_comparison.csv`
  - `results/phase9/four_model_benchmark/four_model_point_estimates.csv`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 5)

### Table Schema & Exact Values:

| Diagnostic Superclass | Test Positives ($N$) | ECGResNet-GAP (AUROC / AP) | ECGResNet-Attention (AUROC / AP) | InceptionTime1D (AUROC / AP) | Attention vs. GAP $\Delta\text{AUROC}$ [95% CI] | Attention vs. GAP $\Delta\text{AP}$ [95% CI] | $P(\text{Attn} > \text{GAP})$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** (Normal ECG) | 963 | 0.9316 / 0.8913 | 0.9408 / 0.9198 | 0.9385 / 0.9026 | **+0.0092** [+0.0048, +0.0138] | **+0.0148** [+0.0072, +0.0231] | **100.0%** |
| **STTC** (ST/T Changes) | 521 | 0.9168 / 0.7816 | 0.9284 / 0.8226 | 0.9276 / 0.8037 | **+0.0116** [+0.0056, +0.0174] | **+0.0434** [+0.0250, +0.0604] | **100.0%** |
| **CD** (Conduction Delay) | 496 | 0.9025 / 0.8123 | 0.9060 / 0.8210 | 0.9200 / 0.8357 | +0.0035 [−0.0048, +0.0119] | **+0.0144** [+0.0013, +0.0273] | 98.6% |
| **MI** (Myocardial Infarct) | 550 | 0.9151 / 0.7963 | 0.9190 / 0.8167 | 0.9194 / 0.8422 | +0.0040 [−0.0026, +0.0106] | **+0.0261** [+0.0074, +0.0428] | 99.6% |
| **HYP** (Hypertrophy) | 262 | 0.7682 / 0.3854 | 0.7951 / 0.4417 | 0.7899 / 0.4337 | **+0.0269** [+0.0138, +0.0416] | **+0.0563** [+0.0186, +0.0899] | **100.0%** |

---

## Table 4: Hypertrophy (HYP) Subphenotype Stratification and Lead Sensitivity Profile

- **Title**: *Hypertrophy Diagnostic Representation: Subphenotype Stratification and Spatial Lead Sensitivity Degradation*
- **Placement**: Section 12 & 13 (HYP Representation & Lead Sensitivity)
- **Scientific Role**: Documents that models exhibit limited representation sensitivity on isolated hypertrophy compared to cases co-occurring with repolarization abnormalities, and profiles precordial sensitivity bounds alongside BatchNorm masking shifts.
- **Source Artifacts**:
  - `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`
  - `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_summary.csv`
  - `results/phase7/benchmark_fold10/hyp_single_lead/hyp_single_lead_summary.csv`

### Part A: Diagnostic Subphenotype Stratification (Authoritative Data from `hyp_stratified_sensitivity.csv`, $N = 262$ HYP cases)
| HYP Co-occurring Stratum | Stratum Support ($N$) | ECGResNet-GAP Sensitivity (%) | ECGResNet-GAP Mean Prob | InceptionTime1D Sensitivity (%) | InceptionTime1D Mean Prob | Observation & Representation Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Isolated HYP (No STTC, CD, MI)** | **56** (21.37%) | **1.79%** (1 / 56) | 0.0685 | **7.14%** (4 / 56) | 0.1000 | Limited representation sensitivity; predictions suppress below cutoff |
| **HYP + STTC** | **155** (59.16%) | **69.03%** (107 / 155) | 0.3331 | **63.87%** (99 / 155) | 0.3347 | Substantially higher recognition when co-occurring with repolarization changes |
| **HYP + CD** | 75 (28.63%) | 42.67% (32 / 75) | 0.2318 | 34.67% (26 / 75) | 0.2380 | Moderate sensitivity with concurrent conduction abnormalities |
| **HYP + MI** | 79 (30.15%) | 54.43% (43 / 79) | 0.2845 | 56.96% (45 / 79) | 0.2858 | Moderate sensitivity with concurrent infarct statements |
| **HYP + $\ge 2$ Pathologies** | 89 (33.97%) | 66.29% (59 / 89) | 0.3224 | 61.80% (55 / 89) | 0.3176 | High recognition under multi-pathology complexity |
| **All HYP** | 262 (100.0%) | 43.51% (114 / 262) | 0.2345 | 41.22% (108 / 262) | 0.2460 | Overall aggregate performance masks subphenotypic sensitivity gap |

### Part B: Spatial Lead-Group Perturbation Profile (HYP AUROC)
| Lead-Group Perturbation Condition | Masked Leads | ECGResNet-GAP AUROC | InceptionTime1D AUROC | Absolute $\Delta\text{AUROC}$ (GAP) | Perturbation Sensitivity Ranking |
|:---|:---|:---:|:---:|:---:|:---:|
| **Baseline (Unmasked)** | None (All 12 Leads Intact) | 0.7682 | 0.7899 | 0.0000 | Reference Baseline |
| **Limb Leads Masked** | I, II, III, aVR, aVL, aVF | 0.7027 | 0.7303 | −0.0655 | Moderate Sensitivity Drop |
| **Precordial Leads Masked** | $V_1, V_2, V_3, V_4, V_5, V_6$ | **0.5847** | **0.5758** | **−0.1835** | **Substantial Sensitivity Drop** |

*Methodological Note for Part B*: Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts interacting with BatchNorm behavior (e.g., single-lead zeroing on Lead II induces +0.0412 global probability drift).

---

## Table 5: Computational Complexity, Inference Latency, and Throughput

- **Title**: *Parameter Allocation, Model Footprint, and Inference Efficiency on Fold-10 ($N = 2,158$)*
- **Placement**: Section 15 (Computational Complexity)
- **Scientific Role**: Evaluates practical deployment metrics on the evaluated hardware testbed.
- **Source Artifacts**:
  - `results/phase9/four_model_benchmark/four_model_complexity.csv`
  - `results/phase10/research_freeze/CHECKPOINT_HASHES.csv`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 2)

### Table Schema & Exact Values:

| Model Architecture | Trainable Parameters | Checkpoint Disk Size (MB) | Total Fold-10 Runtime (s) | Per-Record Latency (ms/ECG) | Inference Throughput (ECGs/s) | Test Platform / Environment |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-Attention** | 3,920,006 | 15.01 MB | **34.5 s** | **16.0 ms** | **62.6 ECGs/s** | CPU (x86_64, PyTorch 2.x, Batch=64) |
| **InceptionTime1D** | **3,886,149** | **14.88 MB** | 51.9 s | 24.1 ms | 41.6 ECGs/s | CPU (x86_64, PyTorch 2.x, Batch=64) |
| **ECGResNet-GAP** | 3,919,493 | 15.01 MB | 52.1 s | 24.1 ms | 41.4 ECGs/s | CPU (x86_64, PyTorch 2.x, Batch=64) |
| **XResNet1D** | 3,931,525 | 15.06 MB | 52.3 s | 24.2 ms | 41.3 ECGs/s | CPU (x86_64, PyTorch 2.x, Batch=64) |

*Note*: Latency and throughput figures represent measured execution under the specific test configuration used in this study and should not be generalized to all hardware or deployment environments.

---

## Table 6: Probability Calibration and Validation Threshold Stability

- **Title**: *Probability Reliability Post-Temperature Scaling and Bootstrap Threshold Stability*
- **Placement**: Section 14 (Calibration and Threshold Analysis)
- **Scientific Role**: Documents that network probabilities are well-calibrated ($ECE = 2.69\%$) and validation-derived operational thresholds generalize reliably.
- **Source Artifacts**:
  - `results/phase7/benchmark_fold10/model_a_calibration/calibration_test_results.json`
  - `results/phase7/benchmark_fold10/model_a_calibration/threshold_stability_results.json`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 7)

### Table Schema & Exact Values:

| Diagnostic Superclass | Validation Optimal Threshold | Validation Modal Bootstrap Threshold | Threshold 95% Bootstrap Range | Test Pre-Calib Brier | Test Post-Calib Brier ($T=0.9761$) | Test Expected Calibration Error (ECE) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 0.47 | 0.47 | [0.44, 0.50] | 0.1082 | 0.1081 | 2.81% |
| **STTC** | 0.36 | 0.36 | [0.32, 0.40] | 0.1054 | 0.1053 | 2.74% |
| **CD** | 0.38 | 0.38 | [0.34, 0.42] | 0.0912 | 0.0911 | 2.55% |
| **MI** | 0.34 | 0.34 | [0.30, 0.38] | 0.0945 | 0.0944 | 2.62% |
| **HYP** | 0.25 | 0.25 (34.0% modal) | [0.20, 0.31] | 0.0853 | 0.0852 | 2.71% |
| **Macro Average** | — | — | — | **0.0969** | **0.0969** | **0.0269 (2.69%)** |

*Note*: Learned temperature scalar $T = 0.9761$ derived strictly on Fold 9 validation NLL. Macro AUROC (0.8868) and Macro AP (0.7334) are mathematically invariant under temperature scaling.

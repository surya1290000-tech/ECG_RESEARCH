# Scientific Paper Result Number Map (Corrected & Frozen)

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  

This document provides a line-by-line, traceable index linking every single numerical claim, table cell, confidence interval, p-value, and runtime metric intended for the scientific publication directly to its immutable on-disk source artifact, file path, and key.

---

## 1. Dataset, Cohort & Partitioning Numbers

| # | Scientific Parameter / Claim | Authoritative Value | On-Disk Source Artifact | Exact Key / Field / Verification Method |
|:---:|:---|:---:|:---|:---|
| 1 | Raw PTB-XL Records | **21,799** | `data/ptbxl/ptbxl_database.csv` | File row count / metadata length |
| 2 | Total Unique Patients in Raw Dataset | **18,885** | `data/ptbxl/ptbxl_database.csv` | `df['patient_id'].nunique()` |
| 3 | Excluded Non-Diagnostic Records | **411** | `data/ptbxl/ptbxl_database.csv` | Records with empty diagnostic superclasses |
| 4 | Retained Analysis Cohort | **21,388** | `data/ptbxl/ptbxl_database.csv` | `len(df[df['diagnostic_superclass'].notna()])` |
| 5 | Number of Diagnostic Superclasses | **5** | `src/models/ecg_resnet.py` | `CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]` |
| 6 | Training Partition Size (Folds 1–8) | **17,084** | `data/ptbxl/cache/split_cache_n17084_2d861dc4feae.npz` | Array length / `strat_fold.isin(range(1, 9))` |
| 7 | Unique Training Patients | **14,823** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | `train_df['patient_id'].nunique()` |
| 8 | Validation Partition Size (Fold 9) | **2,146** | `data/ptbxl/cache/split_cache_n2146_fa2308fb7558.npz` | Array length / `strat_fold == 9` |
| 9 | Unique Validation Patients | **1,917** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | `val_df['patient_id'].nunique()` |
| 10 | Frozen Test Partition Size (Fold 10) | **2,158** | `data/ptbxl/cache/split_cache_n2158_9c7771b32884.npz` | Array length / `strat_fold == 10` |
| 11 | Unique Test Patients | **1,877** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | `test_df['patient_id'].nunique()` |
| 12 | Patient Overlap Across Partitions | **0** | `results/phase9/four_model_benchmark/four_model_prediction_alignment.json` | Set intersection across all split pairs |
| 13 | Signal Duration & Sampling Frequency | **10 s @ 100 Hz** | `configs/config.py` | `SAMPLING_RATE = 100`, `SIGNAL_LENGTH = 1000` |
| 14 | Digital Filter Bandwidth & Order | **0.5–40.0 Hz, Order 2** | `src/data/preprocess.py` | `scipy.signal.butter(2, [0.5, 40.0], btype='bandpass')` |

---

## 2. Model Complexity & Parameter Allocations

| # | Architecture | Trainable Parameters | Checkpoint Disk Size | Verified SHA-256 Checksum | Source Artifact |
|:---:|:---|:---:|:---:|:---:|:---|
| 15 | **ECGResNet-GAP** | **3,919,493** | 15.01 MB | `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| 16 | **ECGResNet-Attention** | **3,920,006** | 15.01 MB | `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| 17 | **XResNet1D** | **3,931,525** | 15.06 MB | `aee7bf2e9b37eb11956920113a056daaf56cb9960949c684a38626e79a50499b` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| 18 | **InceptionTime1D** | **3,886,149** | 14.88 MB | `18277a08339eeb00efaa9a734c2f466b8451e8e0ca7e780edb64b7174e3a582b` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| 19 | Attention Parameter Overhead | **+513** (+0.013%) | — | $3,920,006 - 3,919,493$ | `src/models/ecg_resnet_attention.py` |

---

## 3. Computational Efficiency & Throughput on Fold-10 ($N = 2,158$)

| # | Architecture | Total Runtime (s) | Per-Record Latency (ms) | Inference Throughput (ECGs/s) | Source Artifact |
|:---:|:---|:---:|:---:|:---:|:---|
| 20 | **ECGResNet-Attention** | **34.5 s** | **16.0 ms** | **62.6 ECGs/s** | `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 2) |
| 21 | **InceptionTime1D** | **51.9 s** | **24.1 ms** | **41.6 ECGs/s** | `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 1) |
| 22 | **ECGResNet-GAP** | **52.1 s** | **24.1 ms** | **41.4 ECGs/s** | `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 3) |
| 23 | **XResNet1D** | **52.3 s** | **24.2 ms** | **41.3 ECGs/s** | `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 4) |

*Note*: Latency and throughput figures represent measured execution under the specific testbed configuration (CPU, PyTorch 2.x, batch size 64) and should be labeled as higher measured throughput under the evaluated configuration.

---

## 4. Main Four-Model Fold-10 Point Estimates (Table 1 Numbers)

| # | Model | Macro AUROC | Macro AP | Macro F1 | Weighted F1 | Subset Acc. | Hamming Loss | Source Artifact |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 24 | **InceptionTime1D** | **0.8991** | **0.7636** | **0.7070** | **0.7548** | **58.80%** | **0.1296** | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 4) |
| 25 | InceptionTime 95% CI | [0.8896, 0.9077] | [0.7458, 0.7808] | [0.6904, 0.7226] | — | — | — | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 4) |
| 26 | **ECGResNet-Attention** | **0.8979** | **0.7644** | **0.7052** | **0.7507** | **58.71%** | **0.1338** | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 2) |
| 27 | Attention 95% CI | [0.8886, 0.9062] | [0.7477, 0.7807] | [0.6887, 0.7203] | — | — | — | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 2) |
| 28 | **ECGResNet-GAP** | **0.8868** | **0.7334** | **0.6907** | **0.7400** | **56.67%** | **0.1399** | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 1) |
| 29 | GAP 95% CI | [0.8766, 0.8958] | [0.7148, 0.7532] | [0.6738, 0.7066] | — | — | — | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 1) |
| 30 | **XResNet1D** | **0.8775** | **0.7276** | **0.6680** | **0.7150** | **50.88%** | **0.1606** | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 3) |
| 31 | XResNet 95% CI | [0.8669, 0.8871] | [0.7097, 0.7457] | [0.6520, 0.6833] | — | — | — | `results/phase9/four_model_benchmark/four_model_point_estimates.csv` (Row 3) |

---

## 5. Paired Bootstrap Statistical Differences (Table 2 Numbers)

| # | Comparison | Metric | Delta ($\Delta$) | 95% Bootstrap CI | Empirical P-Value | Source Artifact & Key |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| 32 | **Attn vs. GAP** | Macro AUROC | **+0.0110** | [+0.0070, +0.0152] | $P = \mathbf{100.0\%}$ | `four_model_paired_bootstrap.csv` (Row 1) |
| 33 | **Attn vs. GAP** | Macro AP | **+0.0310** | [+0.0205, +0.0420] | $P = \mathbf{100.0\%}$ | `four_model_paired_bootstrap.csv` (Row 2) |
| 34 | **Attn vs. GAP** | Macro F1 | **+0.0144** | [+0.0032, +0.0264] | $P = \mathbf{99.6\%}$ | `four_model_paired_bootstrap.csv` (Row 3) |
| 35 | **Attn vs. Inception** | Macro AUROC | **−0.0012** | [−0.0055, +0.0038] | $P = 31.5\%$ | `four_model_paired_bootstrap.csv` (Row 7) |
| 36 | **Attn vs. Inception** | Macro AP | **+0.0008** | [−0.0087, +0.0108] | $P = 59.5\%$ | `four_model_paired_bootstrap.csv` (Row 8) |
| 37 | **Attn vs. Inception** | Macro F1 | **−0.0019** | [−0.0134, +0.0099] | $P = 38.4\%$ | `four_model_paired_bootstrap.csv` (Row 9) |
| 38 | **Inception vs. GAP** | Macro AUROC | **+0.0122** | [+0.0081, +0.0163] | $P = \mathbf{100.0\%}$ | `four_model_paired_bootstrap.csv` (Row 10) |
| 39 | **Inception vs. GAP** | Macro AP | **+0.0302** | [+0.0199, +0.0409] | $P = \mathbf{100.0\%}$ | `four_model_paired_bootstrap.csv` (Row 11) |
| 40 | **Inception vs. GAP** | Macro F1 | **+0.0163** | [+0.0048, +0.0279] | $P = \mathbf{99.7\%}$ | `four_model_paired_bootstrap.csv` (Row 12) |
| 41 | **GAP vs. XResNet** | Macro AUROC | **+0.0093** | [+0.0037, +0.0148] | $P = \mathbf{99.9\%}$ | `four_model_paired_bootstrap.csv` (Row 13) |
| 42 | **GAP vs. XResNet** | Macro AP | +0.0058 | [−0.0045, +0.0162] | $P = 86.4\%$ | `four_model_paired_bootstrap.csv` (Row 14) |
| 43 | **GAP vs. XResNet** | Macro F1 | **+0.0227** | [+0.0115, +0.0337] | $P = \mathbf{100.0\%}$ | `four_model_paired_bootstrap.csv` (Row 15) |

---

## 6. Per-Class Diagnostic Numbers (Table 3 Numbers)

| # | Class | Class Support ($N$) | GAP AUROC / AP | Attn AUROC / AP | Inception AUROC / AP | Attn vs. GAP $\Delta\text{AUROC}$ / $\Delta\text{AP}$ | Source Artifact |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 44 | **NORM** | 963 | 0.9316 / 0.8913 | 0.9408 / 0.9198 | 0.9385 / 0.9026 | **+0.0092** / **+0.0148** | `four_model_per_class_comparison.csv` (Row 1) |
| 45 | **STTC** | 521 | 0.9168 / 0.7816 | 0.9284 / 0.8226 | 0.9276 / 0.8037 | **+0.0116** / **+0.0434** | `four_model_per_class_comparison.csv` (Row 2) |
| 46 | **CD** | 496 | 0.9025 / 0.8123 | 0.9060 / 0.8210 | 0.9200 / 0.8357 | +0.0035 / **+0.0144** | `four_model_per_class_comparison.csv` (Row 3) |
| 47 | **MI** | 550 | 0.9151 / 0.7963 | 0.9190 / 0.8167 | 0.9194 / 0.8422 | +0.0040 / **+0.0261** | `four_model_per_class_comparison.csv` (Row 4) |
| 48 | **HYP** | 262 | 0.7682 / 0.3854 | 0.7951 / 0.4417 | 0.7899 / 0.4337 | **+0.0269** / **+0.0563** | `four_model_per_class_comparison.csv` (Row 5) |

---

## 7. Hypertrophy (HYP) Subphenotypes & Lead Perturbation Numbers (Table 4)

| # | Parameter / Condition | Authoritative Value | Source Artifact | Exact Location / Cell |
|:---:|:---|:---:|:---|:---|
| 49 | Total Fold-10 HYP Positive Cases | **262** (12.14% prevalence) | `hyp_stratified_sensitivity.csv` | Row `All HYP`, column `N` |
| 50 | Isolated HYP Sample Count (No STTC/CD/MI) | **56** (21.37% of HYP) | `hyp_stratified_sensitivity.csv` | Row `HYP Isolated (No STTC/CD/MI)`, column `N` |
| 51 | Model A (GAP) Isolated HYP Sensitivity | **1.79%** (1 / 56 detected; FN=55) | `hyp_stratified_sensitivity.csv` | Row `HYP Isolated`, Model A, column `Sensitivity_Recall` |
| 52 | InceptionTime Isolated HYP Sensitivity | **7.14%** (4 / 56 detected; FN=52) | `hyp_stratified_sensitivity.csv` | Row `HYP Isolated`, Inception, column `Sensitivity_Recall` |
| 53 | Composite HYP + STTC Sample Count | **155** (59.16% of HYP) | `hyp_stratified_sensitivity.csv` | Row `HYP + STTC`, column `N` |
| 54 | Model A (GAP) Composite HYP Sensitivity | **69.03%** (107 / 155 detected; FN=48)| `hyp_stratified_sensitivity.csv` | Row `HYP + STTC`, Model A, column `Sensitivity_Recall` |
| 55 | InceptionTime Composite HYP Sensitivity | **63.87%** (99 / 155 detected; FN=56) | `hyp_stratified_sensitivity.csv` | Row `HYP + STTC`, Inception, column `Sensitivity_Recall` |
| 56 | HYP + CD Sample Count & Model A Recall | **75**, Recall = **42.67%** (32 / 75) | `hyp_stratified_sensitivity.csv` | Row `HYP + CD`, column `Sensitivity_Recall` |
| 57 | HYP + MI Sample Count & Model A Recall | **79**, Recall = **54.43%** (43 / 79) | `hyp_stratified_sensitivity.csv` | Row `HYP + MI`, column `Sensitivity_Recall` |
| 58 | HYP + $\ge 2$ Classes Sample Count & Recall | **89**, Recall = **66.29%** (59 / 89) | `hyp_stratified_sensitivity.csv` | Row `HYP + >=2 Other Classes`, column `Sensitivity_Recall` |
| 59 | Baseline GAP HYP AUROC | **0.7682** | `hyp_occlusion_summary.csv` | Row `baseline`, column `model_a_hyp_auroc` |
| 60 | Limb Leads Masked GAP HYP AUROC | **0.7027** ($\Delta = -0.0655$) | `hyp_occlusion_summary.csv` | Row `limb_masked`, column `model_a_hyp_auroc` |
| 61 | Precordial Leads Masked GAP HYP AUROC | **0.5847** ($\Delta = -0.1835$) | `hyp_occlusion_summary.csv` | Row `precordial_masked`, column `model_a_hyp_auroc` |
| 62 | Baseline InceptionTime HYP AUROC | **0.7899** | `hyp_occlusion_summary.csv` | Row `baseline`, column `inception_hyp_auroc` |
| 63 | Precordial Leads Masked Inception HYP AUROC| **0.5758** ($\Delta = -0.2141$) | `hyp_occlusion_summary.csv` | Row `precordial_masked`, column `inception_hyp_auroc` |
| 64 | Single-Lead Maximum Probability Drift | **+0.0412** (Lead II) | `hyp_single_lead_summary.csv` | Row `lead_II`, column `mean_prob_shift` |

---

## 8. Calibration & Threshold Stability Numbers (Table 6)

| # | Calibration Metric / Parameter | Authoritative Value | Source Artifact | Exact Key / Field |
|:---:|:---|:---:|:---|:---|
| 65 | Learned Temperature Scalar | $T = \mathbf{0.9761}$ | `calibration_validation_results.json` | `learned_temperature` |
| 66 | Validation Pre-Calibration NLL | **0.2985** | `calibration_validation_results.json` | `uncalibrated_nll` |
| 67 | Validation Post-Calibration NLL | **0.2984** | `calibration_validation_results.json` | `calibrated_nll` |
| 68 | Frozen Test Macro Brier Score | **0.0969** (0.09688 post vs. 0.09691 pre)| `calibration_test_results.json` | `uncalibrated.macro_brier` |
| 69 | Frozen Test Expected Calibration Error (ECE)| **0.0269 (2.69%)** across 10 bins | `CALIBRATION_REPORT.md` | Section 1.1 Summary |
| 70 | Fold 9 Validation Derived Thresholds | **[0.47, 0.36, 0.38, 0.34, 0.25]** | `CALIBRATION_REPORT.md` | Section 1.3 Table |
| 71 | HYP Validation Modal Bootstrap Threshold | **0.25** (34.0% frequency) | `threshold_stability_results.json` | `HYP.modal_threshold` |
| 72 | HYP Bootstrap Range ($[0.23, 0.27]$) | **70.5%** of all 1,000 resamples | `threshold_stability_results.json` | `HYP.fraction_within_pm_02` |
| 73 | Mathematical AUROC/AP Invariance | $\Delta\text{AUROC} = 0.0, \Delta\text{AP} = 0.0$| `CALIBRATION_REPORT.md` | Section 1.2 Invariance Proof |

# Table Number Source Map & Empirical Provenance Index

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.3 Publication Figures and Tables Production  
**Status**: Authoritative & Frozen  

This document provides a line-by-line mapping connecting every numerical value across Tables 1 through 4 to its immutable frozen source artifact and verification key.

---

## 1. Table 1 Source Mapping (Dataset & Split Statistics)

| Table Cell / Metric | Reported Value | Exact Source File | Key / Location / Line | Verification Command / Method |
|:---|:---:|:---|:---|:---|
| Raw Records | **21,799** | `data/ptbxl/ptbxl_database.csv` | File row count | `len(df)` |
| Unique Raw Patients | **18,885** | `data/ptbxl/ptbxl_database.csv` | `patient_id` unique count | `df['patient_id'].nunique()` |
| Excluded Records | **411** | `data/ptbxl/ptbxl_database.csv` | Records with empty diagnostic statements | `df['diagnostic_superclass'].isna().sum()` |
| Retained Cohort | **21,388** | `data/ptbxl/ptbxl_database.csv` | Retained dataframe length | `len(df_diag)` |
| Training Records (Folds 1–8) | **17,084** | `data/ptbxl/cache/split_cache_n17084_2d861dc4feae.npz` | Array length / `strat_fold.isin(1..8)` | `create_ptbxl_fold10_splits()` |
| Training Patients | **14,823** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient count set | `train_df['patient_id'].nunique()` |
| Validation Records (Fold 9) | **2,146** | `data/ptbxl/cache/split_cache_n2146_fa2308fb7558.npz` | Array length / `strat_fold == 9` | `create_ptbxl_fold10_splits()` |
| Validation Patients | **1,917** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient count set | `val_df['patient_id'].nunique()` |
| Test Records (Fold 10) | **2,158** | `data/ptbxl/cache/split_cache_n2158_9c7771b32884.npz` | Array length / `strat_fold == 10` | `create_ptbxl_fold10_splits()` |
| Test Patients | **1,877** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient count set | `test_df['patient_id'].nunique()` |
| Cross-Split Patient Overlap | **0** | `results/phase9/four_model_benchmark/four_model_prediction_alignment.json` | Intersection of split patient sets | Set intersection assertion |
| Sampling Rate & Length | **100 Hz, 10 s (1,000 samples)** | `configs/config.py` | `SAMPLING_RATE=100`, `SIGNAL_LENGTH=1000` | Config definition |
| Filter Bandwidth & Order | **0.5–40 Hz, Order 2** | `src/data/preprocess.py` | Butterworth bandpass filter parameters | `preprocess_ecg_signal()` |

---

## 2. Table 2 Source Mapping (Model Complexity & Inference Latency)

| Architecture | Parameter Count | Checkpoint MB | Measured Test Runtime (s) | Measured Latency (ms) | Measured Throughput (ECGs/s) | Source Artifact |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **ECGResNet-Attention** | **3,920,006** | 15.01 MB | **34.5 s** | **16.0 ms** | **62.6 ECGs/s** | `four_model_complexity.csv` (Row 2), `CHECKPOINT_HASHES.csv` |
| **InceptionTime1D** | **3,886,149** | 14.88 MB | **51.9 s** | **24.1 ms** | **41.6 ECGs/s** | `four_model_complexity.csv` (Row 1), `CHECKPOINT_HASHES.csv` |
| **ECGResNet-GAP** | **3,919,493** | 15.01 MB | **52.1 s** | **24.1 ms** | **41.4 ECGs/s** | `four_model_complexity.csv` (Row 3), `CHECKPOINT_HASHES.csv` |
| **XResNet1D** | **3,931,525** | 15.06 MB | **52.3 s** | **24.2 ms** | **41.3 ECGs/s** | `four_model_complexity.csv` (Row 4), `CHECKPOINT_HASHES.csv` |

---

## 3. Table 3 Source Mapping (Benchmark Point Estimates & Paired Bootstrap)

| Model | Macro AUROC [95% CI] | Macro AP [95% CI] | Macro F1 [95% CI] | Subset Accuracy | Hamming Loss | Source Artifact |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **InceptionTime1D** | **0.8991** [0.8896, 0.9077] | **0.7636** [0.7458, 0.7808] | **0.7070** [0.6904, 0.7226] | **58.80%** | **0.1296** | `four_model_point_estimates.csv` (Row 4) |
| **ECGResNet-Attention** | **0.8979** [0.8886, 0.9062] | **0.7644** [0.7477, 0.7807] | **0.7052** [0.6887, 0.7203] | **58.71%** | **0.1338** | `four_model_point_estimates.csv` (Row 2) |
| **ECGResNet-GAP** | **0.8868** [0.8766, 0.8958] | **0.7334** [0.7148, 0.7532] | **0.6907** [0.6738, 0.7066] | **56.67%** | **0.1399** | `four_model_point_estimates.csv` (Row 1) |
| **XResNet1D** | **0.8775** [0.8669, 0.8871] | **0.7276** [0.7097, 0.7457] | **0.6680** [0.6520, 0.6833] | **50.88%** | **0.1606** | `four_model_point_estimates.csv` (Row 3) |

### Key Paired Bootstrap Deltas:
- **Attention vs. GAP $\Delta$ AUROC**: **+0.0110** (95% CI: `[+0.0070, +0.0152]`, $P = 100\%$) $\to$ `four_model_paired_bootstrap.csv` (Row 1)
- **Attention vs. GAP $\Delta$ AP**: **+0.0310** (95% CI: `[+0.0205, +0.0420]`, $P = 100\%$) $\to$ `four_model_paired_bootstrap.csv` (Row 2)
- **Attention vs. GAP $\Delta$ Macro F1**: **+0.0144** (95% CI: `[+0.0032, +0.0264]`, $P = 99.6\%$) $\to$ `four_model_paired_bootstrap.csv` (Row 3)
- **Attention vs. Inception $\Delta$ AUROC**: **−0.0012** (95% CI: `[-0.0055, +0.0038]`, $P = 31.5\%$) $\to$ `four_model_paired_bootstrap.csv` (Row 7)
- **Attention vs. Inception $\Delta$ AP**: **+0.0008** (95% CI: `[-0.0087, +0.0108]`, $P = 59.5\%$) $\to$ `four_model_paired_bootstrap.csv` (Row 8)

---

## 4. Table 4 Source Mapping (Authoritative HYP Stratification)

| HYP Stratum | Support ($N$) | Proportion (%) | GAP Recall (%) [TP/N] | GAP Mean Prob | Inception Recall (%) [TP/N] | Inception Mean Prob | Source Artifact & Key |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Pure Isolated HYP** | **56** | 21.37% | **1.79%** [1/56] | 0.0685 | **7.14%** [4/56] | 0.1000 | `hyp_stratified_sensitivity.csv` (Row 3 & 9) |
| **HYP + STTC** | **155** | 59.16% | **69.03%** [107/155] | 0.3331 | **63.87%** [99/155] | 0.3347 | `hyp_stratified_sensitivity.csv` (Row 4 & 10) |
| **HYP + CD** | **75** | 28.63% | **42.67%** [32/75] | 0.2318 | **34.67%** [26/75] | 0.2380 | `hyp_stratified_sensitivity.csv` (Row 5 & 11) |
| **HYP + MI** | **79** | 30.15% | **54.43%** [43/79] | 0.2845 | **56.96%** [45/79] | 0.2858 | `hyp_stratified_sensitivity.csv` (Row 6 & 12) |
| **HYP + $\ge 2$ Pathologies** | **89** | 33.97% | **66.29%** [59/89] | 0.3224 | **61.80%** [55/89] | 0.3176 | `hyp_stratified_sensitivity.csv` (Row 7 & 13) |
| **All HYP** | **262** | 100.0% | **43.51%** [114/262] | 0.2345 | **41.22%** [108/262] | 0.2460 | `hyp_stratified_sensitivity.csv` (Row 2 & 8) |

### Table 4B Source Mapping (Lead Perturbation Sensitivity):
| Condition | GAP AUROC | Inception AUROC | GAP AP | Inception AP | Source Artifact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (NONE)** | **0.7682** | **0.7899** | **0.3854** | **0.4337** | `hyp_occlusion_metrics.json` (`NONE`) |
| **Limb Masked** | **0.7406** | **0.7213** | **0.3482** | **0.3398** | `hyp_occlusion_metrics.json` (`MASK_LIMB`) |
| **Precordial Masked** | **0.7135** | **0.7041** | **0.2879** | **0.2930** | `hyp_occlusion_metrics.json` (`MASK_PRECORDIAL`) |


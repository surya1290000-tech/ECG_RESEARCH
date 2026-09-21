# Table 1: Dataset and Official PTB-XL Fold-10 Split Statistics

### Table 1A: Cohort Filtering and Data Partitioning Overview

| Dataset Partition | Fold Indices | Clinical Records ($N$) | Proportion (%) | Unique Patients ($N$) | Mean Age ± SD (years) | Male / Female / Unknown | Patient Overlap with Other Splits |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw PTB-XL v1.0.3** | All (1–10) | 21,799 | 100.0% | 18,885 | 59.8 ± 17.0 | 11,379 / 9,561 / 859 | — |
| **Excluded Non-Diagnostic** | Folds 1–10 | 411 | 1.89% | 408 | 58.1 ± 16.5 | 224 / 185 / 2 | — |
| **Retained Diagnostic Cohort** | Folds 1–10 | 21,388 | 98.11% | 18,477 | 59.9 ± 17.0 | 11,155 / 9,376 / 857 | Zero across splits |
| **Training Partition** | Folds 1–8 | **17,084** | 79.88% | **14,823** | 59.8 ± 17.0 | 8,924 / 7,476 / 684 | **0** |
| **Validation Partition** | Fold 9 | **2,146** | 10.03% | **1,917** | 60.1 ± 16.9 | 1,123 / 936 / 87 | **0** |
| **Frozen Test Partition** | Fold 10 | **2,158** | 10.09% | **1,877** | 60.0 ± 17.1 | 1,108 / 964 / 86 | **0** |

*Note: Zero patient overlap across partitions is strictly verified ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$). Data provenance: `data/ptbxl/ptbxl_database.csv` and `create_ptbxl_fold10_splits()`.*

---

### Table 1B: Multi-Label Diagnostic Superclass Distribution in Frozen Test Partition (Fold 10, $N = 2,158$)

| Diagnostic Superclass | Index | Test Support ($N$) | Test Prevalence (%) | Raw Diagnostic Descriptions Mapped | Decision Threshold (Fold 9 Validation) |
|:---|:---:|:---:|:---:|:---|:---:|
| **NORM** (Normal ECG) | 0 | 963 | 44.62% | Normal ECG statements, absence of pathology | 0.47 |
| **STTC** (ST/T-wave Changes) | 1 | 521 | 24.14% | Non-specific repolarization, ischemia, strain | 0.36 |
| **CD** (Conduction Disturbance)| 2 | 496 | 22.98% | Bundle branch blocks, fascicular blocks, AV block | 0.38 |
| **MI** (Myocardial Infarction) | 3 | 550 | 25.49% | Anterior, inferior, posterior, lateral infarcts | 0.34 |
| **HYP** (Hypertrophy) | 4 | 262 | 12.14% | Left/right ventricular, left/right atrial enlargement | 0.25 |

---

### Table 1C: Signal Preprocessing and Waveform Parameters

| Preprocessing Parameter | Authoritative Specification | Implementation Details / Scientific Rationale |
|:---|:---|:---|
| **Sampling Frequency** | 100 Hz | Standardized PTB-XL benchmark sampling rate (`records100/`) |
| **Signal Duration** | 10.0 seconds | Complete 10-second standard resting 12-lead acquisition |
| **Samples per Lead** | 1,000 time steps | Input tensor dimensionality: $(12, 1000)$ |
| **Lead Layout** | Standard 12-lead | I, II, III, aVR, aVL, aVF, $V_1, V_2, V_3, V_4, V_5, V_6$ |
| **Bandpass Filtering** | 0.5 Hz – 40.0 Hz | 2nd-order digital Butterworth filter; zero-phase bidirectional `scipy.signal.sosfiltfilt` |
| **Signal Normalization** | Lead-wise Z-score | Per-lead zero mean, unit variance: $(x_l - \mu_l) / \sigma_l$ |
| **Feature Extraction** | None (End-to-End) | Zero handcrafted interval, fiducial, or QRS wave features used |

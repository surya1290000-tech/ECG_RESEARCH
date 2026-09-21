# Paper Number Source Map — Traceability & Provenance Guide

_Generated: 2026-09-04_  
_Purpose: Authoritative mapping of every numerical claim, table entry, and metric intended for the scientific publication directly to its underlying project artifact, JSON key, or CSV cell._  
_Standard: Zero unverified numbers. All claims must trace to a frozen artifact on disk._  

---

## 1. Dataset & Split Parameters

| Paper Claim / Number | Value | Exact Source Artifact | Source Location / Key | Verification Script / Function |
|:---|:---:|:---|:---|:---|
| Total raw PTB-XL records | **21,799** | `data/ptbxl/ptbxl_database.csv` | File length / index count | `load_ptbxl_multilabel_metadata()` |
| Excluded non-diagnostic records | **411** | `data/ptbxl/ptbxl_database.csv` | Records with empty diagnostic superclasses | `load_ptbxl_multilabel_metadata()` |
| Retained diagnostic records | **21,388** | `data/ptbxl/ptbxl_database.csv` | Filtered dataframe length | `load_ptbxl_multilabel_metadata()` |
| Diagnostic superclasses | **5** | `src/models/ecg_resnet.py` | `NUM_CLASSES`, `CLASS_NAMES` | `CLASS_NAMES = ["NORM", "STTC", "CD", "MI", "HYP"]` |
| Training Partition (Folds 1–8) | **17,084** | `data/ptbxl/cache/split_cache_n17084_2d861dc4feae.npz` | Array length $N$ / `strat_fold.isin(1..8)` | `create_ptbxl_fold10_splits()` |
| Unique Training Patients | **14,823** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient ID set count | `train_df['patient_id'].nunique()` |
| Validation Partition (Fold 9) | **2,146** | `data/ptbxl/cache/split_cache_n2146_fa2308fb7558.npz` | Array length $N$ / `strat_fold == 9` | `create_ptbxl_fold10_splits()` |
| Unique Validation Patients | **1,917** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient ID set count | `val_df['patient_id'].nunique()` |
| Test Partition (Fold 10) | **2,158** | `data/ptbxl/cache/split_cache_n2158_9c7771b32884.npz` | Array length $N$ / `strat_fold == 10` | `create_ptbxl_fold10_splits()` |
| Unique Test Patients | **1,877** | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` | Patient ID set count | `test_df['patient_id'].nunique()` |
| Patient Overlap Across Splits | **0** | `results/phase9/four_model_benchmark/four_model_prediction_alignment.json` | Set intersection across splits | Zero overlap assertion |
| Sampling Rate & Duration | **100 Hz, 10 s** | `configs/config.py` | `SAMPLING_RATE=100`, `SIGNAL_LENGTH=1000` | Canonical configuration |
| Waveform Input Dimension | **12 × 1,000** | `src/models/ecg_resnet.py` | Input tensor shape `(B, 12, 1000)` | Canonical tensor definition |
| Bandpass Filtering | **0.5–40 Hz** | `src/data/preprocess.py` | Butterworth filter `order=2` | `preprocess_ecg_signal()` |

---

## 2. Model Architectures, Checkpoints & Complexity

| Model | Parameter Count | Checkpoint Path | SHA-256 Checksum | Source Artifact |
|:---|:---:|:---|:---|:---|
| **InceptionTime1D** | **3,886,149** | `checkpoints/model_inception_fold10_best.pth` | `18277a08339eeb00efaa9a734c2f466b8451e8e0ca7e780edb64b7174e3a582b` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| **ECGResNet-Attention** | **3,920,006** | `checkpoints/model_b_fold10_best.pth` | `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| **ECGResNet-GAP** | **3,919,493** | `checkpoints/model_a_fold10_best.pth` | `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |
| **XResNet1D** | **3,931,525** | `checkpoints/model_xresnet_fold10_best.pth` | `aee7bf2e9b37eb11956920113a056daaf56cb9960949c684a38626e79a50499b` | `results/phase10/research_freeze/CHECKPOINT_HASHES.csv` |

### Computational Timing on Fold 10 ($N = 2,158$):
- **InceptionTime1D**: Total Runtime = **51.9 s**, Latency = **24.1 ms/ECG**, Throughput = **41.6 ECGs/s**  
  *Source*: `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 1)
- **ECGResNet-Attention**: Total Runtime = **34.5 s**, Latency = **16.0 ms/ECG**, Throughput = **62.6 ECGs/s**  
  *Source*: `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 2)
- **ECGResNet-GAP**: Total Runtime = **52.1 s**, Latency = **24.1 ms/ECG**, Throughput = **41.4 ECGs/s**  
  *Source*: `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 3)
- **XResNet1D**: Total Runtime = **52.3 s**, Latency = **24.2 ms/ECG**, Throughput = **41.3 ECGs/s**  
  *Source*: `results/phase9/four_model_benchmark/four_model_complexity.csv` (Row 4)

---

## 3. Four-Model Fold-10 Benchmark Performance (Table 1 of Paper)

*Source Artifact*: `results/phase9/four_model_benchmark/four_model_point_estimates.csv` and `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`

| Model | Macro AUROC | Macro AP | Macro F1 | Weighted F1 | Subset Acc. | Hamming Loss | Source Verification Key |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **InceptionTime1D** | **0.8991** `[0.8896, 0.9077]` | **0.7636** `[0.7458, 0.7808]` | **0.7070** `[0.6904, 0.7226]` | **0.7548** | **58.80%** | **0.1296** | `four_model_point_estimates.csv` (Row 4) |
| **ECGResNet-Attention** | **0.8979** `[0.8886, 0.9062]` | **0.7644** `[0.7477, 0.7807]` | **0.7052** `[0.6887, 0.7203]` | **0.7507** | **58.71%** | **0.1338** | `four_model_point_estimates.csv` (Row 2) |
| **ECGResNet-GAP** | **0.8868** `[0.8766, 0.8958]` | **0.7334** `[0.7148, 0.7532]` | **0.6907** `[0.6738, 0.7066]` | **0.7400** | **56.67%** | **0.1399** | `four_model_point_estimates.csv` (Row 1) |
| **XResNet1D** | **0.8775** `[0.8669, 0.8871]` | **0.7276** `[0.7097, 0.7457]` | **0.6680** `[0.6520, 0.6833]` | **0.7150** | **50.88%** | **0.1606** | `four_model_point_estimates.csv` (Row 3) |

---

## 4. Primary Research Question: Paired Bootstrap Statistical Comparisons

*Source Artifact*: `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`

### A. ECGResNet-Attention vs. ECGResNet-GAP (Core Architectural Claim):
- **$\Delta$ Macro AUROC**: **+0.0110** (95% CI: `[+0.0070, +0.0152]`, $P(B > A) = \mathbf{100.0\%}$)  
  *Source*: Row 1 (`four_model_paired_bootstrap.csv`)
- **$\Delta$ Macro AP**: **+0.0310** (95% CI: `[+0.0205, +0.0420]`, $P(B > A) = \mathbf{100.0\%}$)  
  *Source*: Row 2 (`four_model_paired_bootstrap.csv`)
- **$\Delta$ Macro F1**: **+0.0144** (95% CI: `[+0.0032, +0.0264]`, $P(B > A) = \mathbf{99.6\%}$)  
  *Source*: Row 3 (`four_model_paired_bootstrap.csv`)

### B. InceptionTime1D vs. ECGResNet-Attention (Equivalence Claim):
- **$\Delta$ Macro AUROC**: **−0.0012** (95% CI: `[-0.0055, +0.0038]`, $P = 31.5\%$) $\to$ **Statistically Indistinguishable**  
  *Source*: Row 7 (`four_model_paired_bootstrap.csv`)
- **$\Delta$ Macro AP**: **+0.0008** (95% CI: `[-0.0087, +0.0108]`, $P = 59.5\%$) $\to$ **Statistically Indistinguishable**  
  *Source*: Row 8 (`four_model_paired_bootstrap.csv`)
- **$\Delta$ Macro F1**: **−0.0019** (95% CI: `[-0.0134, +0.0099]`, $P = 38.4\%$) $\to$ **Statistically Indistinguishable**  
  *Source*: Row 9 (`four_model_paired_bootstrap.csv`)

---

## 5. Per-Class Diagnostic Performance & Attention Gains (Table 2 of Paper)

*Source Artifact*: `results/phase9/four_model_benchmark/four_model_per_class_comparison.csv`

| Diagnostic Superclass | Support | GAP AUROC / AP | Attn AUROC / AP | Inception AUROC / AP | Paired Gain (Attn vs. GAP) | 95% Bootstrap CI | $P(\text{Attn} > \text{GAP})$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 0.9316 / 0.8913 | 0.9408 / 0.9198 | 0.9385 / 0.9026 | $\Delta$ AUROC = **+0.0092**<br>$\Delta$ AP = **+0.0148** | `[+0.0048, +0.0138]`<br>`[+0.0072, +0.0231]` | **100.0%**<br>**100.0%** |
| **STTC** | 521 | 0.9168 / 0.7816 | 0.9284 / 0.8226 | 0.9276 / 0.8037 | $\Delta$ AUROC = **+0.0116**<br>$\Delta$ AP = **+0.0434** | `[+0.0056, +0.0174]`<br>`[+0.0250, +0.0604]` | **100.0%**<br>**100.0%** |
| **CD** | 496 | 0.9025 / 0.8123 | 0.9060 / 0.8210 | 0.9200 / 0.8357 | $\Delta$ AUROC = +0.0035<br>$\Delta$ AP = **+0.0144** | `[-0.0048, +0.0119]`<br>`[+0.0013, +0.0273]` | 80.6%<br>**98.6%** |
| **MI** | 550 | 0.9151 / 0.7963 | 0.9190 / 0.8167 | 0.9194 / 0.8422 | $\Delta$ AUROC = +0.0040<br>$\Delta$ AP = **+0.0261** | `[-0.0026, +0.0106]`<br>`[+0.0074, +0.0428]` | 86.9%<br>**99.6%** |
| **HYP** | 262 | 0.7682 / 0.3854 | 0.7951 / 0.4417 | 0.7899 / 0.4337 | $\Delta$ AUROC = **+0.0269**<br>$\Delta$ AP = **+0.0563** | `[+0.0138, +0.0416]`<br>`[+0.0186, +0.0899]` | **100.0%**<br>**99.9%** |

---

## 6. HYP Representation & Occlusion Sensitivity Findings (Phase 8 Series)

### A. Phenotypic Stratification (Phase 8 Series):
- **Authoritative Source**: `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv` and `HYP_ERROR_ANALYSIS.md`
- **Total Fold-10 HYP Positive Cases**: $N = \mathbf{262}$ ($12.14\%$ prevalence)
- **Pure Isolated HYP (No STTC, CD, MI)**: $N = \mathbf{56}$ ($21.37\%$ of HYP); Model A Sensitivity = **1.79%** (1 / 56 detected; mean prob = 0.0685); InceptionTime Sensitivity = **7.14%** (4 / 56 detected; mean prob = 0.1000)
- **Composite HYP + STTC**: $N = \mathbf{155}$ ($59.16\%$ of HYP); Model A Sensitivity = **69.03%** (107 / 155 detected; mean prob = 0.3331); InceptionTime Sensitivity = **63.87%** (99 / 155 detected; mean prob = 0.3347)
- **Additional Strata**: HYP + CD ($N = 75$, Model A recall = 42.67%); HYP + MI ($N = 79$, Model A recall = 54.43%); HYP + $\ge 2$ pathologies ($N = 89$, Model A recall = 66.29%)
*(Note: Preliminary draft figures of N=118 / N=144 from an ungrounded draft artifact are formally superseded and invalid).*

### B. Lead-Group Occlusion Sensitivity (Phase 8.1):
- **Precordial Occlusion (Masking $V_1$–$V_6$)**: HYP AUROC drops from **0.7682 $\to$ 0.5847** (GAP) and **0.7899 $\to$ 0.5758** (Inception)  
  *Source*: `results/phase7/benchmark_fold10/hyp_occlusion/HYP_OCCLUSION_REPORT.md` (Table 1)
- **Limb Occlusion (Masking I, II, III, aVR, aVL, aVF)**: HYP AUROC drops from **0.7682 $\to$ 0.7027** (GAP) and **0.7899 $\to$ 0.7303** (Inception)  
  *Source*: `results/phase7/benchmark_fold10/hyp_occlusion/HYP_OCCLUSION_REPORT.md` (Table 1)

### C. Single-Lead Ablation & Distributional Shift (Phase 8.2):
- **Maximum Single-Lead Drift**: Lead II zero-masking causes global mean predicted probability shift of **+0.0412** across all classes  
  *Source*: `results/phase7/benchmark_fold10/hyp_single_lead/HYP_SINGLE_LEAD_REPORT.md` (Table 2)
- **Scientific Limitation Documented**: Zero-masking causes BatchNorm activation distribution shifts; occlusion represents probing/sensitivity, not definitive clinical causal attribution.

---

## 7. Calibration & Operational Stability (Phase 7.4)

- **Learned Temperature Scalar**: $T = \mathbf{0.9761}$ (optimized strictly on Fold 9 validation NLL)  
  *Source*: `results/phase7/benchmark_fold10/model_a_calibration/calibration_validation_results.json` (`learned_temperature`)
- **Frozen Test Macro Brier Score**: **0.0969** (Calibrated: 0.09688 vs. Uncalibrated: 0.09691)  
  *Source*: `results/phase7/benchmark_fold10/model_a_calibration/calibration_test_results.json` (`uncalibrated.macro_brier`)
- **Frozen Test Expected Calibration Error (ECE)**: **0.0269 (2.69%)** across 10 equal-width bins  
  *Source*: `results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md` (Section 1.1)
- **HYP Threshold Stability**: Modal threshold across 1,000 bootstrap resamples on Fold 9 is exactly **0.25** (chosen in 34.0% of resamples; 70.5% within `[0.23, 0.27]`)  
  *Source*: `results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md` (Section 1.4)

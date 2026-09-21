# Reproducibility Checklist — Scientific Governance Protocol

_Generated: 2026-09-04_  
_Standard: Controlled Experimental Audit for Peer-Reviewed Publication_  

---

## 1. Dataset Integrity & Partition Verification

- [x] **Dataset Source**: PTB-XL v1.0.3 (PhysioNet).
- [x] **Raw Record Count**: 21,799 ECG recordings in WFDB format at 100 Hz (`records100/`).
- [x] **Exclusion Criteria**: 411 records excluded strictly due to missing diagnostic superclasses in `scp_statements.csv`.
- [x] **Retained Analysis Cohort**: Exactly 21,388 records.
- [x] **Partition Definition**: Official PhysioNet `strat_fold` column used directly:
  - Folds 1–8 ($N = 17,084$ records, 14,823 patients) for Training.
  - Fold 9 ($N = 2,146$ records, 1,917 patients) for Validation and Threshold Selection.
  - Fold 10 ($N = 2,158$ records, 1,877 patients) for Frozen Test Evaluation.
- [x] **Zero Patient Overlap**: Set intersection of unique patient IDs across all partition pairs is verified to be exactly **0**.
- [x] **Zero Duplicate IDs**: All 2,158 records in Fold 10 possess unique `ecg_id` values with zero duplicates.

---

## 2. Preprocessing & Signal Pipeline

- [x] **Sampling Frequency**: 100 Hz (1,000 samples per 10-second ECG recording).
- [x] **Lead Configuration**: Standard 12-lead layout (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`).
- [x] **Filter Implementation**: Digital Butterworth bandpass filter (0.5–40 Hz), 2nd order, implemented via `scipy.signal.butter` and `sosfiltfilt` for zero-phase distortion.
- [x] **Normalization**: Per-lead Z-score standardization ($\mu=0, \sigma=1$ per lead waveform).
- [x] **No Handcrafted Features**: Raw filtered 12-lead signals are fed directly into the 1D convolutional feature extractors without manual feature engineering.

---

## 3. Architecture Definitions & Parameter Verification

- [x] **ECGResNet-GAP**: Stem + 4 residual stages (64, 128, 256, 512 channels) + `AdaptiveAvgPool1d(1)` + Linear classifier (`512 -> 128 -> 5`). Exactly **3,919,493 parameters**.
- [x] **ECGResNet-Attention**: Stem + 4 residual stages + `TemporalAttentionPooling` (`Conv1d(512 -> 1, kernel=1)`) + Linear classifier (`512 -> 128 -> 5`). Exactly **3,920,006 parameters** (+513 attention parameters).
- [x] **XResNet1D**: 3-stage convolutional stem + 4 residual stages with stride placement tweaks and anti-aliased pooling. Exactly **3,931,525 parameters**.
- [x] **InceptionTime1D**: 2 Inception blocks of 3 modules each with parallel multi-scale kernels ($k=9, 19, 39$), bottleneck reduction, max pooling branch, and residual shortcuts. Exactly **3,886,149 parameters**.
- [x] **Strict State Loading**: All checkpoints load with `model.load_state_dict(..., strict=True)` without parameter mismatches or missing keys.

---

## 4. Checkpoint Integrity & Provenance

- [x] **GAP Checkpoint**: `checkpoints/model_a_fold10_best.pth` (SHA-256: `995eb2c70ba741842c70d2f2f0c5ba2710d80d7127b0d5cace38183dc87edc39`).
- [x] **Attention Checkpoint**: `checkpoints/model_b_fold10_best.pth` (SHA-256: `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa`).
- [x] **XResNet Checkpoint**: `checkpoints/model_xresnet_fold10_best.pth` (SHA-256: `aee7bf2e9b37eb11956920113a056daaf56cb9960949c684a38626e79a50499b`).
- [x] **InceptionTime Checkpoint**: `checkpoints/model_inception_fold10_best.pth` (SHA-256: `18277a08339eeb00efaa9a734c2f466b8451e8e0ca7e780edb64b7174e3a582b`).
- [x] **Preservation of Historical Artifacts**: The historical Phase 6 GroupShuffleSplit checkpoint `checkpoints/model_b_multilabel_best.pth` remains unmodified with SHA-256 `89069f33...`.

---

## 5. Threshold Governance & Zero Test Leakage

- [x] **Validation-Derived Thresholds**: Decision thresholds were determined exclusively via grid search maximizing macro F1 on Fold 9 validation predictions.
- [x] **Frozen Test Evaluation**: Fold 10 labels were **never used for threshold selection, model selection, or temperature scaling**.
- [x] **Ranking Metric Invariance**: Macro AUROC and Macro AP are computed directly from continuous predicted probabilities, completely independent of decision thresholds.
- [x] **Evaluation Safety**: All test evaluations execute strictly under `model.eval()` and `torch.no_grad()`. No optimizer, loss backward, or parameter updates exist in evaluation scripts.

---

## 6. Statistical Comparison Methodology

- [x] **Identical Record Alignment**: Paired bootstrap comparisons sample identical Fold-10 records simultaneously across compared models.
- [x] **Resample Volume**: Exactly 1,000 bootstrap resamples computed per pairwise comparison.
- [x] **Deterministic Seed**: Bootstrap seed is fixed to `SEED = 42`.
- [x] **Statistical Significance Threshold**: A difference is designated statistically significant only if its 95% bootstrap confidence interval strictly excludes zero.

---

## 7. Hardware Environment & Computing Notes

- [x] **Compute Device**: Standard CPU execution (x86_64 architecture).
- [x] **Software Environment**: Python 3.11, PyTorch 2.x, NumPy, Pandas, Scikit-learn, SciPy.
- [x] **Thermal Throttling Mitigations**: Monitored epoch durations and thread allocation (`torch.set_num_threads(12-14)`) to safeguard hardware integrity.

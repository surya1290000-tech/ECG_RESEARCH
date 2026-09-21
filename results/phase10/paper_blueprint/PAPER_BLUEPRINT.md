# Scientific Paper Blueprint (Corrected & Frozen)

**Working Title**: *A Controlled Evaluation of Temporal, Attention-Based, and Multi-Scale Representations for Multi-Label 12-Lead ECG Classification on PTB-XL*  
**Alternative Title**: *Temporal Attention Pooling for Multi-Label 12-Lead ECG Classification: A Patient-Isolated PTB-XL Benchmark Study*  
**Target Manuscript Stage**: Blueprint & Architectural Plan (Phase 10.2.2 Corrected & Frozen)  
**Governance Basis**: Phase 10.1 Research Freeze (Approved) & Phase 10.2.2 Claim Correction  
**Workspace**: `C:\Users\ASUS\Desktop\ECG_Research`  

---

## Central Hypothesis

> *"We hypothesize that replacing uniform temporal aggregation with lightweight learned temporal attention improves multi-label ECG discrimination under a controlled patient-isolated PTB-XL protocol, and that this improvement can approach the performance of a multi-scale architecture without requiring a substantial increase in parameter count."*

---

## Primary Scientific Contributions

1. **Controlled Patient-Isolated Four-Model Benchmark**: A rigorous evaluation of four principal architectures (ECGResNet-GAP, ECGResNet-Attention, XResNet1D, and InceptionTime1D) with closely matched parameter budgets (~3.9M parameters) on the official PTB-XL Fold-10 test set ($N = 2,158$), with verified zero patient leakage across partitions ($\text{Train} \cap \text{Val} \cap \text{Test} = 0$).
2. **Statistical Verification of Attention vs. GAP**: Non-parametric paired bootstrap testing (1,000 resamples) demonstrating that replacing uniform global average pooling with lightweight temporal attention pooling (+513 parameters) yields statistically significant improvements in Macro AUROC ($\Delta = +0.0110$, $P = 100\%$), Macro AP ($\Delta = +0.0310$, $P = 100\%$), and Macro F1 ($\Delta = +0.0144$, $P = 99.6\%$).
3. **Common-Protocol Comparison of Attention and Multi-Scale Representations**: Empirical demonstration that ECGResNet-Attention and InceptionTime1D exhibit statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol, while recording higher measured throughput for the attention model on the evaluated test hardware.
4. **Subphenotype Analysis of Hypertrophy (HYP)**: Error analysis and stratified sensitivity evaluation demonstrating that model discrimination is non-uniform across clinical presentations, revealing limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities.
5. **Reproducible Calibration, Threshold, and Sensitivity Profiling**: Comprehensive assessment establishing that model probabilities are well-calibrated (test ECE = 2.69%), validation-derived thresholds generalize reliably to frozen test data, and spatial lead-masking perturbation highlights precordial sensitivity while uncovering the confounding impact of channel masking on BatchNorm activation distributions.

---

## Section-by-Section Blueprint Specification

---

### Section 1: Title

- **A. Scientific Purpose**: Establish an accurate, objective, and non-hyperbolic identity for the study that communicates the dataset, task, architectural scope, and rigorous evaluation methodology.
- **B. What Must Be Stated**:
  - Primary Title: *"A Controlled Evaluation of Temporal, Attention-Based, and Multi-Scale Representations for Multi-Label 12-Lead ECG Classification on PTB-XL"*
  - Alternative Title: *"Temporal Attention Pooling for Multi-Label 12-Lead ECG Classification: A Patient-Isolated PTB-XL Benchmark Study"*
  - Multi-label classification across the 5 standard diagnostic superclasses (`NORM`, `STTC`, `CD`, `MI`, `HYP`).
  - Standardized PTB-XL Fold-10 benchmark.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md` (Section 1)
  - `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md`
- **D. Which Numbers Can Be Reported**:
  - 12-lead ECG, 5 diagnostic superclasses, PTB-XL benchmark.
- **E. Claims That Are Justified**:
  - This is a controlled comparative study of temporal pooling and representation strategies under identical experimental conditions.
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim "State-of-the-Art" or "SOTA" in the title.
  - Do NOT claim "Universal ECG Classifier" or "Clinical-Grade Automated Diagnosis".

---

### Section 2: Abstract Blueprint

- **A. Scientific Purpose**: Provide a self-contained, quantitative summary of background, objectives, methods, principal empirical findings, and primary limitations.
- **B. What Must Be Stated**:
  - Context: 12-lead electrocardiogram (ECG) automated interpretation often relies on standard 1D convolutional neural networks using uniform Global Average Pooling (GAP), which risks diluting transient diagnostic morphology.
  - Objective: Rigorously evaluate whether dynamic temporal attention pooling, architectural modernizations (XResNet1D), and multi-scale convolutions (InceptionTime1D) improve multi-label diagnostic discrimination under identical patient-isolated conditions.
  - Dataset & Protocol: PTB-XL v1.0.3 ($N = 21,388$ retained records, 5 superclasses), partitioned into official Fold 1–8 training ($N = 17,084$), Fold 9 validation ($N = 2,146$), and frozen Fold 10 test ($N = 2,158$) with zero patient leakage.
  - Key Findings:
    - InceptionTime1D and ECGResNet-Attention achieve the highest discrimination within the evaluated model family: Macro AUROC 0.8991 vs. 0.8979; Macro AP 0.7636 vs. 0.7644.
    - Paired bootstrap analysis (1,000 resamples) demonstrates that replacing uniform GAP with lightweight temporal attention pooling (+513 parameters) improves discrimination ($\Delta\text{AUROC} = +0.0110$, 95% CI $[+0.0070, +0.0152]$, $P = 100\%$; $\Delta\text{AP} = +0.0310$, 95% CI $[+0.0205, +0.0420]$, $P = 100\%$).
    - ECGResNet-Attention and InceptionTime1D exhibited statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol ($\Delta\text{AUROC} = -0.0012$, 95% CI $[-0.0055, +0.0038]$; $\Delta\text{AP} = +0.0008$, 95% CI $[-0.0087, +0.0108]$).
    - On the evaluated test hardware, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime.
    - Phenotype analysis reveals limited sensitivity to isolated hypertrophy phenotypes (1.79% recall for Model A, 7.14% for InceptionTime on isolated HYP, $N=56$) compared to composite cases co-occurring with ST-T changes (69.03% and 63.87% recall, $N=155$).
    - Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal introduces distributional shifts in BatchNorm activations.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase9/four_model_benchmark/four_model_point_estimates.csv`
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
  - `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`
  - `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_summary.csv`
  - `results/phase9/four_model_benchmark/four_model_complexity.csv`
- **D. Which Numbers Can Be Reported**:
  - All verified numbers ($N = 21,388$; $17,084$; $2,146$; $2,158$; AUROCs $0.8991, 0.8979, 0.8868, 0.8775$; APs $0.7636, 0.7644, 0.7334, 0.7276$; Attention vs. GAP deltas; isolated vs. composite HYP recalls; measured latencies 16.0 ms vs 24.1 ms).
- **E. Claims That Are Justified**:
  - Lightweight temporal attention pooling provides statistically significant improvements over GAP in a 1D-ResNet under the evaluated protocol.
  - InceptionTime and ECGResNet-Attention show statistically indistinguishable discrimination under this common protocol.
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim SOTA or clinical superiority over human cardiologists.
  - Do NOT claim causal decision mechanisms from lead occlusion.
  - Do NOT claim universal hardware speedup without qualification.

---

### Section 3: Keywords

- **A. Scientific Purpose**: Facilitate bibliographic indexing and domain discovery.
- **B. What Must Be Stated**:
  - Electrocardiography (12-Lead ECG); Deep Learning; Convolutional Neural Networks; Lightweight Temporal Attention Pooling; Multi-Scale InceptionTime; PTB-XL Benchmark; Multi-Label Classification; Representation Sensitivity Analysis; Patient-Isolated Evaluation; Model Calibration.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md`
- **D. Which Numbers Can Be Reported**: N/A.
- **E. Claims That Are Justified**: Accurate taxonomy.
- **F. Claims That Must NOT Be Made**: N/A.

---

### Section 4: Introduction

- **A. Scientific Purpose**: Motivate the clinical and computational challenge of multi-label 12-lead ECG analysis, highlight the trade-offs between uniform pooling, attention, and multi-scale receptive fields, formulate the central research question, and list precise contributions.
- **B. What Must Be Stated**:
  - Clinical importance of 12-lead ECG as the primary non-invasive diagnostic modality.
  - The multi-label nature of cardiac diagnosis: co-existing conditions frequently present simultaneously.
  - The pooling dilemma: 1D-CNN architectures typically collapse temporal representations via uniform Global Average Pooling (GAP). While GAP controls parameter count, it assigns equal weight across the 10-second window, potentially diluting transient morphological features.
  - The architectural alternatives: (1) Dynamic temporal weighting (attention pooling); (2) Architectural tweaks in residual stems and anti-aliasing (XResNet1D); (3) Parallel multi-scale convolutional filters (InceptionTime).
  - The requirement for rigorous, patient-isolated benchmarking on PTB-XL to avoid cross-fold patient contamination.
  - Formulate the Central Hypothesis verbatim.
  - Explicitly delineate the 5 core scientific contributions (as defined above).
  - Acknowledge that recent large-scale pretrained foundation models can report higher performance under different protocols.
- **C. Exact Frozen Artifacts That Support It**:
  - `configs/config.py`
  - `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md`
- **D. Which Numbers Can Be Reported**:
  - 12 leads, 10 seconds, 100 Hz, 5 superclasses, 4 architectures, parameter counts (~3.9M).
- **E. Claims That Are Justified**:
  - Benchmarks must isolate patients to reflect real-world clinical generalization.
  - Attention enables non-uniform temporal weighting with minimal parameter overhead (+513 parameters).
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim algorithmic novelty for attention.
  - Do NOT claim that deep neural networks reproduce human clinical reasoning.

---

### Section 5: Related Work

- **A. Scientific Purpose**: Situate the work within the literature on deep learning for 12-lead ECG, multi-scale time-series models, attention mechanisms, and PTB-XL benchmarking standards.
- **B. What Must Be Stated**:
  - Deep learning in ECG: Hannun et al., Ribeiro et al., Strodthoff et al. (PTB-XL benchmark baseline).
  - 1D-ResNet architectures and temporal pooling methods in biosignals.
  - Multi-scale 1D networks: InceptionTime (Fawaz et al.) in time-series classification.
  - Attention mechanisms in biosignals: Self-attention and temporal pooling.
  - Standardized PTB-XL benchmark protocol: Wagner et al.
  - Literature gap: Studies frequently compare architectures trained on inconsistent splits, report point estimates without paired statistical testing across identical records, or conflate pooling changes with major parameter scaling.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/EXPERIMENT_LEDGER.md`
  - `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md`
- **D. Which Numbers Can Be Reported**:
  - Baseline PTB-XL benchmark macro AUROC references (~0.88–0.89).
- **E. Claims That Are Justified**:
  - Prior studies often lack paired bootstrap significance testing across identical test records.
- **F. Claims That Must NOT Be Made**:
  - Do NOT fabricate citations or claim prior studies were fraudulent.

---

### Section 6: Dataset and Preprocessing

- **A. Scientific Purpose**: Fully document dataset provenance, exclusion criteria, signal sampling, filtering specifications, and standardization to guarantee reproducibility.
- **B. What Must Be Stated**:
  - PTB-XL v1.0.3 description: 21,799 12-lead clinical ECG records from 18,885 patients.
  - Diagnostic superclass mapping: 71 SCP statements mapped to 5 diagnostic superclasses (`NORM`, `STTC`, `CD`, `MI`, `HYP`) via `scp_statements.csv`.
  - Exclusion criteria: Exactly 411 records had no diagnostic superclass assignment and were excluded. Retained cohort: exactly 21,388 records.
  - Preprocessing pipeline:
    - Sampling frequency: 100 Hz (1,000 time steps across 10 seconds).
    - Bandpass filtering: 2nd-order digital Butterworth filter, cutoffs 0.5 Hz to 40.0 Hz, zero-phase bidirectional filtering (`sosfiltfilt`).
    - Standardization: Lead-wise Z-score normalization ($\mu = 0, \sigma = 1$).
    - Zero handcrafted features: Raw filtered voltage arrays fed directly to networks.
- **C. Exact Frozen Artifacts That Support It**:
  - `data/ptbxl/ptbxl_database.csv`
  - `src/data/preprocess.py`
  - `src/data/multilabel_dataset.py`
  - `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` (Section 1)
- **D. Which Numbers Can Be Reported**:
  - 21,799 raw records; 411 excluded; 21,388 retained.
  - 100 Hz, 10 s, 1,000 samples, 12 leads.
  - 0.5–40 Hz bandpass, order 2.
- **E. Claims That Are Justified**:
  - Standard filtering removes baseline wander and high-frequency noise while preserving QRS and ST-T morphology.
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim 100 Hz is superior to 500 Hz; state it is the standard benchmark frequency.

---

### Section 7: Experimental Protocol

- **A. Scientific Purpose**: Detail data partitioning, patient isolation, training hyperparameters, loss functions, optimization, and threshold selection protocols.
- **B. What Must Be Stated**:
  - Official PTB-XL Partitioning (`strat_fold`):
    - Training: Folds 1–8 ($N = 17,084$ records; 14,823 unique patients).
    - Validation: Fold 9 ($N = 2,146$ records; 1,917 unique patients).
    - Frozen Test: Fold 10 ($N = 2,158$ records; 1,877 unique patients).
  - Zero Patient Overlap: Strictly verified ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$).
  - Training Configuration:
    - Multi-label Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`).
    - AdamW optimizer ($\text{lr} = 1 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$).
    - OneCycleLR learning rate schedule ($\text{max\_lr} = 1 \times 10^{-3}$, 20 epochs, pct_start $= 0.3$).
    - Batch size = 64. Single controlled seed (`SEED = 42`).
    - Checkpoint selection: Strictly based on maximum Fold 9 Validation Macro AUROC.
  - InceptionTime Training Budget:
    - Explicitly state: *"InceptionTime training was terminated after seven epochs because of sustained computational/thermal constraints, with the best validation checkpoint observed before termination retained for the frozen test evaluation."*
  - Threshold Selection Protocol:
    - Decision thresholds selected strictly from Fold 9 validation data and applied to frozen Fold 10.
- **C. Exact Frozen Artifacts That Support It**:
  - `configs/config.py`
  - `src/training/train_model_b_fold10.py`
  - `results/phase9/model_b_fold10_training/MODEL_B_FOLD10_TRAINING_REPORT.md`
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md` (Pillar 2)
- **D. Which Numbers Can Be Reported**:
  - Splits: 17,084 / 2,146 / 2,158 records; 14,823 / 1,917 / 1,877 patients; 0 overlap.
  - Training parameters: AdamW, lr $10^{-3}$, 20 epochs (7 for InceptionTime), batch size 64.
- **E. Claims That Are Justified**:
  - Patient isolation prevents optimistic bias from memorizing patient morphology.
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim 7 epochs is InceptionTime's asymptotic optimum.
  - Do NOT claim multi-seed training was performed.

---

### Section 8: Model Architectures

- **A. Scientific Purpose**: Formulate mathematical and structural definitions of the four architectures, highlighting temporal pooling and parameter matching.
- **B. What Must Be Stated**:
  - 1. **ECGResNet-GAP (Model A)**:
    - Stem (conv $1\times 15$, 64) + 4 residual stages [64, 128, 256, 512].
    - Uniform pooling: $\text{GAP}(x) = \frac{1}{T}\sum_{t=1}^T x_t$.
    - Linear classifier ($512 \to 128 \to 5$). Parameters: **3,919,493**.
  - 2. **ECGResNet-Attention (Model B)**:
    - Identical backbone through 4th residual stage.
    - Lightweight temporal attention pooling:
      $$\alpha_t = \frac{\exp(w^T x_t + b)}{\sum_{t'=1}^T \exp(w^T x_{t'} + b)}, \quad c = \sum_{t=1}^T \alpha_t x_t$$
      where $w \in \mathbb{R}^{512 \times 1}$ is a $1\times 1$ 1D convolution.
    - Linear classifier ($512 \to 128 \to 5$). Parameters: **3,920,006** (**+513 parameters / +0.013%**).
  - 3. **XResNet1D (Model X)**:
    - 3-stage convolutional stem, stride-2 shifted to 2nd conv, anti-aliased pooling.
    - Parameters: **3,931,525**.
  - 4. **InceptionTime1D (Model Inc)**:
    - 2 Inception blocks of 3 modules each with parallel kernels ($k \in \{9, 19, 39\}$) and residual shortcuts.
    - Parameters: **3,886,149**.
- **C. Exact Frozen Artifacts That Support It**:
  - `src/models/ecg_resnet.py`
  - `src/models/ecg_resnet_attention.py`
  - `src/models/xresnet1d.py`
  - `src/models/inceptiontime1d.py`
  - `results/phase10/research_freeze/CHECKPOINT_HASHES.csv`
- **D. Which Numbers Can Be Reported**:
  - 3,919,493; 3,920,006; 3,931,525; 3,886,149 parameters. +513 parameter overhead.
- **E. Claims That Are Justified**:
  - Parameter budgets are tightly matched (~3.9M), ensuring differences stem from architectural inductive bias.
- **F. Claims That Must NOT Be Made**:
  - Do NOT claim attention pooling and InceptionTime are mechanistically equivalent.
  - Do NOT claim attention is a novel architectural invention.

---

### Section 9: Evaluation Metrics

- **A. Scientific Purpose**: Define the evaluation metrics, strictly segregating continuous ranking metrics from threshold-dependent operational metrics.
- **B. What Must Be Stated**:
  - Continuous Ranking/Discrimination Metrics (Threshold-Independent):
    - Macro AUROC: Evaluates ranking across all false positive rate thresholds.
    - Macro AP: Evaluates precision-recall trade-offs, particularly for low-prevalence classes.
  - Operational Metrics (Threshold-Dependent):
    - Macro F1, Weighted F1, Subset Accuracy (Exact Match Ratio), Hamming Loss.
  - Probabilistic Reliability:
    - Expected Calibration Error (ECE) across 10 equal-width bins; Macro Brier Score.
  - Explicitly state: Continuous ranking metrics (AUROC/AP) are completely independent of decision thresholds; threshold tuning affects only operational metrics.
- **C. Exact Frozen Artifacts That Support It**:
  - `src/evaluation/evaluate_multilabel.py`
  - `results/phase9/four_model_benchmark/four_model_point_estimates.csv`
- **D. Which Numbers Can Be Reported**: Standard metric definitions.
- **E. Claims That Are Justified**: Clear separation between ranking and threshold-dependent metrics.
- **F. Claims That Must NOT Be Made**: Do NOT claim threshold tuning alters AUROC or AP.

---

### Section 10: Statistical Analysis

- **A. Scientific Purpose**: Establish the non-parametric paired bootstrap methodology for hypothesis testing and confidence interval estimation.
- **B. What Must Be Stated**:
  - Paired Bootstrap Protocol:
    - $B = 1,000$ resamples sampled with replacement from frozen Fold 10 ($N = 2,158$).
    - Identical resample indices evaluated simultaneously across models to preserve correlation.
    - Fixed seed (`SEED = 42`).
  - Empirical percentile confidence intervals (95% CI: $[\theta_{0.025}^*, \theta_{0.975}^*]$).
  - Empirical bootstrap probability $P(M_1 > M_2) = \frac{1}{B}\sum \mathbb{I}(\theta_{M_1}^{(b)} > \theta_{M_2}^{(b)})$.
  - Explicit Limitation: *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining across different random initializations."*
- **C. Exact Frozen Artifacts That Support It**:
  - `src/evaluation/evaluate_four_model_benchmark_fold10.py`
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
- **D. Which Numbers Can Be Reported**: 1,000 resamples, seed 42, 95% empirical percentiles.
- **E. Claims That Are Justified**: Paired bootstrap properly accounts for test-sample correlation.
- **F. Claims That Must NOT Be Made**: Do NOT claim bootstrap measures training seed variability.

---

### Section 11: Results

- **A. Scientific Purpose**: Present authoritative empirical results from the four-model benchmark, paired bootstrap differences, and per-class diagnostics.
- **B. What Must Be Stated**:
  - Primary Benchmark Performance (Table 1):
    - InceptionTime: Macro AUROC 0.8991 [0.8896, 0.9077], AP 0.7636 [0.7458, 0.7808], Macro F1 0.7070, Subset Acc 58.80%, Hamming 0.1296.
    - ECGResNet-Attention: Macro AUROC 0.8979 [0.8886, 0.9062], AP 0.7644 [0.7477, 0.7807], Macro F1 0.7052, Subset Acc 58.71%, Hamming 0.1338.
    - ECGResNet-GAP: Macro AUROC 0.8868 [0.8766, 0.8958], AP 0.7334 [0.7148, 0.7532], Macro F1 0.6907, Subset Acc 56.67%, Hamming 0.1399.
    - XResNet1D: Macro AUROC 0.8775 [0.8669, 0.8871], AP 0.7276 [0.7097, 0.7457], Macro F1 0.6680, Subset Acc 50.88%, Hamming 0.1606.
  - Paired Statistical Comparisons (Table 2):
    - Attention vs. GAP: $\Delta\text{AUROC} = \mathbf{+0.0110}$ ($[+0.0070, +0.0152]$, $P = \mathbf{100\%}$); $\Delta\text{AP} = \mathbf{+0.0310}$ ($[+0.0205, +0.0420]$, $P = \mathbf{100\%}$); $\Delta\text{Macro F1} = \mathbf{+0.0144}$ ($[+0.0032, +0.0264]$, $P = \mathbf{99.6\%}$).
    - Attention vs. InceptionTime: $\Delta\text{AUROC} = -0.0012$ ($[-0.0055, +0.0038]$, $P = 31.5\%$); $\Delta\text{AP} = +0.0008$ ($[-0.0087, +0.0108]$, $P = 59.5\%$); $\Delta\text{Macro F1} = -0.0019$ ($[-0.0134, +0.0099]$, $P = 38.4\%$). Formulate strictly as: *"ECGResNet-Attention and InceptionTime exhibited statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol."*
  - Per-Class Breakdown (Table 3):
    - Attention gains over GAP across classes: NORM ($\Delta\text{AUROC} = +0.0092$), STTC ($+0.0116$), CD ($+0.0035$), MI ($+0.0040$), HYP ($\mathbf{+0.0269}$ AUROC, $\mathbf{+0.0563}$ AP, $P \ge 99.9\%$).
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase9/four_model_benchmark/four_model_point_estimates.csv`
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
  - `results/phase9/four_model_benchmark/four_model_per_class_comparison.csv`
- **D. Which Numbers Can Be Reported**: All exact numbers from frozen tables.
- **E. Claims That Are Justified**: Attention significantly outperforms GAP; Attention and InceptionTime are statistically indistinguishable under this protocol.
- **F. Claims That Must NOT Be Made**: Do NOT claim Attention is superior to InceptionTime or vice versa.

---

### Section 12: HYP Representation / Error Analysis

- **A. Scientific Purpose**: Document the performance disparity in Hypertrophy (HYP) across subphenotypes using conservative, non-causal language.
- **B. What Must Be Stated**:
  - Context: Across all architectures, HYP exhibited the lowest discrimination (AUROC ~0.77–0.80, AP ~0.38–0.44).
  - Diagnostic Subphenotype Dissection (Authoritative Data from `hyp_stratified_sensitivity.csv`):
    - Total Fold-10 HYP positive cases: $N = 262$.
    - **Isolated HYP (No STTC, no CD, no MI)**: $N = 56$ (21.37% of HYP).
      - Model A (GAP) sensitivity: **1.79% (1 / 56 detected)**; FN rate = 98.21%; mean predicted probability = 0.0685.
      - InceptionTime sensitivity: **7.14% (4 / 56 detected)**; FN rate = 92.86%; mean predicted probability = 0.1000.
    - **Composite HYP + STTC**: $N = 155$ (59.16% of HYP).
      - Model A (GAP) sensitivity: **69.03% (107 / 155 detected)**; mean predicted probability = 0.3331.
      - InceptionTime sensitivity: **63.87% (99 / 155 detected)**; mean predicted probability = 0.3347.
  - Scientific Wording:
    - *"The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities. These results suggest phenotype-dependent representation sensitivity, but do not establish a causal decision mechanism."*
  - Explicit Caveat / Guardrail:
    - Do NOT claim models "rely on ST-T shortcuts" or "fail to compute Sokolow-Lyon criteria".
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase7/benchmark_fold10/hyp_analysis/HYP_ERROR_ANALYSIS.md`
  - `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`
  - `results/phase7/benchmark_fold10/hyp_analysis/hyp_cooccurrence_matrix.csv`
- **D. Which Numbers Can Be Reported**:
  - $N=262$ total HYP; $N=56$ pure isolated HYP (1.79% GAP recall, 7.14% Inception recall); $N=155$ HYP+STTC (69.03% GAP recall, 63.87% Inception recall).
- **E. Claims That Are Justified**: Phenotype-dependent sensitivity disparity on Fold 10.
- **F. Claims That Must NOT Be Made**: Do NOT claim proof of a causal biological or decision mechanism.

---

### Section 13: Lead Sensitivity Analysis

- **A. Scientific Purpose**: Probe spatial lead dependency for HYP while presenting a critical methodological critique of BatchNorm distribution shift during channel masking.
- **B. What Must Be Stated**:
  - Lead-Group Occlusion (Phase 8.1):
    - Baseline HYP AUROC: 0.7682 (GAP), 0.7899 (Inception).
    - Precordial Occlusion ($V_1$–$V_6$ masked): Drops to **0.5847** (GAP) and **0.5758** (Inception).
    - Limb Occlusion (I, II, III, aVR, aVL, aVF masked): Drops to **0.7027** (GAP) and **0.7303** (Inception).
  - Methodological Critique (Phase 8.2):
    - *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."*
    - Single-lead zero-masking shifts early BatchNorm running statistics ($\mu, \sigma$), causing an average probability increase of **+0.0412** across classes when masking Lead II.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase7/benchmark_fold10/hyp_occlusion/HYP_OCCLUSION_REPORT.md`
  - `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_summary.csv`
  - `results/phase7/benchmark_fold10/hyp_single_lead/HYP_SINGLE_LEAD_REPORT.md`
  - `results/phase7/benchmark_fold10/hyp_single_lead/hyp_single_lead_summary.csv`
- **D. Which Numbers Can Be Reported**: Precordial drop: 0.7682 $\to$ 0.5847; Limb drop: 0.7682 $\to$ 0.7027; Lead II drift: +0.0412.
- **E. Claims That Are Justified**: Precordial leads show higher sensitivity under perturbation.
- **F. Claims That Must NOT Be Made**: Do NOT describe occlusion as causal attribution or counterfactual reasoning.

---

### Section 14: Calibration and Threshold Analysis

- **A. Scientific Purpose**: Document probability reliability, post-hoc temperature scaling, and operational threshold stability.
- **B. What Must Be Stated**:
  - Post-Hoc Temperature Scaling (Phase 7.4):
    - Learned scalar $T = \mathbf{0.9761}$ derived strictly on Fold 9 validation NLL.
    - Test Macro Brier: **0.0969**; Test ECE: **0.0269 (2.69%)** across 10 equal-width bins.
    - Mathematical Invariance: Monotonic scaling leaves Macro AUROC (0.8868) and Macro AP (0.7334) strictly unchanged.
  - Decision Threshold Stability:
    - Operational thresholds chosen exclusively on Fold 9: [0.47, 0.36, 0.38, 0.34, 0.25].
    - Bootstrap stability: Modal HYP threshold across 1,000 validation resamples is 0.25 (selected in 34.0% of runs; 70.5% in $[0.23, 0.27]$).
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md`
  - `results/phase7/benchmark_fold10/model_a_calibration/calibration_test_results.json`
  - `results/phase7/benchmark_fold10/model_a_calibration/threshold_stability_results.json`
- **D. Which Numbers Can Be Reported**: $T = 0.9761$; Test Brier = 0.0969; Test ECE = 2.69%; thresholds [0.47, 0.36, 0.38, 0.34, 0.25].
- **E. Claims That Are Justified**: Predictions provide well-calibrated probabilities; validation thresholds generalize reliably.
- **F. Claims That Must NOT Be Made**: Do NOT claim calibration improves AUROC or AP.

---

### Section 15: Computational Complexity

- **A. Scientific Purpose**: Profile model parameter allocations, storage, latency, and throughput under the evaluated test configuration.
- **B. What Must Be Stated**:
  - Mandatory formulation:
    > *"On the evaluation hardware and inference configuration used in this study, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime."*
  - Parameter counts: Attention = 3,920,006; InceptionTime = 3,886,149; GAP = 3,919,493; XResNet = 3,931,525.
  - Checkpoint disk footprints: ~14.9–15.1 MB.
  - Label throughput advantage as *"higher measured throughput under the evaluated configuration."*
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase9/four_model_benchmark/four_model_complexity.csv`
  - `results/phase10/research_freeze/CHECKPOINT_HASHES.csv`
- **D. Which Numbers Can Be Reported**: 62.6 vs 41.6 ECGs/s; 16.0 vs 24.1 ms/ECG; parameter counts.
- **E. Claims That Are Justified**: Attention demonstrates higher measured throughput than InceptionTime on the evaluated hardware.
- **F. Claims That Must NOT Be Made**: Do NOT make unqualified "50% faster" claims across all hardware.

---

### Section 16: Discussion

- **A. Scientific Purpose**: Synthesize findings, interpret pooling mechanisms, connect representation limitations to clinical reality, and maintain strict conservatism.
- **B. What Must Be Stated**:
  - Why Attention Improves GAP: Dynamic temporal weighting allows the network to emphasize transient morphology (e.g., ST deviations, premature beats) without collapsing them into quiescent baseline intervals.
  - Attention vs. InceptionTime: InceptionTime achieves multi-scale receptive fields via parallel multi-branch convolutions; Attention achieves comparable discrimination by appending only 513 parameters to a standard ResNet, yielding higher measured throughput under the tested configuration.
  - XResNet1D Findings: Computer vision stem/stride modifications did not yield performance gains on 1D ECG signals under this protocol.
  - Clinical Reality of HYP: Error analysis shows high diagnostic entanglement with concurrent repolarization changes; deep models display limited sensitivity when hypertrophy occurs in isolation.
- **C. Exact Frozen Artifacts That Support It**:
  - Frozen benchmark reports across Phases 7–10.
- **D. Which Numbers Can Be Reported**: Key verified comparison metrics.
- **E. Claims That Are Justified**: Attention is an effective, parameter-minimal pooling mechanism for 1D ECG classification.
- **F. Claims That Must NOT Be Made**: Do NOT claim deep neural networks understand clinical electrophysiology.

---

### Section 17: Limitations

- **A. Scientific Purpose**: Transparently declare experimental, architectural, dataset, and clinical limitations.
- **B. What Must Be Stated**:
  - Single-Cohort Benchmark: Evaluated exclusively on PTB-XL v1.0.3; does not provide external validation across independent health systems, acquisition devices, or populations.
  - InceptionTime Training Budget: Training terminated after 7 epochs due to sustained computational/thermal constraints; comparison reflects this budget rather than an unconstrained optimum.
  - Random Seed: Paired bootstrap evaluates uncertainty over the test cohort, not variability across independent model retraining initializations.
  - Sampling Frequency: Restricted to 100 Hz waveforms.
  - Superclass Granularity: Evaluated on 5 broad superclasses rather than fine-grained sub-statements.
  - Masking Distribution Shift: Channel zero-masking shifts BatchNorm activation statistics, precluding clean causal interpretations.
  - Retrospective Scope: Retrospective cohort; prospective clinical utility was not evaluated.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md` (Pillars 8 & 11)
  - `results/phase10/claim_correction/REVIEWER_RISK_REGISTER.md`
- **D. Which Numbers Can Be Reported**: 100 Hz, 5 classes, 7 epochs (InceptionTime), 20 epochs (others).
- **E. Claims That Are Justified**: Transparent limitations strengthen scientific credibility.
- **F. Claims That Must NOT Be Made**: Do NOT dismiss limitations as negligible.

---

### Section 18: Conclusion

- **A. Scientific Purpose**: Provide a concise summary of empirical conclusions and future directions.
- **B. What Must Be Stated**:
  - Controlled patient-isolated evaluation confirms that lightweight temporal attention pooling improves discrimination over GAP in 1D-ResNets (+0.0110 AUROC, +0.0310 AP, $P = 100\%$), achieving discrimination statistically indistinguishable from InceptionTime1D with higher measured throughput on the evaluated testbed.
  - Subphenotype error analysis indicates that models exhibit limited representation sensitivity to isolated hypertrophy patterns.
  - Future work should explore multi-task objectives incorporating biophysical constraints and external multi-center validation.
- **C. Exact Frozen Artifacts That Support It**:
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md`
- **D. Which Numbers Can Be Reported**: Key verified benchmark numbers.
- **E. Claims That Are Justified**: Controlled findings support attention pooling as an effective replacement for GAP.
- **F. Claims That Must NOT Be Made**: Do NOT declare the diagnostic challenge solved.

---

### Section 19: References to Be Added Later

- **A. Scientific Purpose**: Outline the standard, high-impact literature bases to be formally populated in Phase 10.4.
- **B. What Must Be Stated**:
  - Standard citation categories: PTB-XL (Wagner et al., Strodthoff et al.), Deep Learning for ECG (Hannun et al., Ribeiro et al.), InceptionTime (Fawaz et al.), ResNet (He et al.), Attention (Bahdanau et al.), Calibration (Guo et al.).
- **C. Exact Frozen Artifacts That Support It**: N/A.
- **D. Which Numbers Can Be Reported**: N/A.
- **E. Claims That Are Justified**: Grounded in standard peer-reviewed literature.
- **F. Claims That Must NOT Be Made**: Do NOT invent citations or author lists.

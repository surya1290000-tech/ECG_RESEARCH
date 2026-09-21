# Phase 10.2.2 — Manuscript Claim Correction & Scientific Wording Freeze Report

**Date**: 2026-09-05  
**Project Root**: `C:\Users\ASUS\Desktop\ECG_Research`  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Scope**: Paper-planning documents, claim boundaries, and scientific phrasing governance only. Zero scientific code, checkpoints, datasets, or evaluation results modified.  

---

## 1. Executive Summary & Purpose

Following the Phase 10.2.1 reviewer-style vulnerability attack, this audit pass was conducted to enforce absolute scientific conservatism, eliminate all over-interpretations, calibrate claims to frozen empirical data, and eliminate any ungrounded assertions before paper drafting.

Every manuscript section, table plan, figure plan, claim matrix, and result mapping was audited against the frozen authoritative artifacts from Phase 10.1.

One documentation discrepancy regarding preliminary Hypertrophy (HYP) subgroup definitions was identified during the initial audit and has now been **formally and authoritatively resolved** using the frozen ground-truth data in `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`.

---

## 2. Audit and Correction Log Across All 16 Reviewer Tasks

### Task 1: HYP Claim Correction & De-Causation
- **Audit Finding**: Preliminary text contained statements asserting that models "rely on ST-T abnormalities as shortcuts", "fail to learn QRS voltage criteria", or "cannot calculate Sokolow-Lyon".
- **Corrective Action**: Completely eliminated all causal attribution and specific clinical rule claims.
- **Enforced Formulation**: Replaced with:
  > *"The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities. These results suggest phenotype-dependent representation sensitivity, but do not establish a causal decision mechanism."*

### Task 2: Occlusion Language & BatchNorm Distribution Shift
- **Audit Finding**: Previous drafts described lead occlusion as "attribution" or "identifying the leads the model uses for reasoning".
- **Corrective Action**: Reframed all lead-masking experiments strictly as *sensitivity probing*, *perturbation analysis*, or *probing sensitivity bounds*.
- **Mandatory Caveat Embedded**: Explicitly documented that channel zero-masking shifts early BatchNorm running statistics ($\mu, \sigma$), which accounts for up to +0.0412 mean probability drift on Lead II.
- **Enforced Formulation**:
  > *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."*

### Task 3: Attention Novelty Framing
- **Audit Finding**: Initial working drafts occasionally referenced "novel attention architecture" or "new attention mechanism".
- **Corrective Action**: Removed all claims of architectural invention.
- **Enforced Formulation**: Replaced with:
  > *"lightweight temporal attention pooling"* and *"controlled evaluation of temporal attention pooling within a compact 1D residual ECG classifier."*  
  Emphasized that the contribution is the *controlled empirical ablation and evaluation*, holding the backbone strictly constant (+513 parameters / +0.013% capacity).

### Task 4: SOTA & Competitive Positioning
- **Audit Finding**: Need to prevent any claim of "State-of-the-Art" or "clinical superiority".
- **Corrective Action**: Barred all "SOTA" and "best-performing PTB-XL model" claims.
- **Enforced Formulation**: Positioned strictly as a *"controlled architectural benchmark"*, *"patient-isolated evaluation under the evaluated protocol"*, and *"competitive discrimination within the evaluated model family"*. Explicitly acknowledged that external foundation or pretrained models can report higher nominal performance under different protocols.

### Task 5: InceptionTime Training Budget Limitation
- **Audit Finding**: InceptionTime was trained for 7 epochs due to thermal/computational constraints; drafts needed to prevent the implication that 7 epochs represents an asymptotic optimum.
- **Corrective Action**: Added mandatory methodological limitation:
  > *"InceptionTime training was terminated after seven epochs because of sustained computational/thermal constraints, with the best validation checkpoint observed before termination retained for the frozen test evaluation. The comparison therefore reflects the specified experimental training budget rather than an unconstrained optimum."*

### Task 6: Hardware-Specific Latency Claims
- **Audit Finding**: Unqualified claims such as "Attention is 50% faster than InceptionTime" are hardware-dependent.
- **Corrective Action**: Replaced with exact hardware-qualified statements:
  > *"On the evaluation hardware and inference configuration used in this study, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime."* Labeled relative throughput strictly as *"higher measured throughput under the evaluated configuration."*

### Task 7: External Validation Limitation
- **Audit Finding**: Single-cohort PTB-XL evaluation cannot claim multi-hospital generalization.
- **Corrective Action**: Added explicit limitation:
  > *"The study uses a single public ECG cohort and does not provide external validation across independent institutions, acquisition devices, or populations."*

### Task 8: Random Seed & Uncertainty Characterization
- **Audit Finding**: Paired bootstrap tests sample variability across test ECGs, not across retraining seeds.
- **Corrective Action**: Added explicit limitation:
  > *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining across different random initializations."*

### Task 9: Metric Taxonomy & Threshold Governance
- **Audit Finding**: Critical to prevent conflation of ranking metrics with threshold-dependent metrics.
- **Corrective Action**:
  - Macro AUROC and Macro AP are designated exclusively as *continuous ranking/discrimination metrics*.
  - F1, precision, recall, subset accuracy, and Hamming loss are designated as *threshold-dependent operational metrics*.
  - Prohibited any claim that threshold optimization alters AUROC or AP. Confirmed that thresholds were derived purely on Fold 9 validation data and applied to frozen Fold 10.

### Task 10: Attention vs. InceptionTime Indistinguishability
- **Audit Finding**: Terms like "equivalent" or "identical" overstate negative findings.
- **Corrective Action**: Replaced with:
  > *"ECGResNet-Attention and InceptionTime exhibited statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol."*

### Task 11: Attention vs. GAP Controlled Contrast
- **Audit Finding**: This is the principal controlled contrast.
- **Corrective Action**: Formulated strictly as:
  > *"Replacing uniform global average pooling with lightweight temporal attention pooling improved macro-AUROC, macro-AP, and macro-F1 under the controlled patient-isolated protocol."*

### Task 12: HYP Sample Definitions & Resolution of Discrepancy
- **Audit Finding & Discrepancy Resolution**:
  1. The authoritative on-disk empirical artifacts are:
     - `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`
     - `results/phase7/benchmark_fold10/hyp_analysis/HYP_ERROR_ANALYSIS.md`
  2. Authoritative Frozen Metrics:
     - **Total Fold-10 HYP positive cases**: Exactly **262** records ($12.14\%$ prevalence).
     - **Pure Isolated HYP (Definition: HYP present AND STTC absent AND CD absent AND MI absent)**: Exactly **56** records ($21.37\%$ of HYP).
       - Model A (GAP) Sensitivity: **1.79% (1 / 56 detected)**; FN rate = 98.21%; mean probability = 0.0685.
       - InceptionTime Sensitivity: **7.14% (4 / 56 detected)**; FN rate = 92.86%; mean probability = 0.1000.
     - **HYP + STTC (Co-occurring with repolarization abnormality)**: Exactly **155** records ($59.16\%$ of HYP).
       - Model A (GAP) Sensitivity: **69.03% (107 / 155 detected)**; mean probability = 0.3331.
       - InceptionTime Sensitivity: **63.87% (99 / 155 detected)**; mean probability = 0.3347.
     - **HYP + CD**: Exactly **75** records (Model A sensitivity: 42.67%).
     - **HYP + MI**: Exactly **79** records (Model A sensitivity: 54.43%).
     - **HYP + $\ge 2$ Pathologies**: Exactly **89** records (Model A sensitivity: 66.29%).
  3. **Resolution**: All references to ungrounded draft numbers ($N = 118$, $N = 144$, $13.6\%$, $68.1\%$, `hyp_subphenotype_metrics.csv`) have been completely purged from all active manuscript planning materials. The historical audit trail documents that these figures were preliminary draft artifacts that are now formally superseded and invalid. The authoritative $N=56$ (pure isolated) and $N=155$ (HYP+STTC) stratification is now universally sealed across all documents.

### Task 13: Central Hypothesis Replacement
- **Corrective Action**: Replaced the working hypothesis with the exact approved formulation:
  > *"We hypothesize that replacing uniform temporal aggregation with lightweight learned temporal attention improves multi-label ECG discrimination under a controlled patient-isolated PTB-XL protocol, and that this improvement can approach the performance of a multi-scale architecture without requiring a substantial increase in parameter count."*

### Task 14: Primary Contributions Realignment
- **Corrective Action**: Restructured the 5 contributions to emphasize empirical benchmarking, paired statistical testing, Pareto efficiency, subphenotype error analysis, and reproducible calibration/occlusion profiling without biological decision claims.

### Task 15: Claim Matrix Realignment
- **Corrective Action**: Updated `results/phase10/paper_blueprint/PAPER_CLAIM_MATRIX.md` and created `CORRECTED_CLAIM_MATRIX.md`, strictly categorizing all claims into SUPPORTED, SUPPORTED WITH QUALIFICATION, and NOT SUPPORTED.

### Task 16: Numerical Traceability Verification
- **Corrective Action**: Verified that every reported number points to `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` and underlying authoritative result files, with the HYP definition discrepancy fully resolved to `hyp_stratified_sensitivity.csv`.

---

## 3. Discrepancy Resolution Table: Hypertrophy Subgroup Counts

| Stratum / Definition | Previous Draft Mention (Superseded & Invalid) | Authoritative Frozen Artifact (`hyp_stratified_sensitivity.csv`) | Final Resolution Status |
|:---|:---:|:---:|:---:|
| **Total Test HYP Cases** | $N = 262$ | $N = 262$ | **SEALED** ($262 / 2,158 = 12.14\%$) |
| **Pure Isolated HYP** (HYP=1, STTC=0, CD=0, MI=0) | $N = 118$ (13.6% recall) *(Superseded)* | **$N = 56$ (1.79% recall in GAP, 7.14% in Inception)** | **RESOLVED & SEALED**: Authoritative definition is HYP with zero other superclasses. |
| **HYP + STTC Co-occurrence** | $N = 144$ (68.1% recall) *(Superseded)* | **$N = 155$ (69.03% recall in GAP, 63.87% in Inception)** | **RESOLVED & SEALED**: Authoritative count from ground-truth co-occurrence matrix is 155 ($59.16\%$). |
| **HYP + CD Co-occurrence** | Not reported | **$N = 75$ (42.67% recall)** | **RESOLVED & SEALED** |
| **HYP + MI Co-occurrence** | Not reported | **$N = 79$ (54.43% recall)** | **RESOLVED & SEALED** |
| **HYP + $\ge 2$ Superclasses**| Not reported | **$N = 89$ (66.29% recall)** | **RESOLVED & SEALED** |

---

## 4. Governance Deliverables in `results/phase10/claim_correction/`

1. [`CLAIM_CORRECTION_REPORT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/claim_correction/CLAIM_CORRECTION_REPORT.md) — Master claim correction and discrepancy audit report (Updated to APPROVED).
2. [`CORRECTED_CLAIM_MATRIX.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/claim_correction/CORRECTED_CLAIM_MATRIX.md) — Exhaustive classification of claims with strict empirical guardrails.
3. [`MANUSCRIPT_LANGUAGE_GUARDRAILS.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/claim_correction/MANUSCRIPT_LANGUAGE_GUARDRAILS.md) — Mandatory and prohibited scientific phrasing standards for manuscript drafting.
4. [`REVIEWER_RISK_REGISTER.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/claim_correction/REVIEWER_RISK_REGISTER.md) — Defensibility protocols for the 10 most critical reviewer challenges.

---

## 5. Final Gate

```
================================================================================
SCIENCE CHANGES:
NONE

EXPERIMENTS RUN:
NONE

CHECKPOINTS MODIFIED:
NONE

FROZEN RESULTS MODIFIED:
NONE

CLAIM CORRECTIONS:
16 / 16 Tasks Executed and Applied Across All Governance Files

BLOCKING ISSUES:
0

STATUS:
CLAIM CORRECTION APPROVED
================================================================================
```
*(All active manuscript planning claims now use exclusively the authoritative HYP definitions and numbers from `hyp_stratified_sensitivity.csv`. The project is formally approved and sealed).*

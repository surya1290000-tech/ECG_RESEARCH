# Manuscript Language Guardrails & Scientific Terminology Protocol

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  
**Enforcement Level**: Strict / Zero Tolerance for Unjustified or Hyperbolic Claims  

This document codifies the mandatory scientific terminology, required phrasing formulas, prohibited phrases, and empirical boundaries for the preparation of the peer-reviewed manuscript.

---

## 1. Central Hypothesis Formulation

### Mandatory Formulation:
> *"We hypothesize that replacing uniform temporal aggregation with lightweight learned temporal attention improves multi-label ECG discrimination under a controlled patient-isolated PTB-XL protocol, and that this improvement can approach the performance of a multi-scale architecture without requiring a substantial increase in parameter count."*

### Prohibited Formulations:
- ❌ *"We hypothesize that temporal attention pooling discovers the true clinical features used by cardiologists."*
- ❌ *"We hypothesize that attention pooling provides state-of-the-art diagnostic accuracy superior to all existing models."*

---

## 2. Core Scientific Contributions (5 Mandatory Formulations)

The manuscript must frame its primary contributions strictly as:
1. **Controlled Patient-Isolated Four-Model Benchmark**: A rigorous benchmark of four principal architectures (ECGResNet-GAP, ECGResNet-Attention, XResNet1D, InceptionTime1D) with closely matched parameter budgets (~3.9M parameters) under the official PTB-XL Fold-10 split with verified zero patient leakage.
2. **Statistical Verification of Attention vs. GAP**: Paired bootstrap evaluation across 1,000 resamples showing that replacing uniform global average pooling with lightweight temporal attention pooling (+513 parameters) yields statistically significant improvements in Macro AUROC, Macro AP, and Macro F1 ($P \ge 99.6\%$).
3. **Common-Protocol Comparison of Attention and Multi-Scale Architectures**: Demonstrating that ECGResNet-Attention and InceptionTime1D achieve statistically indistinguishable discrimination under this common protocol, while documenting higher measured throughput for the attention model on the evaluated hardware.
4. **Subphenotype Analysis of Hypertrophy (HYP)**: Error and stratified sensitivity analysis demonstrating that model discrimination is non-uniform across clinical presentations, exhibiting limited sensitivity to isolated hypertrophy and stronger recognition when hypertrophy co-occurs with repolarization abnormalities.
5. **Reproducible Calibration, Threshold, and Sensitivity Profiling**: Comprehensive assessment showing raw probabilities are well-calibrated (ECE 2.69%), validation-derived thresholds generalize reliably to the frozen test set, and lead-masking perturbation reveals precordial sensitivity while highlighting the confounding impact of BatchNorm distribution shifts.

### Prohibited Contribution Claims:
- ❌ *"We propose a fundamentally novel attention architecture / new attention mechanism."*
- ❌ *"Our experiments establish the biological or clinical decision mechanisms of deep neural networks."*
- ❌ *"We demonstrate clinical readiness or external diagnostic equivalence to expert cardiologists."*

---

## 3. Pathology & Representation Analysis (HYP Specifics)

### Mandatory Phrasing Formulas:
- ✅ *"The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities."*
- ✅ *"These results suggest phenotype-dependent representation sensitivity, but do not establish a causal decision mechanism."*
- ✅ *"Consistent with limited sensitivity to multi-lead spatial amplitude relationships."*

### Prohibited Phrasing (Zero Tolerance):
- ❌ Do NOT claim the experiments PROVE that models:
  - *"rely on ST-T abnormalities as shortcuts"*
  - *"fail to learn QRS voltage criteria"*
  - *"use a specific clinical diagnostic criterion"*
  - *"use ST-T information causally"*
  - *"calculate or fail to calculate Sokolow-Lyon, Cornell, or any specific clinical criterion"*

---

## 4. Lead Perturbation & Occlusion Language

### Mandatory Terminology:
- ✅ **Approved terms**: *"sensitivity probing"*, *"representation sensitivity analysis"*, *"lead perturbation analysis"*, *"probing sensitivity bounds"*.
- ✅ **Mandatory caveat**: *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."*
- ✅ Explicitly document that channel zero-masking shifts early BatchNorm running statistics ($\mu, \sigma$), which can induce global logit inflation.

### Prohibited Terminology (Zero Tolerance):
- ❌ *"causal attribution"*
- ❌ *"causal explanation"*
- ❌ *"counterfactual clinical reasoning"*
- ❌ *"proof of model decision mechanism"*
- ❌ *"biological lead ablation"*

---

## 5. Architectural Novelty & Attention Positioning

### Mandatory Terminology:
- ✅ **Approved terms**: *"lightweight temporal attention pooling"*, *"controlled evaluation of temporal attention pooling within a compact 1D residual ECG classifier"*, *"dynamic temporal aggregation"*.
- ✅ Emphasize that the contribution lies in the *controlled empirical evaluation and ablation*, not the invention of the attention concept.

### Prohibited Terminology:
- ❌ *"novel attention architecture"*
- ❌ *"new attention mechanism"*
- ❌ *"pioneering attention framework"*

---

## 6. SOTA & Competitive Positioning

### Mandatory Terminology:
- ✅ **Approved terms**: *"controlled architectural benchmark"*, *"patient-isolated evaluation"*, *"under the evaluated protocol"*, *"competitive discrimination within the evaluated model family"*, *"the highest-performing model among the evaluated project configurations"*.
- ✅ Explicitly state: *"Recent large-scale pretrained or foundation models for electrocardiography may report higher nominal performance when evaluated under different training regimes, larger datasets, or alternative evaluation protocols."*

### Prohibited Terminology:
- ❌ *"state-of-the-art"* or *"SOTA"*
- ❌ *"best-performing PTB-XL model"*
- ❌ *"superior to all published approaches"*
- ❌ *"clinical superiority over cardiologists"*

---

## 7. Model-to-Model Comparison Language

### Attention vs. InceptionTime:
- ✅ Mandatory phrasing: *"ECGResNet-Attention and InceptionTime exhibited statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol."*
- ❌ Prohibited terms: *"equivalent"*, *"same performance"*, *"identical"*, *"mechanistically equivalent"*.

### Attention vs. GAP:
- ✅ Mandatory phrasing: *"Replacing uniform global average pooling with lightweight temporal attention pooling improved macro-AUROC, macro-AP, and macro-F1 under the controlled patient-isolated protocol."*
- ❌ Prohibited terms: *"proves biological feature detection"*, *"solves transient feature loss"*.

---

## 8. Latency, Throughput & Hardware Claims

### Mandatory Phrasing:
- ✅ *"On the evaluation hardware and inference configuration used in this study, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime."*
- ✅ *"Higher measured throughput under the evaluated configuration."*

### Prohibited Phrasing:
- ❌ *"Attention is 50% faster than InceptionTime"* (unqualified / hardware-independent).
- ❌ Claims of general real-time embedded or mobile suitability without embedded hardware validation.

---

## 9. Methodological Limitations Required in Text

The manuscript must include explicit statements for each of the following:
1. **InceptionTime Training Budget**:
   - *"InceptionTime training was terminated after seven epochs because of sustained computational/thermal constraints, with the best validation checkpoint observed before termination retained for the frozen test evaluation. The comparison therefore reflects the specified training budget rather than an asymptotic optimal performance."*
2. **External Validation**:
   - *"The study uses a single public ECG cohort and does not provide external validation across independent institutions, acquisition devices, or populations."*
3. **Random Seed & Uncertainty Characterization**:
   - *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining."*
4. **Metric Distinction & Threshold Governance**:
   - AUROC and AP must be consistently designated as *continuous ranking/discrimination metrics*.
   - F1, precision, recall, subset accuracy, and Hamming loss must be designated as *threshold-dependent operational metrics*.
   - It must be explicitly stated that decision thresholds were optimized strictly on Fold 9 validation data and applied to frozen Fold 10 without tuning on test labels.
   - The text must never claim that threshold optimization alters or improves continuous ranking metrics (AUROC or AP).

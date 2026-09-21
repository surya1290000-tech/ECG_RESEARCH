# Table 4: Hypertrophy (HYP) Diagnostic Stratification and Error Analysis

**Authoritative Source**: `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv` and `HYP_ERROR_ANALYSIS.md`  
**Dataset & Partition**: PTB-XL v1.0.3, Frozen Fold-10 Test Set ($N = 2,158$)  
**Total HYP Ground-Truth Positive Cases**: $N = 262$ (12.14% prevalence)  

---

### Table 4A: Stratified Sensitivity and Error Rates Across Clinical Co-occurrence Subgroups

| Clinical Co-occurring Stratum | Stratum Definition / Inclusion Criteria | Cohort Support ($N$) | Proportion of HYP (%) | Model A (GAP) Sensitivity (%) [TP / $N$] | Model A False Neg. Rate (%) | Model A Mean Prob. | InceptionTime Sensitivity (%) [TP / $N$] | InceptionTime False Neg. Rate (%) | InceptionTime Mean Prob. | Diagnostic Representation Finding |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Pure Isolated HYP** | $\text{HYP}=1 \land \text{STTC}=0 \land \text{CD}=0 \land \text{MI}=0$ | **56** | 21.37% | **1.79%** [1 / 56] | **98.21%** | 0.0685 | **7.14%** [4 / 56] | **92.86%** | 0.1000 | Limited representation sensitivity; predictions suppress below cutoff |
| **HYP + STTC** | $\text{HYP}=1 \land \text{STTC}=1$ | **155** | 59.16% | **69.03%** [107 / 155] | 30.97% | 0.3331 | **63.87%** [99 / 155] | 36.13% | 0.3347 | Substantially higher recognition when co-occurring with repolarization changes |
| **HYP + CD** | $\text{HYP}=1 \land \text{CD}=1$ | **75** | 28.63% | 42.67% [32 / 75] | 57.33% | 0.2318 | 34.67% [26 / 75] | 65.33% | 0.2380 | Moderate sensitivity with concurrent conduction abnormalities |
| **HYP + MI** | $\text{HYP}=1 \land \text{MI}=1$ | **79** | 30.15% | 54.43% [43 / 79] | 45.57% | 0.2845 | 56.96% [45 / 79] | 43.04% | 0.2858 | Moderate sensitivity with concurrent infarct statements |
| **HYP + $\ge 2$ Pathologies** | Co-occurring with $\ge 2$ other superclasses | **89** | 33.97% | 66.29% [59 / 89] | 33.71% | 0.3224 | 61.80% [55 / 89] | 38.20% | 0.3176 | High recognition under multi-pathology complexity |
| **All HYP Cohort** | Overall ground-truth positive HYP records | **262** | 100.0% | 43.51% [114 / 262] | 56.49% | 0.2345 | 41.22% [108 / 262] | 58.78% | 0.2460 | Overall aggregate performance masks subphenotypic sensitivity gap |

---

### Table 4B: Spatial Lead Perturbation Sensitivity Profile on HYP

| Lead Perturbation Condition | Masked Input Channels | ECGResNet-GAP HYP AUROC | InceptionTime1D HYP AUROC | Absolute $\Delta\text{AUROC}$ (GAP) | Relative Impact Ranking | Methodological Interpretation |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Unmasked)** | None (All 12 Leads Intact) | 0.7682 | 0.7899 | 0.0000 | Reference Baseline | Full 12-lead baseline discrimination |
| **Limb Leads Masked** | I, II, III, aVR, aVL, aVF | 0.7406 | 0.7213 | −0.0276 | Secondary Impact | Moderate sensitivity to frontal plane voltages |
| **Precordial Leads Masked** | $V_1, V_2, V_3, V_4, V_5, V_6$ | **0.7135** | **0.7041** | **−0.0547** | **Primary Dominant Driver** | Substantial degradation under precordial perturbation (nearly 2.0× larger AUROC drop; AP drop: −0.0975 vs −0.0372) |

*Methodological Guardrails & Scientific Constraints*:
1. **Conservative Scientific Phrasing**: The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities. These results suggest phenotype-dependent representation sensitivity, but do not establish a causal internal decision mechanism.
2. **Clinical Rules Constraint**: The manuscript does NOT claim that models explicitly calculate or fail to calculate Sokolow-Lyon, Cornell, or any specific clinical diagnostic criteria.
3. **Perturbation Constraint**: Lead masking is interpreted as a sensitivity probe rather than a causal attribution method because channel removal introduces distributional shifts that interact with BatchNorm running statistics (e.g., Lead II zero-masking induces +0.0412 global probability drift).
4. **Historical Discrepancy Clearance**: Preliminary draft values of $N=118$ (isolated), $N=144$ (composite), 13.6% (isolated recall), and 68.1% (composite recall) originated from an ungrounded non-existent draft artifact (`hyp_subphenotype_metrics.csv`) and are formally superseded and invalid. The values in Table 4 derive exclusively from the authoritative frozen file `hyp_stratified_sensitivity.csv`.

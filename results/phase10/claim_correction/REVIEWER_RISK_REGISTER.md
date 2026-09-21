# Reviewer Risk Register & Methodological Defensibility Plan

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  

This register catalogues the 10 most critical methodological, architectural, and interpretation risks that peer reviewers may raise, assigns an objective severity rating, assesses the current empirical evidence, outlines the mandatory manuscript rebuttal strategy, and explicitly specifies whether an experiment is required.

---

## Risk Summary Matrix

| Risk ID | Methodological / Reviewer Risk | Severity | Why Reviewer May Raise It | Current Available Evidence | Required Manuscript Response / Strategic Framing | New Experiment Necessary? |
|:---:|:---|:---:|:---|:---|:---|:---:|
| **RSK-01** | **Limited Architectural Novelty** | **MODERATE** | Temporal attention pooling is a known technique in NLP/speech; reviewers may claim the study lacks algorithmic novelty. | Extensive paired statistical benchmark (1,000 resamples) quantifying exact empirical gains over GAP in a controlled 1D-ResNet. | Position the paper as a *controlled empirical benchmark and ablation study*, not an architecture invention. Emphasize that isolating the contribution of pooling with zero backbone divergence is precisely what prior literature lacked. | **NO** |
| **RSK-02** | **Lack of Global SOTA Performance** | **MODERATE** | Foundation models (e.g., pretrained on millions of ECGs) or large ensembles report nominal AUROCs > 0.92 on PTB-XL. | Matched parameter budget (~3.9M params) across 4 standard models trained strictly from scratch under official patient-isolated Fold-10. | Acknowledge pretrained/foundation models explicitly. Emphasize that the study's scope is evaluating inductive biases and temporal aggregation under controlled from-scratch conditions, where fair apples-to-apples comparison requires identical training budgets. | **NO** |
| **RSK-03** | **InceptionTime Seven-Epoch Limitation** | **HIGH** | InceptionTime was stopped at epoch 7 due to thermal/compute constraints; reviewers may argue it was undertrained compared to 20-epoch ResNets. | InceptionTime converged rapidly by epoch 5–7 (Val AUROC 0.8967), achieving the highest nominal test AUROC (0.8991) among all models. | Transparently disclose the 7-epoch stopping criterion due to sustained computational/thermal constraints. Note that the best validation checkpoint was evaluated and that the comparison reflects this specific computational budget rather than an unconstrained optimum. | **NO** |
| **RSK-04** | **Single Training Seed (Lack of Multi-Seed Runs)** | **MODERATE** | The main benchmark models were trained using a single fixed seed (`SEED = 42`); reviewers may ask whether gains persist across retraining seeds. | High statistical significance of Attention vs. GAP ($P = 100\%$, CI strictly positive) across 1,000 test-set bootstrap resamples. | Add an explicit limitation: *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining."* Present bootstrap as evaluating sample uncertainty over patient populations. | **NO** |
| **RSK-05** | **Lack of External Cohort Validation** | **HIGH** | Models evaluated solely on PTB-XL; reviewers frequently demand cross-dataset validation (e.g., CPSC 2018 or Chapman). | Standardized official benchmark split (`strat_fold`) with zero patient leakage provides rigorous internal generalization. | Explicitly state as a limitation: *"The study uses a single public ECG cohort and does not provide external validation across independent institutions, acquisition devices, or populations."* Clarify that the goal is controlled architectural comparison rather than clinical product validation. | **NO** |
| **RSK-06** | **HYP Causal Overinterpretation** | **HIGH** | Reviewers with clinical expertise will push back against claims that models "rely on ST-T shortcuts" or "fail to calculate Sokolow-Lyon". | Stratified sensitivity data (`hyp_stratified_sensitivity.csv`) shows 1.79% recall on isolated HYP vs. 69.03% on HYP+STTC. | Strictly eliminate causal claims. Adopt the approved phrasing: *"The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities, but do not establish a causal decision mechanism."* | **NO** |
| **RSK-07** | **Occlusion Masking Induces Distribution Shift** | **MODERATE** | Zero-masking entire channels creates out-of-distribution inputs that disrupt BatchNorm running statistics, confounding sensitivity analysis. | Single-lead ablation (`hyp_single_lead_summary.csv`) quantitatively measures this drift (+0.0412 mean probability drift on Lead II). | Proactively present this caveat as a methodological insight: *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."* This transforms a potential reviewer critique into a rigorous scientific strength. | **NO** |
| **RSK-08** | **Hardware-Specific Latency & Throughput** | **LOW** | "50% faster" claims can be challenged as implementation- or hardware-dependent. | Detailed CPU profiling table recording latency (16.0 ms vs. 24.1 ms) and batch size (64) under PyTorch 2.x on x86_64. | Report absolute figures with exact hardware/software details: *"On the evaluation hardware and inference configuration used in this study, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime."* | **NO** |
| **RSK-09** | **Threshold Overfitting / Leakage Concern** | **LOW** | In multi-label classification, threshold tuning on the test set is a common source of data leakage. | Decision thresholds were optimized purely on Fold 9 validation data and frozen before test inference; bootstrap validation confirms stability (modal HYP threshold 0.25 in 34% of runs). | Detail the strict separation in the methods: Fold 9 was used for grid search; Fold 10 was strictly frozen. Clearly distinguish continuous ranking metrics (AUROC/AP) from threshold-dependent metrics (F1/subset accuracy). | **NO** |
| **RSK-10** | **Multi-Label Class Imbalance Handling** | **LOW** | HYP has only 12.1% prevalence, leading to low F1 scores across all models. | Full evaluation reporting both prevalence-insensitive AUROC and prevalence-sensitive Average Precision (AP), alongside unweighted and weighted F1. | Note that BCE loss with equal positive weighting was used across all models to maintain comparability. Highlight that Attention's largest gain occurred on the minority class (HYP AP $+0.0563$, $P = 99.9\%$). | **NO** |

---

## Detailed Defense Protocols by Severity

### A. High-Severity Reviewer Risks

#### 1. Risk RSK-03: InceptionTime Seven-Epoch Training Budget
- **The Attack**: *"The authors claim InceptionTime and Attention are statistically indistinguishable, but InceptionTime was only trained for 7 epochs whereas ResNets were trained for 20 epochs. InceptionTime was undertrained, invalidating the equivalence claim."*
- **The Defensible Manuscript Response**:
  1. We state clearly in Section 7 (Experimental Protocol) and Section 17 (Limitations) that InceptionTime training was terminated after 7 epochs due to sustained computational and thermal constraints in the evaluation environment.
  2. We document that InceptionTime exhibited rapid empirical convergence on validation Fold 9 (best validation loss and AUROC attained by epoch 5, with epoch 7 AUROC plateauing at 0.8967).
  3. We state: *"Our comparison evaluates whether adding a lightweight attention pooling layer to a standard ResNet can match the discriminative capacity of a multi-scale convolutional network under the specified computational budget, without implying that seven epochs represents InceptionTime's asymptotic optimum under unconstrained compute."*
  4. Crucially, InceptionTime *already achieved the highest nominal test AUROC (0.8991)*; undertraining would bias against InceptionTime, yet it still matched Attention.

#### 2. Risk RSK-05: Lack of External Cohort Validation
- **The Attack**: *"The authors report results exclusively on PTB-XL. Without evaluating on an external dataset like CPSC-2018 or Georgia/Chapman, the clinical generalizability of these findings is unproven."*
- **The Defensible Manuscript Response**:
  1. Concede immediately that external validation is not presented and belongs in future translational work.
  2. Frame the paper's contribution as a *methodological and representation benchmark* rather than a clinical deployment study.
  3. Point out that cross-dataset evaluation in multi-label ECG often introduces confounding label taxonomy mismatches (e.g., different definitions of rhythm vs. morphology statements), whereas PTB-XL v1.0.3 provides a standardized, peer-reviewed clinical label ontology.

#### 3. Risk RSK-06: Clinical Causal Claims Regarding Hypertrophy
- **The Attack**: *"The claim that the model relies on 'repolarization shortcuts' or 'cannot compute Sokolow-Lyon criteria' is speculative. The authors did not dissect the neural network weights to prove what features are used."*
- **The Defensible Manuscript Response**:
  1. Ensure zero causal attribution language remains in the text.
  2. Frame the finding as *behavioral phenotype sensitivity*:
     *"When evaluated on the 262 true positive hypertrophy cases in Fold 10, the models exhibit an operational sensitivity of 1.79%–7.14% on cases presenting without co-occurring diagnoses ($N = 56$), compared to 63.87%–69.03% on cases presenting with co-occurring ST/T changes ($N = 155$). These empirical findings are consistent with limited representation sensitivity to isolated hypertrophy patterns, but do not establish a causal mechanistic decision rule."*

---

### B. Moderate-Severity Reviewer Risks

#### 1. Risk RSK-01: Limited Architectural Novelty
- **The Attack**: *"Attention pooling is an existing technique. The paper simply applies a standard attention formula to an existing ECG dataset. What is the novel research contribution?"*
- **The Defensible Manuscript Response**:
  1. Do not claim algorithmic novelty for the attention formula.
  2. Emphasize that the biomedical ECG literature is saturated with studies introducing complex architectures without controlled ablations, where improvements are conflated with parameter scaling or contaminated test splits.
  3. The novel scientific contribution is:
     - Isolating the specific impact of temporal pooling by holding the 1D-ResNet backbone strictly constant (+513 parameters / +0.013% capacity).
     - Proving via paired bootstrap ($B=1,000$) that dynamic pooling alone accounts for +0.0110 AUROC and +0.0310 AP.
     - Uncovering the subphenotype sensitivity collapse in hypertrophy, which standard aggregate metrics completely mask.

#### 2. Risk RSK-04: Single Training Seed
- **The Attack**: *"The authors report 95% bootstrap confidence intervals, but these are test-sample bootstrap intervals, not multi-run training intervals. How do we know the +0.0110 AUROC gain is not just a lucky seed?"*
- **The Defensible Manuscript Response**:
  1. Clearly explain in Section 10 (Statistical Analysis) that paired bootstrap resampling over the frozen test set evaluates uncertainty across patients/records under a fixed trained model state.
  2. Transparently state in Section 17 (Limitations): *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining across different random initializations."*
  3. Emphasize that the empirical probability $P(\text{Attention} > \text{GAP}) = 100.0\%$ over 1,000 resamples provides overwhelming evidence that the performance difference is robust across patient subpopulations in the test cohort.

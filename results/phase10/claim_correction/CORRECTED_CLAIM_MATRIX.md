# Corrected Scientific Claim Matrix & Evidence Classification

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  

This matrix supersedes all prior claim drafts. Every major claim proposed for inclusion in the manuscript is categorized as **SUPPORTED**, **SUPPORTED WITH QUALIFICATION**, or **NOT SUPPORTED**, with explicit empirical rationale, supporting frozen artifacts, and mandatory phrasing guardrails.

---

## 1. Architectural & Pooling Claims

| Claim ID | Proposition | Formal Classification | Empirical Rationale & Artifact | Mandatory Phrasing & Restrictions |
|:---|:---|:---:|:---|:---|
| **CLM-01** | Replacing uniform Global Average Pooling (GAP) with lightweight temporal attention pooling improves multi-label ECG classification. | **SUPPORTED** | 1,000 paired bootstrap resamples on Fold-10 test records ($N=2,158$) show statistically significant gains across all primary metrics: $\Delta\text{AUROC} = +0.0110$ (95% CI $[+0.0070, +0.0152]$, $P = 100\%$), $\Delta\text{AP} = +0.0310$ ($P = 100\%$), and $\Delta\text{Macro F1} = +0.0144$ ($P = 99.6\%$). Source: `four_model_paired_bootstrap.csv`. | Approved for affirmative statement under the evaluated protocol. Do not claim this proves an internal biological or clinical decision mechanism. |
| **CLM-02** | Temporal attention pooling is a fundamentally novel architecture or new attention mechanism. | **NOT SUPPORTED** | The attention pooling layer is a standard formulation ($\text{Conv1d}(512 \to 1) + \text{Softmax} + \text{Sum}$) adapted from general time-series literature. The scientific contribution is the controlled empirical evaluation within a compact 1D ECG ResNet, not algorithmic invention. | **STRICTLY PROHIBITED**. Must describe as *"lightweight temporal attention pooling"* within a controlled comparative evaluation. |
| **CLM-03** | ECGResNet-Attention is statistically indistinguishable from InceptionTime1D under the evaluated protocol. | **SUPPORTED WITH QUALIFICATION** | Paired bootstrap differences span zero across all primary metrics: $\Delta\text{AUROC} = -0.0012$ (95% CI $[-0.0055, +0.0038]$, $P = 31.5\%$), $\Delta\text{AP} = +0.0008$ ($[-0.0087, +0.0108]$, $P = 59.5\%$), and $\Delta\text{Macro F1} = -0.0019$ ($[-0.0134, +0.0099]$, $P = 38.4\%$). Source: `four_model_paired_bootstrap.csv`. | Must state: *"ECGResNet-Attention and InceptionTime exhibited statistically indistinguishable macro-AUROC, AP, and macro-F1 under the evaluated paired bootstrap protocol."* Do NOT claim they are "equivalent" or "mechanistically identical". |
| **CLM-04** | The tested models establish state-of-the-art (SOTA) performance on PTB-XL. | **NOT SUPPORTED** | While InceptionTime (AUROC 0.8991) and Attention (AUROC 0.8979) achieved the highest numbers among our four evaluated architectures, external foundation/pretrained models trained on broader data or under different regimes report higher nominal scores. | **STRICTLY PROHIBITED**. Must frame as a *"controlled architectural benchmark"* and *"competitive discrimination within the evaluated model family"*. |
| **CLM-05** | ECGResNet-Attention has a 50% inference speed advantage over InceptionTime across all computing platforms. | **NOT SUPPORTED** | Throughput was measured on a single CPU testbed (x86_64, PyTorch 2.x, 64-batch size), yielding 62.6 ECGs/s vs. 41.6 ECGs/s. Performance will vary across GPUs, Apple Silicon, ARM, and embedded edge devices. | **STRICTLY PROHIBITED**. Must state: *"On the evaluation hardware and inference configuration used in this study, ECGResNet-Attention achieved 62.6 ECG/s (16.0 ms/ECG), compared with 41.6 ECG/s (24.1 ms/ECG) for InceptionTime."* |
| **CLM-06** | InceptionTime training for 7 epochs represents its optimal achievable performance. | **NOT SUPPORTED** | InceptionTime training was terminated after 7 epochs due to sustained computational/thermal constraints. The checkpoint retained was the best validation checkpoint observed prior to termination. | **STRICTLY PROHIBITED**. Must explicitly state the 7-epoch training budget as a methodological constraint. |

---

## 2. Hypertrophy (HYP) Subphenotype & Representation Claims

| Claim ID | Proposition | Formal Classification | Empirical Rationale & Artifact | Mandatory Phrasing & Restrictions |
|:---|:---|:---:|:---|:---|
| **CLM-07** | Models exhibit limited representation sensitivity to isolated hypertrophy phenotypes compared to hypertrophy co-occurring with repolarization abnormalities. | **SUPPORTED WITH QUALIFICATION** | On frozen Fold-10 ($N=2,158$), true HYP records ($N=262$) evaluated under frozen validation thresholds show that isolated HYP (no STTC, no CD, no MI, $N=56$) yields only 1.79% sensitivity in Model A (1/56) and 7.14% in InceptionTime (4/56), whereas composite HYP + STTC ($N=155$) yields 69.03% (107/155) in Model A and 63.87% (99/155) in InceptionTime. Source: `hyp_stratified_sensitivity.csv`. | Must state: *"The findings are consistent with limited sensitivity to isolated hypertrophy phenotypes and stronger recognition when hypertrophy co-occurs with repolarization abnormalities."* Must NOT claim this proves a causal decision mechanism. |
| **CLM-08** | Deep learning models rely on ST-T abnormalities as causal shortcuts and fail to learn QRS voltage criteria (e.g., Sokolow-Lyon). | **NOT SUPPORTED** | While behavioral predictions correlate strongly with co-occurring STTC, our experiments did not perform internal mechanistic circuit dissection of QRS voltage summation, nor did they prove causal reliance. | **STRICTLY PROHIBITED**. Must NOT claim the models "use shortcuts causally", "fail to calculate Sokolow-Lyon", or "use specific clinical criteria". Use *"phenotype-dependent representation sensitivity"* instead. |

---

## 3. Spatial Lead Perturbation & Occlusion Claims

| Claim ID | Proposition | Formal Classification | Empirical Rationale & Artifact | Mandatory Phrasing & Restrictions |
|:---|:---|:---:|:---|:---|
| **CLM-09** | Lead occlusion analysis provides causal attribution and proves which leads the model uses for clinical reasoning. | **NOT SUPPORTED** | Lead perturbation analysis via zero-masking shifts input channel distributions and alters early BatchNorm activation statistics ($\mu, \sigma$), as shown by single-lead ablation inducing up to +0.0412 mean probability drift on Lead II. Source: `hyp_single_lead_summary.csv`. | **STRICTLY PROHIBITED**. Must state: *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."* |
| **CLM-10** | Precordial lead perturbation induces substantial degradation in HYP discrimination. | **SUPPORTED WITH QUALIFICATION** | Precordial zero-masking ($V_1$–$V_6$) degrades HYP AUROC from 0.7682 to 0.5847 in Model A and 0.7899 to 0.5758 in InceptionTime, whereas limb lead masking drops AUROC to 0.7027 and 0.7303. Source: `hyp_occlusion_summary.csv`. | Report as a *perturbation sensitivity profile* demonstrating precordial sensitivity, while explicitly citing the BatchNorm shift caveat. |

---

## 4. Calibration, Thresholds & External Generalization Claims

| Claim ID | Proposition | Formal Classification | Empirical Rationale & Artifact | Mandatory Phrasing & Restrictions |
|:---|:---|:---:|:---|:---|
| **CLM-11** | Model probabilities are well-calibrated and validation-derived operational thresholds generalize to the test set. | **SUPPORTED** | Temperature scalar $T = 0.9761$ derived from Fold 9 validation NLL yielded test ECE of 2.69% and Brier score of 0.0969. Fold-9 tuned thresholds generalized effectively to Fold 10. Source: `calibration_test_results.json`. | Emphasize that Fold 10 was strictly unexposed to threshold selection. Reiterate that temperature scaling leaves continuous ranking metrics (AUROC/AP) mathematically invariant. |
| **CLM-12** | The trained models generalize to external clinical patient populations and diverse recording equipment. | **NOT SUPPORTED** | The study is strictly an internal patient-isolated benchmark on PTB-XL v1.0.3. No external validation across independent health systems, hardware vendors, or geographic cohorts was conducted. | **STRICTLY PROHIBITED**. Must include explicit limitation: *"The study uses a single public ECG cohort and does not provide external validation across independent institutions, acquisition devices, or populations."* |
| **CLM-13** | Paired bootstrap analysis captures uncertainty across model retraining seeds. | **NOT SUPPORTED** | Paired bootstrap was conducted over the 2,158 test records using fixed model checkpoints trained from a single seed (`SEED = 42`). It characterizes test-sample variability, not training stochasticity. | **STRICTLY PROHIBITED**. Must state: *"The paired bootstrap analysis characterizes uncertainty over the frozen test cohort and does not quantify variability arising from independent model retraining."* |

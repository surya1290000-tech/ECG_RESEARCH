# Scientific Paper Figure Plan (Corrected & Frozen)

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.2.2 Manuscript Claim Correction & Scientific Wording Freeze  
**Status**: Authoritative & Frozen  

This document details the complete publication figure suite (Figures 1 through 7), specifying the scientific motivation, visual layout, data source provenance, and rendering guidelines for Phase 10.3 production, synchronized with the authoritative frozen artifacts and conservative language guardrails.

---

## Figure 1: Overall Research Pipeline and Patient-Isolated Benchmark Governance

- **Scientific Purpose**: Provide a comprehensive visual roadmap of the entire experimental workflow, illustrating data ingestion, cohort filtering, patient-isolated splitting, model training, frozen evaluation, and rigorous statistical testing.
- **What It Should Show**:
  1. **Dataset Cohort Flow**: PTB-XL raw cohort ($N = 21,799$) $\to$ exclusion of 411 records lacking diagnostic superclasses $\to$ retained diagnostic cohort ($N = 21,388$).
  2. **Patient Isolation Protocol**: Official `strat_fold` partitioning into Training Folds 1–8 ($N = 17,084$; 14,823 patients), Validation Fold 9 ($N = 2,146$; 1,917 patients), and Frozen Test Fold 10 ($N = 2,158$; 1,877 patients), explicitly illustrating the mathematical constraint of zero patient intersection ($\text{Overlap} = 0$).
  3. **Preprocessing Block**: 12-lead ECG timeseries at 100 Hz $\to$ 2nd-order Butterworth bandpass (0.5–40 Hz) $\to$ per-lead Z-score normalization $\to$ tensor shape $(12, 1000)$.
  4. **Four-Model Parallel Execution**: Dispatching input tensors to ECGResNet-GAP, ECGResNet-Attention, XResNet1D, and InceptionTime1D.
  5. **Governance & Decision Firewall**: Clear dashed firewall indicating that Fold 10 predictions were generated once from frozen checkpoints without feedback loops, and operational thresholds were derived strictly from Fold 9.
  6. **Statistical Analysis Engine**: Paired bootstrap resampling ($B = 1,000$, $\text{seed} = 42$) generating empirical 95% confidence intervals and p-values over test-sample variability.
- **Why It Matters**: Establishes immediate scientific credibility by demonstrating rigorous avoidance of data leakage, transparent cohort governance, and reproducible pipeline architecture.
- **Exact Frozen Artifacts Supplying Data**:
  - `data/ptbxl/ptbxl_database.csv`
  - `configs/config.py`
  - `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md` (Pillars 1, 2, 3)
- **Visual Layout & Aesthetics**:
  - Multi-panel schematic flowchart (left-to-right or top-to-bottom).
  - Distinct palette: Soft slate blue for data flow, coral/crimson for the frozen test firewall, teal for model architectures.

---

## Figure 2: Neural Architectural Typology and Temporal Pooling Mechanisms

- **Scientific Purpose**: Contrast the internal structural mechanics of the four architectures, with a specific focus on the mathematical and structural difference between uniform Global Average Pooling (GAP) and lightweight Temporal Attention Pooling.
- **What It Should Show**:
  - **Panel A (ECGResNet Backbone & Pooling Comparison)**:
    - 1D Residual backbone common to Models A and B: Stem conv ($1\times 15, 64$), 4 residual stages (channels 64, 128, 256, 512).
    - Fork at temporal aggregation:
      - *Top branch (GAP)*: Uniform temporal averaging: $\frac{1}{T}\sum_{t=1}^T x_t$, collapsing temporal activations uniformly into a static 512-dim vector.
      - *Bottom branch (Attention Pooling)*: Lightweight 1D convolution ($1\times 1, 512 \to 1$) producing scalar temporal logits $s_t$, Softmax normalization producing attention weights $\alpha_t$, and weighted context aggregation $c = \sum \alpha_t x_t$.
      - Parameter callout: GAP = 3,919,493 vs. Attention = 3,920,006 (**+513 parameters / +0.013%**).
  - **Panel B (XResNet1D Modernizations)**:
    - ResNet tweaks: 3-stage convolutional stem ($3 \times \text{conv } 1\times 3$), stride-2 shifted to inner convolution, anti-aliased average pooling downsampling branch.
  - **Panel C (InceptionTime1D Multi-Scale Module)**:
    - Parallel multi-branch structure: Bottleneck $1\times 1$ conv ($512 \to 32$) feeding three concurrent 1D convolutions with kernel lengths $k=9, 19, 39$, plus a max-pooling branch ($k=3$), channel concatenation, and residual shortcut.
- **Why It Matters**: Clarifies the central theoretical contrast of the paper: dynamic temporal saliency (Attention) vs. uniform temporal collapse (GAP) vs. multi-frequency receptive field expansion (InceptionTime), emphasizing that Attention requires only 513 additional parameters.
- **Exact Frozen Artifacts Supplying Data**:
  - `src/models/ecg_resnet.py`
  - `src/models/ecg_resnet_attention.py`
  - `src/models/xresnet1d.py`
  - `src/models/inceptiontime1d.py`
  - `results/phase10/research_freeze/CHECKPOINT_HASHES.csv`
- **Visual Layout & Aesthetics**:
  - Clean modular block diagram with tensor dimensions annotated at each stage (e.g., $(B, 12, 1000) \to (B, 64, 500) \dots \to (B, 512, 63)$).

---

## Figure 3: Multi-Label ROC and Precision-Recall Performance Curves

- **Scientific Purpose**: Display the complete discriminative continuum across all operational thresholds for all four architectures, covering both overall macro performance and per-class diagnostic curves.
- **What It Should Show**:
  - **Panel A**: Macro-Averaged Receiver Operating Characteristic (ROC) curves for all 4 models:
    - InceptionTime (AUROC = 0.8991)
    - ECGResNet-Attention (AUROC = 0.8979)
    - ECGResNet-GAP (AUROC = 0.8868)
    - XResNet1D (AUROC = 0.8775)
  - **Panel B**: Macro-Averaged Precision-Recall (PR) curves:
    - InceptionTime (AP = 0.7636)
    - ECGResNet-Attention (AP = 0.7644)
    - ECGResNet-GAP (AP = 0.7334)
    - XResNet1D (AP = 0.7276)
  - **Panels C–G**: Class-specific ROC/PR curves for each diagnostic superclass (`NORM`, `STTC`, `CD`, `MI`, `HYP`), highlighting the wide performance envelope for `HYP` (GAP AP 0.3854 vs. Attention AP 0.4417).
- **Why It Matters**: Proves that the attention mechanism's advantage is sustained across continuous operational ranking thresholds, not merely at a single arbitrary cutoff.
- **Exact Frozen Artifacts Supplying Data**:
  - `results/phase7/benchmark_fold10/model_a/model_a_fold10_predictions.csv`
  - `results/phase9/model_b_fold10_evaluation/model_b_fold10_predictions.csv`
  - `results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv`
  - `results/phase7/benchmark_fold10/model_inception/model_inception_fold10_predictions.csv`
- **Visual Layout & Aesthetics**:
  - Multi-panel grid with high-contrast color palette: InceptionTime (Dark Blue), ECGResNet-Attention (Vibrant Emerald Green), ECGResNet-GAP (Amber/Orange), XResNet1D (Purple).

---

## Figure 4: Paired Bootstrap Resample Difference Distributions & 95% Confidence Intervals

- **Scientific Purpose**: Visually communicate the empirical distribution of performance differences across 1,000 paired bootstrap resamples, proving the statistical significance of Attention over GAP and the statistical indistinguishability of Attention and InceptionTime.
- **What It Should Show**:
  - **Panel A (Attention vs. GAP)**:
    - Histogram / KDE distribution of $\Delta\text{AUROC} = \text{Attn} - \text{GAP}$ across 1,000 resamples.
    - Shaded 95% empirical percentile interval: $[+0.0070, +0.0152]$, point estimate $+0.0110$.
    - Red dashed vertical zero-line ($x = 0$); entire distribution lies strictly to the right ($P = 100\%$).
    - Inset: Distribution of $\Delta\text{AP}$ ($[+0.0205, +0.0420]$, mean $+0.0310$, $P = 100\%$).
  - **Panel B (Attention vs. InceptionTime)**:
    - Histogram / KDE distribution of $\Delta\text{AUROC} = \text{Attn} - \text{Inception}$.
    - Shaded 95% interval: $[-0.0055, +0.0038]$, point estimate $-0.0012$.
    - Vertical zero-line passes directly through the center of mass ($P = 31.5\%$), visually confirming that the models are statistically indistinguishable under this protocol.
    - Inset: Distribution of $\Delta\text{AP}$ ($[-0.0087, +0.0108]$, point estimate $+0.0008$, $P = 59.5\%$).
  - **Panel C (Forest Plot of All 6 Pairwise Comparisons)**:
    - Forest plot showing point estimate deltas and horizontal error bars representing 95% bootstrap CIs for all 6 model pairs across AUROC, AP, and Macro F1.
- **Why It Matters**: Provides definitive, publication-ready statistical proof that eliminates ambiguity regarding whether gains are genuine or attributable to random test-set variation.
- **Exact Frozen Artifacts Supplying Data**:
  - `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`
  - `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md`
- **Visual Layout & Aesthetics**:
  - Paired distribution plots with kernel density estimation curves, shaded 95% confidence bands, and annotated median and zero lines.

---

## Figure 5: Hypertrophy (HYP) Stratification: Isolated vs. Co-occurring Phenotypes

- **Scientific Purpose**: Graphically illustrate the clinical representation disparity in Hypertrophy: the contrast between isolated hypertrophy and hypertrophy co-occurring with repolarization abnormalities or other cardiac conditions.
- **What It Should Show**:
  - **Panel A (Cohort Subphenotype Breakdown, $N = 262$)**:
    - Alluvial or stacked bar breakdown of the 262 HYP records in Fold 10:
      - Pure Isolated HYP (No STTC, no CD, no MI): $N = 56$ (21.37%).
      - Co-occurring with STTC: $N = 155$ (59.16%).
      - Co-occurring with CD: $N = 75$ (28.63%).
      - Co-occurring with MI: $N = 79$ (30.15%).
      - Co-occurring with $\ge 2$ pathologies: $N = 89$ (33.97%).
  - **Panel B (Operational Sensitivity Across Strata)**:
    - Bar chart contrasting sensitivity under validation-tuned thresholds across strata:
      - Isolated HYP: **1.79%** for Model A (1 / 56 detected) and **7.14%** for InceptionTime (4 / 56 detected).
      - HYP + STTC: **69.03%** for Model A (107 / 155 detected) and **63.87%** for InceptionTime (99 / 155 detected).
      - All HYP: **43.51%** for Model A (114 / 262) and **41.22%** for InceptionTime (108 / 262).
  - **Panel C (Predicted Probability Distributions Across Strata)**:
    - Box plots / violin plots showing mean predicted probability: Isolated HYP (mean $p \approx 0.068–0.100$, well below decision thresholds) vs. HYP + STTC (mean $p \approx 0.333–0.335$, well above decision thresholds).
- **Why It Matters**: Graphically substantiates the claim that model predictions exhibit phenotype-dependent representation sensitivity, without asserting causal decision mechanisms.
- **Exact Frozen Artifacts Supplying Data**:
  - `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`
  - `results/phase7/benchmark_fold10/hyp_analysis/HYP_ERROR_ANALYSIS.md`
- **Visual Layout & Aesthetics**:
  - 3-panel clinical diagnostic panel using muted burgundy for isolated disease and deep navy for composite disease.

---

## Figure 6: Spatial Lead Perturbation Sensitivity Profile & BatchNorm Shift

- **Scientific Purpose**: Illustrate spatial lead sensitivities for HYP under group occlusion while visually documenting the methodological caveat of BatchNorm activation drift during channel masking.
- **What It Should Show**:
  - **Panel A (Lead-Group Perturbation Profile)**:
    - Grouped bar chart showing HYP AUROC under three conditions across both ECGResNet-GAP and InceptionTime1D:
      - Baseline (Unmasked): 0.7682 (GAP) / 0.7899 (Inception)
      - Limb Occlusion (I, II, III, aVR, aVL, aVF masked): 0.7027 (GAP) / 0.7303 (Inception)
      - Precordial Occlusion ($V_1$–$V_6$ masked): **0.5847** (GAP) / **0.5758** (Inception)
    - Annotated drop indicating precordial sensitivity.
  - **Panel B (Single-Lead Probability Drift Heatmap / Bar Plot)**:
    - Plot showing global mean predicted probability shifts across all classes when zero-masking individual leads ($I, II, III, \dots, V_6$).
    - Highlight Lead II drift (+0.0412 mean probability shift) and precordial drifts.
    - Schematic caption: *"Lead masking was interpreted as a sensitivity probe rather than a causal attribution method because channel removal can introduce distributional shifts."*
- **Why It Matters**: Combines empirical insight (precordial leads show high perturbation sensitivity) with rigorous methodological restraint (occlusion is not clean causal attribution due to BatchNorm shift).
- **Exact Frozen Artifacts Supplying Data**:
  - `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_summary.csv`
  - `results/phase7/benchmark_fold10/hyp_single_lead/hyp_single_lead_summary.csv`
- **Visual Layout & Aesthetics**:
  - Anatomical lead layout diagram paired with degradation bar charts and drift heatmaps.

---

## Figure 7: Reliability Diagrams & Expected Calibration Error (ECE) Pre- and Post-Scaling

- **Scientific Purpose**: Evaluate the probabilistic reliability of model predictions, demonstrating that raw model probabilities are well-calibrated and showing the minor refinement provided by temperature scaling.
- **What It Should Show**:
  - **Panel A (Reliability Diagrams across 10 Equal-Width Bins)**:
    - Mean predicted probability vs. observed empirical frequency for each of the 5 diagnostic classes (`NORM`, `STTC`, `CD`, `MI`, `HYP`) on frozen Fold-10 test data.
    - Diagonal line representing perfect calibration ($y = x$).
  - **Panel B (Overall Calibration Summary)**:
    - Pre-calibration vs. post-calibration ($T = 0.9761$) reliability comparison.
    - Expected Calibration Error (ECE) annotated: **0.0269 (2.69%)**.
    - Macro Brier score annotated: **0.0969**.
  - **Panel C (Validation Threshold Stability Histogram)**:
    - Histogram of optimal decision thresholds across 1,000 bootstrap resamples of Fold 9 validation data, demonstrating peak stability at 0.25 for HYP (34.0% modal selection; 70.5% in $[0.23, 0.27]$).
- **Why It Matters**: Establishes that the network outputs dependable, clinically interpretable risk estimates, not overconfident binary scores.
- **Exact Frozen Artifacts Supplying Data**:
  - `results/phase7/benchmark_fold10/model_a_calibration/calibration_test_results.json`
  - `results/phase7/benchmark_fold10/model_a_calibration/threshold_stability_results.json`
  - `results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md`
- **Visual Layout & Aesthetics**:
  - 10-bin reliability diagrams with confidence histograms plotted as lower sub-bars.

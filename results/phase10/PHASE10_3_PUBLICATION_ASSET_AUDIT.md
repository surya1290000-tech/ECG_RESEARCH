# Phase 10.3 — Publication Asset Audit & Integrity Verification Report

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.3 Publication Figures and Tables Production  
**Date**: 2026-09-05  
**Governance Status**: APPROVED & FROZEN  

---

## 1. Executive Summary & Audit Scope

Phase 10.3 produced the complete suite of publication-ready figures (Figures 1–6) and tables (Tables 1–4) for the scientific manuscript titled:
> *"A Controlled Evaluation of Temporal, Attention-Based, and Multi-Scale Representations for Multi-Label 12-Lead ECG Classification on PTB-XL"*

Under the strict **Research Freeze Protocol**, this production pass was executed with:
- **Zero model training**
- **Zero model evaluation on new splits**
- **Zero modification of model checkpoints or SHA-256 hashes**
- **Zero modification of dataset files or split definitions**
- **Zero modification of frozen scientific results**
- **Zero introduction of new experiments**

Every single reported numerical value, plotted curve, bar height, confidence interval, and annotation across all assets traces directly to an existing, immutable frozen artifact in `results/phase7/`, `results/phase9/`, or `results/phase10/`.

---

## 2. Publication Figures Audit

| Figure ID | Raster Asset (300 DPI) | Vector Asset (PDF) | Scientific Focus | Grounded Source Artifact(s) | Status |
|:---:|:---|:---|:---|:---|:---:|
| **Figure 1** | [`figure_1_research_pipeline.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_1_research_pipeline.png) | [`figure_1_research_pipeline.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_1_research_pipeline.pdf) | End-to-end patient-isolated research pipeline & governance protocol | `data/ptbxl/ptbxl_database.csv`, `configs/config.py`, `FINAL_RESEARCH_AUDIT.md` | **PASS** |
| **Figure 2** | [`figure_2_architectures.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_2_architectures.png) | [`figure_2_architectures.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_2_architectures.pdf) | Neural architectural typology: Uniform GAP vs lightweight attention pooling (+513 params) and Inception multi-scale blocks | `src/models/ecg_resnet.py`, `src/models/ecg_resnet_attention.py`, `src/models/inceptiontime1d.py` | **PASS** |
| **Figure 3** | [`figure_3_roc_pr_comparison.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_3_roc_pr_comparison.png) | [`figure_3_roc_pr_comparison.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_3_roc_pr_comparison.pdf) | Four-model continuous macro ROC and PR curves across operational thresholds on Fold 10 ($N = 2,158$) | `model_a_fold10_predictions.csv`, `model_b_fold10_predictions.csv`, `model_xresnet_fold10_predictions.csv`, `model_inception_fold10_predictions.csv` | **PASS** |
| **Figure 4** | [`figure_4_paired_bootstrap.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_4_paired_bootstrap.png) | [`figure_4_paired_bootstrap.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_4_paired_bootstrap.pdf) | Paired statistical bootstrap comparison (forest plot of 95% CIs) and per-class diagnostic gains of Attention over GAP | `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`, `four_model_per_class_comparison.csv` | **PASS** |
| **Figure 5** | [`figure_5_hyp_stratification.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_5_hyp_stratification.png) | [`figure_5_hyp_stratification.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_5_hyp_stratification.pdf) | Hypertrophy (HYP) subphenotype stratification ($N=262$ total): Support distribution and severe sensitivity deficit on isolated cases | `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`, `HYP_ERROR_ANALYSIS.md` | **PASS** |
| **Figure 6** | [`figure_6_lead_sensitivity.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_6_lead_sensitivity.png) | [`figure_6_lead_sensitivity.pdf`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_6_lead_sensitivity.pdf) | Spatial lead-group occlusion sensitivity alongside single-lead probability drift quantifying BatchNorm distribution shift | `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_metrics.json`, `hyp_single_lead_global_drift.csv` | **PASS** |

**Figure Quality & Standards**:
- Both high-resolution raster (PNG, 300 DPI) and vector (PDF) formats are generated for all 6 figures.
- Consistent typography (`DejaVu Sans`), styling, linewidths, and unified color palette (Inception Navy `#1f4e79`, Attention Green `#1a8828`, GAP Amber `#d95f02`, XResNet Slate Purple `#7570b3`).
- Complete traceability documented in [`FIGURE_NUMBER_SOURCE_MAP.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/FIGURE_NUMBER_SOURCE_MAP.md) and [`FIGURE_INDEX.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/FIGURE_INDEX.md).

---

## 3. Publication Tables Audit

| Table ID | Markdown Document | Companion CSV | LaTeX Asset | Scientific Scope | Grounded Source Artifact(s) | Status |
|:---:|:---|:---|:---|:---|:---|:---:|
| **Table 1** | [`TABLE_1_DATASET_AND_SPLIT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_1_DATASET_AND_SPLIT.md) | `table_1_splits.csv` | `table_1.tex` | Cohort filtering (21,799 to 21,388), official Fold-10 splits, patient isolation (0 overlap), test class support, and preprocessing | `data/ptbxl/ptbxl_database.csv`, `configs/config.py` | **PASS** |
| **Table 2** | [`TABLE_2_MODEL_COMPLEXITY.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_2_MODEL_COMPLEXITY.md) | `table_2_complexity.csv` | `table_2.tex` | Model architecture specifications, parameter counts (+513 attention params), checkpoint disk footprint, and CPU latency/throughput | `results/phase9/four_model_benchmark/four_model_complexity.csv`, `CHECKPOINT_HASHES.csv` | **PASS** |
| **Table 3** | [`TABLE_3_BENCHMARK_PERFORMANCE.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_3_BENCHMARK_PERFORMANCE.md) | `table_3_point_estimates.csv` | `table_3.tex` | Four-model controlled Fold-10 benchmark results, paired bootstrap differences ($B=1000$), empirical 95% CIs, and per-class breakdown | `results/phase9/four_model_benchmark/four_model_point_estimates.csv`, `four_model_paired_bootstrap.csv` | **PASS** |
| **Table 4** | [`TABLE_4_HYP_STRATIFIED_ANALYSIS.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_4_HYP_STRATIFIED_ANALYSIS.md) | `table_4_hyp_stratification.csv` | `table_4.tex` | Authoritative HYP stratified error analysis ($N=262$) across all 6 clinical strata, sensitivity rates, and spatial lead perturbation bounds | `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`, `hyp_occlusion_metrics.json` | **PASS** |

**Table Quality & Standards**:
- Available in Markdown (`.md`), standardized CSV (`.csv`), and publication-ready LaTeX (`.tex`).
- Continuous ranking metrics (AUROC, AP) are strictly distinguished from threshold-dependent metrics (F1, precision, recall, subset accuracy, Hamming loss).
- Complete traceability documented in [`TABLE_NUMBER_SOURCE_MAP.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_NUMBER_SOURCE_MAP.md) and [`TABLE_INDEX.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_INDEX.md).

---

## 4. Scientific Language & Wording Guardrails Verification

1. **No SOTA Claims**: No asset or document claims "State-of-the-Art" (SOTA) performance. Performance is contextualized strictly within the controlled benchmark.
2. **No Biological Causation Claims**: Occlusion and lead perturbations are strictly termed "sensitivity probes" rather than causal attributions.
3. **No External Generalization Claims**: Evaluation is bounded strictly to the PTB-XL official Fold-10 test set without claiming out-of-distribution or external clinical validity.
4. **Latency Characterization**: Inference runtime is explicitly qualified as "hardware-specific measured latency on the evaluation host CPU (batch size 64)".
5. **Confidence Interval Scope**: Bootstrap confidence intervals are explicitly defined as reflecting test-cohort patient sampling variability, not retraining stability across seeds.
6. **Equivalence Language**: Attention and InceptionTime are characterized as "statistically indistinguishable under the evaluated protocol" rather than functionally equivalent.

---

## 5. Hypertrophy (HYP) Definitional Integrity Audit

Automated regex audit across all publication assets confirmed:
- **Total HYP Count**: Exactly $N = 262$ across all figures and tables.
- **Pure Isolated HYP**: Exactly $N = 56$ (1.79% recall for ECGResNet-GAP, 7.14% for InceptionTime1D).
- **HYP + STTC Composite**: Exactly $N = 155$ (69.03% recall for ECGResNet-GAP, 63.87% for InceptionTime1D).
- **Authoritative Provenance**: Strictly derived from `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`.
- **Forbidden Numbers Purged**: Zero active mentions of preliminary ungrounded draft numbers ($N = 118$, $N = 144$, $13.6\%$, or $68.1\%$).
- **Forbidden Artifact References**: Zero references to non-existent `hyp_subphenotype_metrics.csv`.

---

## 6. Research Freeze & Checkpoint Integrity Confirmation

- **Checkpoints**: All 4 official model checkpoints remain bit-for-bit identical to their frozen SHA-256 hashes recorded in `CHECKPOINT_HASHES.csv`.
- **Datasets**: `data/ptbxl/ptbxl_database.csv` and cached split files in `data/ptbxl/cache/` remain completely untouched.
- **Source Code**: No architecture definitions (`src/models/`), data pipelines (`src/data/`), or evaluation logic were modified.
- **Experiments**: Zero new training runs, zero fine-tuning runs, and zero parameter search runs were executed.

---

## 7. Final Audit Status

```
SCIENCE CHANGES: NONE
EXPERIMENTS RUN: NONE
CHECKPOINTS MODIFIED: NONE
DATASETS MODIFIED: NONE
FROZEN RESULTS MODIFIED: NONE
FIGURES CREATED: 6 (12 files: 6 PNG + 6 PDF)
TABLES CREATED: 4 (12 files: 4 MD + 4 CSV + 4 TEX)
BLOCKING ISSUES: 0
STATUS: PHASE 10.3 COMPLETE
```

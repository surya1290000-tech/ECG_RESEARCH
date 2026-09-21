# Publication Figure Index & Asset Manifest

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.3 Publication Figures and Tables Production  
**Status**: Authoritative, Grounded, and Frozen  

This directory contains the publication-ready figures for the scientific manuscript in both high-resolution raster (`.png`, 300 DPI) and vector (`.pdf`) formats. All visual data are strictly derived from frozen Phase 10.1 empirical artifacts.

---

## Master Figure Manifest

| Figure ID | Raster File (300 DPI) | Vector File (PDF) | Title / Scientific Role | Exact Supporting Artifact(s) |
|:---:|:---|:---|:---|:---|
| **Figure 1** | [`figure_1_research_pipeline.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_1_research_pipeline.png) | `figure_1_research_pipeline.pdf` | **End-to-End Research Workflow & Governance Protocol**: Cohort filtering (21,799 to 21,388), patient-isolated splitting with zero overlap, preprocessing, and frozen test firewall | `ptbxl_database.csv`, `configs/config.py`, `FINAL_RESEARCH_AUDIT.md` |
| **Figure 2** | [`figure_2_architectures.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_2_architectures.png) | `figure_2_architectures.pdf` | **Neural Architectural Typology & Temporal Pooling**: Detailed schematic contrasting uniform GAP vs lightweight attention pooling (+513 params) and InceptionTime multi-scale convolutions | `src/models/ecg_resnet.py`, `src/models/ecg_resnet_attention.py`, `src/models/inceptiontime1d.py` |
| **Figure 3** | [`figure_3_roc_pr_comparison.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_3_roc_pr_comparison.png) | `figure_3_roc_pr_comparison.pdf` | **Four-Model Macro ROC and PR Curves**: Complete continuous discriminative envelope across all operational thresholds on frozen Fold 10 ($N = 2,158$) | `model_a_fold10_predictions.csv`, `model_b_fold10_predictions.csv`, `model_xresnet_fold10_predictions.csv`, `model_inception_fold10_predictions.csv` |
| **Figure 4** | [`figure_4_paired_bootstrap.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_4_paired_bootstrap.png) | `figure_4_paired_bootstrap.pdf` | **Paired Statistical Bootstrap Comparison**: Forest plot of 95% empirical bootstrap CIs across key model pairs and per-class diagnostic gains of Attention over GAP | `results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`, `four_model_per_class_comparison.csv` |
| **Figure 5** | [`figure_5_hyp_stratification.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_5_hyp_stratification.png) | `figure_5_hyp_stratification.pdf` | **Hypertrophy (HYP) Subphenotype Stratification**: Cohort composition across 5 co-occurring strata ($N=262$), sensitivity disparity (1.79% isolated vs 69.03% composite), and predicted probability collapse | `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`, `HYP_ERROR_ANALYSIS.md` |
| **Figure 6** | [`figure_6_lead_sensitivity.png`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_figures/figure_6_lead_sensitivity.png) | `figure_6_lead_sensitivity.pdf` | **Lead Sensitivity Probing & Distributional Shift**: Precordial vs limb lead occlusion degradation alongside single-lead probability drift quantifying BatchNorm shift | `results/phase7/benchmark_fold10/hyp_occlusion/hyp_occlusion_summary.csv`, `hyp_single_lead_summary.csv` |

---

## Technical Specifications
- **Dimensions & DPI**: Standard double-column layout ($13\text{--}14 \times 6\text{--}8$ inches), rasterized at 300 DPI.
- **Color Palette**: Accessible, color-blind friendly corporate palette (Navy Blue `#1f4e79`, Emerald Green `#1a8828`, Amber `#d95f02`, Slate Purple `#7570b3`).
- **Terminology Guardrails Enforced**:
  - Sensitivity probing (NOT causal attribution).
  - Statistically indistinguishable under evaluated protocol (NOT equivalent).
  - Hardware-specific measured latency on CPU testbed.
  - Authoritative HYP numbers ($N=262$ total, $N=56$ isolated, $N=155$ composite).

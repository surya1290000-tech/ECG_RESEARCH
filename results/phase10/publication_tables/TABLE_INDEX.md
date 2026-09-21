# Publication Table Index & Asset Manifest

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.3 Publication Figures and Tables Production  
**Status**: Authoritative, Grounded, and Frozen  

This directory contains the publication-ready tables for the scientific manuscript in Markdown (`.md`) and standard CSV (`.csv`) formats. All numerical entries are directly traceable to frozen Phase 10.1 artifacts.

---

## Master Table Manifest

| Table ID | Canonical Document | Companion CSV | Title / Description | Source Artifact |
|:---:|:---|:---|:---|:---|
| **Table 1** | [`TABLE_1_DATASET_AND_SPLIT.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_1_DATASET_AND_SPLIT.md) | `table_1_splits.csv` | Dataset cohort filtering, official Fold-10 partitioning, patient isolation, and signal preprocessing specifications | `data/ptbxl/ptbxl_database.csv`, `configs/config.py` |
| **Table 2** | [`TABLE_2_MODEL_COMPLEXITY.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_2_MODEL_COMPLEXITY.md) | `table_2_complexity.csv` | Four-model architecture typology, parameter allocation (+513 attention params), checkpoint disk footprint, and CPU latency/throughput | `results/phase9/four_model_benchmark/four_model_complexity.csv`, `CHECKPOINT_HASHES.csv` |
| **Table 3** | [`TABLE_3_BENCHMARK_PERFORMANCE.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_3_BENCHMARK_PERFORMANCE.md) | `table_3_point_estimates.csv` | Controlled Fold-10 benchmark results, paired bootstrap differences (1,000 resamples), empirical CIs, and per-class diagnostic breakdown | `results/phase9/four_model_benchmark/four_model_point_estimates.csv`, `four_model_paired_bootstrap.csv` |
| **Table 4** | [`TABLE_4_HYP_STRATIFIED_ANALYSIS.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase10/publication_tables/TABLE_4_HYP_STRATIFIED_ANALYSIS.md) | `table_4_hyp_stratification.csv` | Authoritative Hypertrophy (HYP) subphenotype error analysis across all 6 clinical strata, sensitivity rates, and lead perturbation bounds | `results/phase7/benchmark_fold10/hyp_analysis/hyp_stratified_sensitivity.csv`, `hyp_occlusion_summary.csv` |

---

## Numerical Integrity & Freeze Confirmation
- Zero unverified or estimated numbers.
- Macro AUROC/AP segregated from threshold-dependent metrics.
- Authoritative HYP numbers strictly adhered to: Total HYP = 262, Pure Isolated HYP = 56 (1.79% recall in GAP, 7.14% in Inception), HYP+STTC = 155 (69.03% recall in GAP, 63.87% in Inception).
- Zero mentions of superseded preliminary figures ($N=118/144$, $13.6\%/68.1\%$).

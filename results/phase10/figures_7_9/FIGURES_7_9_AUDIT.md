# FIGURES 7-9 AUDIT REPORT

**Project**: Controlled 12-Lead ECG Benchmark on PTB-XL  
**Phase**: 10.4 - Additional Publication Figures (7, 8, 9)  
**Generated**: 2026-09-17  
**Status**: COMPLETE AND VERIFIED

---

## 1. Integrity Declarations

| Rule | Status |
|:---|:---:|
| No model retraining performed | CONFIRMED |
| No checkpoint loaded, modified, or evaluated | CONFIRMED |
| No test-set threshold tuning performed | CONFIRMED |
| Fold-10 frozen test set remains intact | CONFIRMED |
| Figures 1-6 were NOT modified | CONFIRMED (all 12 files verified present and untouched) |
| No values estimated or fabricated | CONFIRMED |
| All values traceable to machine-readable Phase 9 artifacts | CONFIRMED |
| Script is deterministic (SEED=42, Agg backend) | CONFIRMED |

---

## 2. Authoritative Source Artifacts Used

### Global Benchmark Artifacts (Phase 9)
| File | Role |
|:---|:---|
| `results/phase9/four_model_benchmark/four_model_point_estimates.csv` | Macro-AUROC, parameter counts (Figures 7 audit + Figure 8) |
| `results/phase9/four_model_benchmark/four_model_complexity.csv` | Inference latency ms/ECG (Figure 8) |
| `results/phase9/four_model_benchmark/four_model_prediction_alignment.json` | Pre-verified alignment record (Figure 9) |

### Per-Model Evaluation Artifacts
| File | Model | Role |
|:---|:---|:---|
| `results/phase7/benchmark_fold10/model_a/model_a_fold10_metrics.json` | ECGResNet-GAP | Per-class AUROC/AP (Figure 7) |
| `results/phase9/model_b_fold10_evaluation/model_b_fold10_per_class.csv` | ECGResNet-Attention | Per-class AUROC/AP (Figure 7) |
| `results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_metrics.json` | XResNet1D | Per-class AUROC/AP (Figure 7) |
| `results/phase7/benchmark_fold10/model_inception/model_inception_fold10_metrics.json` | InceptionTime | Per-class AUROC/AP (Figure 7) |

### Frozen Fold-10 Prediction Files (Figure 9)
| File | Model |
|:---|:---|
| `results/phase7/benchmark_fold10/model_a/model_a_fold10_predictions.csv` | ECGResNet-GAP |
| `results/phase9/model_b_fold10_evaluation/model_b_fold10_predictions.csv` | ECGResNet-Attention |
| `results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv` | XResNet1D |
| `results/phase7/benchmark_fold10/model_inception/model_inception_fold10_predictions.csv` | InceptionTime |

---

## 3. Prediction Alignment Result (Figure 9 Prerequisite)

| Check | Result |
|:---|:---:|
| Row count: all 4 files contain exactly 2,158 records | PASS |
| ECG ID alignment: GAP / Attention / XResNet1D IDs 100% identical | PASS |
| Duplicate ECG IDs: none detected | PASS |
| Ground-truth label alignment: all 4 models 100% identical | PASS |
| Pre-recorded alignment in `four_model_prediction_alignment.json` | VERIFIED (2026-09-04 23:10:17) |

**Alignment conclusion: 100%. Figure 9 computation proceeded.**

---

## 4. Figure 7 - Per-Class Performance

**Title**: "Per-Class Performance Across Four ECG Architectures"  
**Panels**: (A) AUROC by diagnostic superclass | (B) Average Precision by diagnostic superclass  
**X-axis**: NORM, STTC, CD, MI, HYP  
**Models**: ECGResNet-GAP, ECGResNet-Attention, XResNet1D, InceptionTime  

### Exact Values Used (loaded from machine-readable artifacts)

| Model | Class | AUROC | Average Precision |
|:---|:---|:---:|:---:|
| ECGResNet-GAP | NORM | 0.93163 | 0.90504 |
| ECGResNet-GAP | STTC | 0.91679 | 0.77928 |
| ECGResNet-GAP | CD   | 0.90248 | 0.80658 |
| ECGResNet-GAP | MI   | 0.91507 | 0.79065 |
| ECGResNet-GAP | HYP  | 0.76822 | 0.38539 |
| ECGResNet-Attention | NORM | 0.94082 | 0.91981 |
| ECGResNet-Attention | STTC | 0.92844 | 0.82264 |
| ECGResNet-Attention | CD   | 0.90600 | 0.82102 |
| ECGResNet-Attention | MI   | 0.91902 | 0.81670 |
| ECGResNet-Attention | HYP  | 0.79508 | 0.44171 |
| XResNet1D | NORM | 0.91614 | 0.88153 |
| XResNet1D | STTC | 0.91537 | 0.78114 |
| XResNet1D | CD   | 0.89902 | 0.78475 |
| XResNet1D | MI   | 0.87915 | 0.75631 |
| XResNet1D | HYP  | 0.77790 | 0.43421 |
| InceptionTime | NORM | 0.93847 | 0.91106 |
| InceptionTime | STTC | 0.92755 | 0.81328 |
| InceptionTime | CD   | 0.92004 | 0.84327 |
| InceptionTime | MI   | 0.91940 | 0.81664 |
| InceptionTime | HYP  | 0.78991 | 0.43369 |

> **Note**: AUROC and AP are threshold-invariant. Values loaded from the validation-tuned-threshold section of each metrics file (identical to default 0.5 threshold for these two metrics).

**Output files**:
- `figure_7_per_class_performance.png` (SHA-256: 7ad273ef55ae082f66eff2ef5b1e1978f82f136d608e108cbb7d6c485d6c0f95)
- `figure_7_per_class_performance.pdf` (SHA-256: d2b30aeea2d50ced8fca7a24db822516658fcf0833c4aa7da63ad5d5f9a06900)
- `figure_7_per_class_performance.svg` (SHA-256: 21dcae6e070488fad3b2d33313cd8b2dcb257b6e6af684cf0d022e07c4d1fe47)
- `figure_7_per_class_performance.csv` (SHA-256: ff536c7622f8b46516483fe0e143a29ee522f1bbfcef0049e815f48de75ea742)

---

## 5. Figure 8 - Performance-Complexity Trade-off

**Title**: "Performance-Complexity Trade-off Across Four ECG Architectures"  
**Panels**: (A) Macro-AUROC vs. Parameters (Millions) | (B) Macro-AUROC vs. Inference Time (ms/ECG)

### Exact Values Used

| Model | Parameters | Params (M) | Macro-AUROC | Inference (ms/ECG) |
|:---|:---:|:---:|:---:|:---:|
| ECGResNet-GAP | 3,919,493 | 3.919493 | 0.8868369445 | 24.1 |
| ECGResNet-Attention | 3,920,006 | 3.920006 | 0.8978732873 | 16.0 |
| XResNet1D | 3,931,525 | 3.931525 | 0.8775154274 | 24.2 |
| InceptionTime | 3,886,149 | 3.886149 | 0.8990750504 | 24.1 |

> **Interpretation note**: This figure is descriptive only. No regression line, no Pareto frontier, no causal claims.

**Output files**:
- `figure_8_performance_complexity.png` (SHA-256: baca35839f0dc19e4f9b63db680e6113a79137c7c62759ff9dbde13d72a90383)
- `figure_8_performance_complexity.pdf` (SHA-256: 6f20d5f94740d199bc012bfd01a0418480530c0d587039454176fea4473ed0ed)
- `figure_8_performance_complexity.svg` (SHA-256: 21aee61fd9625632428c8613e6dfc7b2b25da752b2b2f1ddf0cffdeaf52f0a3d)
- `figure_8_performance_complexity.csv` (SHA-256: b5bd31c2142eaee541e5e6bd7de197138deea19f7ccfb3c9d1247f5104a6743c)

---

## 6. Figure 9 - Prediction Agreement

**Title**: "Prediction Agreement Across Four ECG Architectures"  
**Panels**: (A) 4x4 Pearson correlation heatmap | (B) Boxplots of pairwise absolute probability differences  
**N**: 2,158 frozen Fold-10 test records  
**Classes**: NORM, STTC, CD, MI, HYP (concatenated: N x 5 = 10,790 values per model)  
**Correlation method**: Pearson r on concatenated continuous probabilities  

### Panel A: Pearson Correlation Matrix

|  | GAP | Attention | XResNet1D | InceptionTime |
|:---|:---:|:---:|:---:|:---:|
| **GAP** | 1.000000 | 0.932891 | 0.885895 | 0.941042 |
| **Attention** | 0.932891 | 1.000000 | 0.886255 | 0.929889 |
| **XResNet1D** | 0.885895 | 0.886255 | 1.000000 | 0.873775 |
| **InceptionTime** | 0.941042 | 0.929889 | 0.873775 | 1.000000 |

### Panel B: Pairwise Absolute Difference Summary

| Pair | Mean |Delta| | Median |Delta| | Q75 |Delta| |
|:---|:---:|:---:|:---:|
| Attention vs GAP | 0.0724 | 0.0337 | 0.0930 |
| Attention vs XResNet1D | 0.0895 | 0.0365 | 0.1103 |
| Attention vs InceptionTime | 0.0704 | 0.0299 | 0.0888 |
| InceptionTime vs GAP | 0.0627 | 0.0285 | 0.0791 |
| InceptionTime vs XResNet1D | 0.0925 | 0.0396 | 0.1158 |
| GAP vs XResNet1D | 0.0868 | 0.0387 | 0.1089 |

> **Interpretation note**: This figure is descriptive only. Correlation does not imply equivalent learned representations. No causal claims are made.

**Output files**:
- `figure_9_prediction_agreement.png` (SHA-256: 15046e8880d8984204872d0736eac45b5909844c22625561e86a97f14877d6cd)
- `figure_9_prediction_agreement.pdf` (SHA-256: 128feb56f4f07b6308babbf41c6224ccdbc5dba26d8cd404f995d149652e6aba)
- `figure_9_prediction_agreement.svg` (SHA-256: f170a9725ca3aeb1de7037ad50d08f5c383bf4337b2a5f4d88a1f1037175134a)
- `figure_9_prediction_agreement_correlations.csv` (SHA-256: a496aea8bb25fc9dd57f601c9e3e2866361642229cf271de14328b14fce17c3b)
- `figure_9_prediction_difference_summary.csv` (SHA-256: 3b1aae338c3d65e0ec1a55ffd1cff2dbb970e744827a9b85ddceed8d23eb9005)

---

## 7. Output File Inventory

All files in `results/phase10/figures_7_9/`:

| File | Type | SHA-256 |
|:---|:---|:---|
| `generate_figures_7_9.py` | Script | (deterministic source) |
| `figure_7_per_class_performance.png` | Figure | 7ad273ef... |
| `figure_7_per_class_performance.pdf` | Figure | d2b30aee... |
| `figure_7_per_class_performance.svg` | Figure | 21dcae6e... |
| `figure_7_per_class_performance.csv` | Data | ff536c76... |
| `figure_8_performance_complexity.png` | Figure | baca3583... |
| `figure_8_performance_complexity.pdf` | Figure | 6f20d5f9... |
| `figure_8_performance_complexity.svg` | Figure | 21aee61f... |
| `figure_8_performance_complexity.csv` | Data | b5bd31c2... |
| `figure_9_prediction_agreement.png` | Figure | 15046e88... |
| `figure_9_prediction_agreement.pdf` | Figure | 128feb56... |
| `figure_9_prediction_agreement.svg` | Figure | f170a972... |
| `figure_9_prediction_agreement_correlations.csv` | Data | a496aea8... |
| `figure_9_prediction_difference_summary.csv` | Data | 3b1aae33... |
| `FIGURES_7_9_AUDIT.md` | Audit | (this file) |

---

## 8. Figures 1-6 Verification

This script does NOT write to `results/phase10/publication_figures/`.

Verification result: All 12 expected Figure 1-6 files (6 PNG + 6 PDF) confirmed present and unmodified. PASS.

---

_Audit generated by: generate_figures_7_9.py_  
_Script timestamp: 2026-09-17_  
_Research freeze timestamp: 2026-09-04_
# ECG Classification on PTB-XL: A Controlled Evaluation of Temporal Aggregation Strategies

This repository contains the complete source code, trained model checkpoints, evaluation pipelines, and frozen research artifacts for a systematic comparison of deep learning architectures for 12-lead ECG classification on the PTB-XL v1.0.3 dataset.

The central research question is whether replacing Global Average Pooling (GAP) with a lightweight learned temporal attention pooling mechanism produces a measurable and statistically significant improvement on a patient-isolated multi-label classification benchmark.

---

## Table of Contents

1. [Research Objective](#1-research-objective)
2. [Dataset](#2-dataset)
3. [Experimental Protocol](#3-experimental-protocol)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Models](#5-models)
6. [Repository Structure](#6-repository-structure)
7. [Notebook Guide](#7-notebook-guide)
8. [Installation](#8-installation)
9. [Dataset Setup](#9-dataset-setup)
10. [Running the Experiments](#10-running-the-experiments)
11. [Trained Checkpoints](#11-trained-checkpoints)
12. [Results](#12-results)
13. [Statistical Analysis](#13-statistical-analysis)
14. [Figures and Reports](#14-figures-and-reports)
15. [Reproducibility](#15-reproducibility)
16. [Dataset Availability](#16-dataset-availability)
17. [Limitations](#17-limitations)
18. [License](#18-license)

---

## 1. Research Objective

This project investigates whether adding a small learned temporal attention pooling layer (513 parameters) to an ECG residual network changes multi-label classification performance compared to a Global Average Pooling (GAP) baseline, under a strict patient-isolated benchmark protocol using the official PTB-XL stratified folds.

Three additional architectures --- XResNet1D and InceptionTime1D --- are evaluated as external reference points in the same benchmark.

---

## 2. Dataset

**PTB-XL: A Large Publicly Available Electrocardiography Dataset**

| Property | Value |
|:---|:---|
| Source | PhysioNet --- https://physionet.org/content/ptb-xl/1.0.3/ |
| Version | 1.0.3 |
| Total records | 21,799 clinical 12-lead ECG recordings |
| ECG duration | 10 seconds |
| Number of leads | 12 (standard clinical configuration) |
| Sampling frequency used | 100 Hz (1,000 samples per lead; `records100/` files) |
| Input tensor shape | (batch, 12, 1,000) |
| Diagnostic labels | 5 superclasses: NORM, STTC, CD, MI, HYP |
| Label format | Multi-label binary (one ECG may belong to multiple classes) |
| Cohort used | 21,388 records after filtering to diagnostic superclasses |
| Excluded | 411 records with no diagnostic superclass mapping |

**Diagnostic Superclasses:**

| Code | Full Name |
|:---|:---|
| NORM | Normal ECG |
| STTC | ST/T-wave Changes |
| CD | Conduction Disturbance |
| MI | Myocardial Infarction |
| HYP | Hypertrophy |

The PTB-XL dataset is not included in this repository. See [Section 9](#9-dataset-setup) for setup instructions.

---

## 3. Experimental Protocol

The project uses the **official PTB-XL stratified fold protocol** (`strat_fold` column) with strict patient-level isolation.

| Partition | Folds | Records | Unique Patients |
|:---|:---|---:|---:|
| Training | 1-8 | 17,084 | 14,823 |
| Validation | 9 | 2,146 | 1,917 |
| Frozen Test | 10 | 2,158 | 1,877 |

- **Zero patient overlap** verified across all three partitions.
- Fold 10 was strictly frozen: never used for training, early stopping, or threshold search.
- **Decision thresholds** derived exclusively from Fold 9 validation predictions (grid search maximizing macro F1).
- **Continuous ranking metrics** (Macro AUROC, Macro AP) and **threshold-dependent metrics** (Macro F1, Weighted F1, Subset Accuracy, Hamming Loss) are treated separately.

---

## 4. Preprocessing Pipeline

All ECG signals are preprocessed identically across all models:

1. **Load**: Read WFDB-format waveforms at 100 Hz from `records100/`
2. **Bandpass filter**: 2nd-order Butterworth, 0.5-40 Hz, zero-phase (`sosfiltfilt`)
3. **Transpose**: Shape to `(12, 1000)` --- 12 leads x 1,000 time samples
4. **Normalize**: Per-lead Z-score normalization (mean=0, std=1)
5. **No handcrafted features**: All classifiers operate on filtered voltage time series only

Preprocessing code: [`src/data/dataset.py`](src/data/dataset.py)

---

## 5. Models

All four models use input shape `(batch, 12, 1000)` and output `(batch, 5)` logits.

| Model | Checkpoint | Architecture | Parameters | Role |
|:---|:---|:---|---:|:---|
| **ECGResNet-GAP** | `model_a_fold10_best.pth` | 4-block 1-D ResNet (kernel-7), Global Average Pooling + MLP head | 3,919,493 | Baseline |
| **ECGResNet-Attention** | `model_b_fold10_best.pth` | Same as GAP but with Temporal Attention Pooling replacing GAP (+513 params) | 3,920,006 | Primary comparison |
| **XResNet1D** | `model_xresnet_fold10_best.pth` | xResNet adapted to 1-D ECG signals | 3,931,525 | External reference |
| **InceptionTime1D** | `model_inception_fold10_best.pth` | InceptionTime adapted to 1-D ECG signals | 3,886,149 | External reference |

The only architectural difference between ECGResNet-GAP and ECGResNet-Attention is the pooling layer. Temporal Attention Pooling applies a shared Conv1d(512->1, kernel_size=1) followed by softmax over 125 temporal frames, producing input-dependent weights for the aggregation. This adds exactly 513 trainable parameters.

Model definitions:
- [`src/models/ecg_resnet.py`](src/models/ecg_resnet.py) --- ECGResNet-GAP
- [`src/models/attention_pooling.py`](src/models/attention_pooling.py) --- ECGResNet-Attention
- [`src/models/xresnet1d.py`](src/models/xresnet1d.py) --- XResNet1D
- [`src/models/inceptiontime1d.py`](src/models/inceptiontime1d.py) --- InceptionTime1D

---

## 6. Repository Structure

```
ECG_RESEARCH/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- .gitattributes                     # Git LFS for *.pth, *.pt, *.onnx, *.bin
+-- verify_checkpoint.py               # Checkpoint integrity checker
+-- arth.ipynb                         # Primary research notebook (all phases)
|
+-- configs/
|   +-- config.py                      # Paths, hyperparameters, class names
|
+-- src/
|   +-- data/
|   |   +-- dataset.py                 # PTB-XL loading, preprocessing, Fold-10 splits
|   |   +-- multilabel_dataset.py
|   +-- models/
|   |   +-- ecg_resnet.py              # ECGResNet-GAP + ResidualBlock
|   |   +-- attention_pooling.py       # ECGResNet-Attention + TemporalAttentionPooling
|   |   +-- xresnet1d.py
|   |   +-- inceptiontime1d.py
|   +-- training/
|   |   +-- train_model_a_fold10.py    # Official Fold-10 GAP training
|   |   +-- train_model_b_fold10.py    # Official Fold-10 Attention training
|   |   +-- train_model_xresnet_fold10.py
|   |   +-- train_model_inception_fold10.py
|   |   +-- (+ additional training scripts)
|   +-- evaluation/
|   |   +-- evaluate_four_model_benchmark_fold10.py
|   |   +-- evaluate_hyp_lead_occlusion_fold10.py
|   |   +-- evaluate_hyp_single_lead_fold10.py
|   |   +-- evaluate_model_a_calibration.py
|   |   +-- generate_publication_figures.py
|   |   +-- (+ additional evaluation scripts)
|   +-- utils/
|       +-- checkpoint.py
|
+-- checkpoints/                       # Model weights via Git LFS
|   +-- model_a_fold10_best.pth        # Official Fold-10 checkpoint: ECGResNet-GAP
|   +-- model_b_fold10_best.pth        # Official Fold-10 checkpoint: ECGResNet-Attention
|   +-- model_xresnet_fold10_best.pth  # Official Fold-10 checkpoint: XResNet1D
|   +-- model_inception_fold10_best.pth  # Official Fold-10 checkpoint: InceptionTime1D
|   +-- (+ historical phase checkpoints)
|
+-- results/
|   +-- EXPERIMENT_LEDGER.md
|   +-- phase3_5/ through phase6/      # Intermediate experiment results
|   +-- phase7/benchmark_fold10/       # Per-model evaluation, calibration, HYP analysis
|   +-- phase9/four_model_benchmark/   # Definitive four-model comparison
|   +-- phase10/
|       +-- research_freeze/           # SHA-256 hashes, artifact inventory, audit
|       +-- publication_figures/       # Publication-ready Figures 1-6 (PNG + PDF)
|       +-- figures_7_9/               # Figures 7-9 (PNG + PDF)
|       +-- publication_tables/        # Tables 1-4 (Markdown, CSV, LaTeX)
|       +-- paper_blueprint/
|       +-- claim_correction/
|
+-- data/
    +-- README.md                      # Dataset download instructions
```

---

## 7. Notebook Guide

| Notebook | Purpose |
|:---|:---|
| [`arth.ipynb`](arth.ipynb) | Primary research notebook covering all experimental phases: data loading and verification, ECG preprocessing, training all four architectures, attention weight visualization, representation probing, calibration analysis, HYP subgroup analysis, lead sensitivity probing, four-model benchmark comparison, and publication figure generation. All `results/` artifacts trace back to this notebook or the corresponding `src/` scripts. |

---

## 8. Installation

```bash
git clone https://github.com/surya1290000-tech/ECG_RESEARCH.git
cd ECG_RESEARCH

# Download LFS-tracked checkpoint files
git lfs install
git lfs pull

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Dataset Setup

The PTB-XL dataset is not included. Download from PhysioNet:

```
https://physionet.org/content/ptb-xl/1.0.3/
```

Required directory layout after extraction:

```
data/ptbxl/
+-- ptbxl_database.csv
+-- scp_statements.csv
+-- records100/
    +-- 00000/
    +-- 01000/
    +-- ...
```

Use the **100 Hz** waveforms (`records100/`). Do not use `records500/`. No code changes are needed if you follow this layout --- `configs/config.py` resolves all paths relative to the repository root.

---

## 10. Running the Experiments

Run from the repository root. All scripts expect the project root on `sys.path`.

```bash
# 1. Verify dataset is accessible
python -c "from src.data.dataset import load_ptbxl_multilabel_metadata; df,_=load_ptbxl_multilabel_metadata(); print(f'Records: {len(df)}')"
# Expected: Records: 21388

# 2. Verify checkpoint integrity (no dataset needed)
python verify_checkpoint.py

# 3. Train all four models (Fold-10 protocol)
python -m src.training.train_model_a_fold10
python -m src.training.train_model_b_fold10
python -m src.training.train_model_xresnet_fold10
python -m src.training.train_model_inception_fold10

# 4. Run the definitive four-model benchmark
python -m src.evaluation.evaluate_four_model_benchmark_fold10

# 5. HYP subgroup analysis
python -m src.evaluation.evaluate_hyp_lead_occlusion_fold10
python -m src.evaluation.evaluate_hyp_single_lead_fold10

# 6. Generate publication figures
python -m src.evaluation.generate_publication_figures
```

---

## 11. Trained Checkpoints

Checkpoint files are stored using **Git LFS**. Run `git lfs pull` after cloning.

### Official Fold-10 Benchmark Checkpoints

| Model | File | SHA-256 (first 16 chars) | Parameters | Epochs |
|:---|:---|:---:|---:|---:|
| ECGResNet-GAP | `model_a_fold10_best.pth` | `995eb2c70ba74184` | 3,919,493 | 20 |
| ECGResNet-Attention | `model_b_fold10_best.pth` | `8361b3bfbb85ec13` | 3,920,006 | 5 |
| XResNet1D | `model_xresnet_fold10_best.pth` | `aee7bf2e9b37eb11` | 3,931,525 | 20 |
| InceptionTime1D | `model_inception_fold10_best.pth` | `18277a08339eeb00` | 3,886,149 | 7 |

Full SHA-256 hashes: [`results/phase10/research_freeze/CHECKPOINT_HASHES.csv`](results/phase10/research_freeze/CHECKPOINT_HASHES.csv)

---

## 12. Results

Evaluated on the frozen Fold-10 test set (N = 2,158 ECGs, N = 1,877 unique patients).

Source: [`results/phase9/four_model_benchmark/four_model_point_estimates.csv`](results/phase9/four_model_benchmark/four_model_point_estimates.csv)

### Benchmark Performance

| Model | Macro AUROC | Macro AP | Macro F1 | Weighted F1 | Subset Acc. | Hamming Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ECGResNet-GAP** | 0.8868 | 0.7334 | 0.6907 | 0.7400 | 56.67% | 0.1399 |
| **ECGResNet-Attention** | 0.8979 | 0.7644 | 0.7052 | 0.7507 | 58.71% | 0.1338 |
| XResNet1D | 0.8775 | 0.7276 | 0.6680 | 0.7150 | 50.88% | 0.1606 |
| InceptionTime1D | 0.8991 | 0.7636 | 0.7070 | 0.7548 | 58.80% | 0.1296 |

### Inference Performance (CPU, Fold-10 test set)

| Model | Latency (ms/ECG) | Throughput (ECGs/s) |
|:---|:---:|:---:|
| ECGResNet-Attention | 16.0 | 62.6 |
| ECGResNet-GAP | 24.1 | 41.4 |
| InceptionTime1D | 24.1 | 41.6 |
| XResNet1D | 24.2 | 41.3 |

---

## 13. Statistical Analysis

Paired bootstrap (1,000 resamples, SEED=42, record-level sampling with identical indices across models).

Source: [`results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv`](results/phase9/four_model_benchmark/four_model_paired_bootstrap.csv)

### ECGResNet-Attention vs. ECGResNet-GAP

| Metric | Delta | 95% CI | P(Attn > GAP) |
|:---|:---:|:---:|:---:|
| Macro AUROC | +0.0110 | [+0.0070, +0.0152] | 100.0% |
| Macro AP | +0.0310 | [+0.0205, +0.0420] | 100.0% |
| Macro F1 | +0.0144 | [+0.0032, +0.0264] | 99.6% |

The attention mechanism produces a statistically significant improvement over GAP across all three primary metrics.

### InceptionTime1D vs. ECGResNet-Attention

| Metric | Delta | 95% CI | P(Inc > Attn) |
|:---|:---:|:---:|:---:|
| Macro AUROC | -0.0012 | [-0.0055, +0.0038] | 31.5% |
| Macro AP | +0.0008 | [-0.0087, +0.0108] | 59.5% |
| Macro F1 | -0.0019 | [-0.0134, +0.0099] | 38.4% |

ECGResNet-Attention and InceptionTime1D are statistically indistinguishable.

---

## 14. Figures and Reports

### Publication Figures (`results/phase10/publication_figures/`)

| Figure | File | Content |
|:---|:---|:---|
| Fig 1 | `figure_1_research_pipeline.png` | Research pipeline diagram |
| Fig 2 | `figure_2_architectures.png` | Model architecture comparison |
| Fig 3 | `figure_3_roc_pr_comparison.png` | ROC and Precision-Recall curves |
| Fig 4 | `figure_4_paired_bootstrap.png` | Bootstrap distribution plots |
| Fig 5 | `figure_5_hyp_stratification.png` | HYP phenotypic stratification |
| Fig 6 | `figure_6_lead_sensitivity.png` | Lead-group sensitivity analysis |
| Fig 7 | `figures_7_9/figure_7_per_class_performance.png` | Per-class AUROC/AP comparison |
| Fig 8 | `figures_7_9/figure_8_performance_complexity.png` | Performance vs complexity tradeoff |
| Fig 9 | `figures_7_9/figure_9_prediction_agreement.png` | Prediction agreement analysis |

All figures also available as PDF and SVG in the same directories.

### Key Reports

| Report | Path |
|:---|:---|
| Experiment Ledger | `results/EXPERIMENT_LEDGER.md` |
| Four-Model Benchmark | `results/phase9/four_model_benchmark/FOUR_MODEL_BENCHMARK_REPORT.md` |
| Research Freeze Audit | `results/phase10/research_freeze/FINAL_RESEARCH_AUDIT.md` |
| Number Source Map | `results/phase10/research_freeze/PAPER_NUMBER_SOURCE_MAP.md` |
| Calibration Report | `results/phase7/benchmark_fold10/model_a_calibration/CALIBRATION_REPORT.md` |
| HYP Occlusion Report | `results/phase7/benchmark_fold10/hyp_occlusion/HYP_OCCLUSION_REPORT.md` |

---

## 15. Reproducibility

1. Clone the repository and run `git lfs pull`
2. Install dependencies with `pip install -r requirements.txt`
3. Download PTB-XL v1.0.3 and place in `data/ptbxl/` (see Section 9)
4. Train models using `src/training/train_model_*_fold10.py`
5. Evaluate with `src/evaluation/evaluate_four_model_benchmark_fold10.py`

The full reproducibility checklist is at: [`results/phase10/research_freeze/REPRODUCIBILITY_CHECKLIST.md`](results/phase10/research_freeze/REPRODUCIBILITY_CHECKLIST.md)

**Seeds:** All scripts use `SEED = 42`. The Fold-10 split is fully determined by the `strat_fold` column in `ptbxl_database.csv`.

---

## 16. Dataset Availability

PTB-XL is distributed under the Open Database License (ODbL) and is freely available on PhysioNet without registration.

> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45

---

## 17. Limitations

- **HYP sensitivity:** Pure isolated HYP cases (no co-occurring STTC/CD/MI; N=56 in Fold 10) show very low detection rates across all models. This reflects the difficulty of voltage-criterion-free hypertrophy detection rather than a pipeline failure.
- **Lead-masking interpretation:** Zero-masking of lead groups causes BatchNorm activation distribution shifts; occlusion results represent sensitivity bounds, not definitive causal attribution.
- **Single training run per model:** No multi-seed ensemble averaging.
- **CPU timing:** Inference benchmarks are CPU-measured; GPU results would differ.

---

## 18. License

No license has been explicitly declared for this repository. Contact the repository owner before reuse or redistribution.

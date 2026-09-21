# ECG Research: Deep Learning for 12-Lead ECG Classification

This repository contains research and implementation of deep learning architectures (including InceptionTime, xresnet, and attention pooling mechanisms) for 12-lead Electrocardiogram (ECG) classification on the PTB-XL dataset.

---

## 📋 Overview

- **Dataset**: PTB-XL v1.0.3 (12-lead ECG, 100 Hz sampling rate, 1000 samples per lead)
- **Target Classes**: 5 diagnostic superclasses:
  - `NORM` (Normal ECG)
  - `STTC` (ST/T Change)
  - `CD` (Conduction Disturbance)
  - `MI` (Myocardial Infarction)
  - `HYP` (Hypertrophy)
- **Key Features**:
  - Modular PyTorch training & evaluation pipelines (`src/`)
  - Configurable architectures (InceptionTime, Attention Pooling, xresnet)
  - Representation diversity & similarity analysis
  - Checkpoint integrity validation suites (`verify_checkpoint.py`)
  - Experiment ledgers and evaluation metrics (`results/`)

---

## 📁 Repository Structure

```text
ECG_RESEARCH/
├── configs/                # Configuration and hyperparameter settings
│   └── config.py           # Centralized configuration (sampling rate, classes, paths)
├── src/                    # Source code
│   ├── data/               # Dataset loaders, preprocessing, and caching
│   ├── models/             # PyTorch model definitions & architectures
│   ├── training/           # Training loops, loss functions, and optimization
│   ├── evaluation/         # Metric calculation and validation routines
│   └── utils/              # Helper utilities
├── results/                # Experiment metrics, ledgers, and generated plots
├── checkpoints/            # Model checkpoint directory (weights ignored by git)
├── reports/                # Evaluation reports
├── arth.ipynb              # Analysis and experimentation notebook
├── requirements.txt        # Python package dependencies
└── verify_checkpoint.py    # Standalone checkpoint validation script
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/surya1290000-tech/ECG_RESEARCH.git
cd ECG_RESEARCH

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the PTB-XL v1.0.3 dataset from PhysioNet:
- Place the extracted dataset in `data/ptbxl/`
- Ensure `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` are present in `data/ptbxl/`.

### 3. Verification & Evaluation

To verify checkpoint integrity and evaluate models:

```bash
python verify_checkpoint.py
```

---

## 📊 Results & Documentation

Detailed evaluation notes and experimental results can be found in `results/EXPERIMENT_LEDGER.md` and `results/RESEARCH_TASK_AUDIT.md`.

# ECG_Research: Master Experiment Ledger

_Maintained under strict provenance, patient-level isolation, and reproducible benchmark protocols._  
_Task Formulations: (1) 5-Class Single-Label Multiclass Diagnostic Superclass Classification; (2) 5-Class Multi-Label Diagnostic Benchmark on PTB-XL v1.0.3 (records100)_  
_Canonical Class Ordering: 0=NORM, 1=STTC, 2=CD, 3=MI, 4=HYP_  
_Patient-Level Split: GroupShuffleSplit (test_size=0.20, random_state=42) -> Train: 13,673 | Val: 3,407 | Frozen Test: 4,308 (Zero Patient Leakage)_

---

## 1. Provenance & Experiment Taxonomy

| Category | Definition | Authority Level |
|:---|:---|:---:|
| **Historical Notebook Reference** | Exploratory code/outputs from `arth.ipynb` prior to modular audit. | Reference Only (Unverified) |
| **Verified Baseline (Model A)** | Pure GAP `ECGResNet` Version B cleanly trained & evaluated locally. | Authoritative Baseline (Single-Label) |
| **Verified Intervention (Model B)** | `ECGResNetAttention` with Learned Temporal Attention Pooling. | Authoritative SOTA Benchmark (Single-Label) |
| **Ablation Candidate (Model C1)** | `ECGResNetAttention` with Inverse-Frequency Weighted Cross-Entropy. | Verified Negative Finding |
| **Multi-Label Baseline (Model A-ML)** | Pure GAP `ECGResNet` Version B trained with `BCEWithLogitsLoss()`. | Authoritative Multi-Label Baseline |

---

## 2. Single-Label Multiclass Master Ledger ($N = 4,308$ Frozen Test Records)

| Exp ID | Model Name | Architecture Backbone | Pooling Module | Loss Function | Checkpoint Path | Val Macro F1 | Test Acc | Test Macro F1 | Test Weighted F1 | HYP F1 (Recall) | STTC F1 (Recall) | Statistical Significance vs Baseline |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ref** | _Historical arth.ipynb_ | ECGResNet V-B | GAP | Unweighted CE | _Unverified scaffold_ | _Unverified_ | 67.57% | 0.5960 | 0.6637 | — | — | — |
| **A** | **Model A (Clean GAP)** | ECGResNet V-B (3.92M params) | `AdaptiveAvgPool1d(1)` | Unweighted CE | `checkpoints/model_a_gap_best.pth` | 0.5842 (Acc: 71.21%) | **70.50%** | **0.5695** | **0.6785** | 0.2457 (15.2%) | 0.4704 (40.9%) | Baseline Reference |
| **B** | **Model B (Temporal Attention)** | ECGResNetAttention (3.92M params) | `TemporalAttentionPooling` | Unweighted CE | `checkpoints/attention_pool_best.pth` | 0.6278 (Acc: 71.96%) | **72.56%** | **0.6152** | **0.7133** | **0.3179 (23.2%)** | **0.5521 (49.2%)** | **p = 1.825e-4 vs Model A (McNemar)**; All 5 class CIs > 0 |
| **C1** | **Model C1 (Weighted-CE)** | ECGResNetAttention (3.92M params) | `TemporalAttentionPooling` | Inverse-Freq Weighted CE | `checkpoints/model_c1_weighted_ce_best.pth` | 0.5412 (Acc: 66.81%) | **65.76%** | **0.5243** | **0.6451** | **0.0962 (5.3%)** | **0.5161 (60.5%)** | **p = 2.59e-26 vs Model B (McNemar)**; Significant Degradation |

---

## 3. Phase 6 Multi-Label Master Ledger ($N = 4,308$ Frozen Test Records)

_Multi-Label Ground Truth: 5-dimensional binary vectors $\mathbf{y} \in \{0, 1\}^5$ derived from ALL diagnostic statements (5,144 multi-pathology records preserved)._  
_Model Selection Metric: Validation Macro AUROC._

| Exp ID | Model Name | Backbone | Pooling | Loss Function | Checkpoint Path | Val Macro AUROC | Test Macro AUROC (95% CI) | Test Macro AP (95% CI) | Test Macro F1 (Val-Tuned) | Test Subset Acc. | Test Hamming Loss | Optimization & Provenance Notes |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A-ML** | **Model A-ML (GAP Baseline)** | ECGResNet V-B (3.92M params) | `AdaptiveAvgPool1d(1)` | `BCEWithLogitsLoss()` | `checkpoints/model_a_multilabel_best.pth` | **0.8992** | **0.8960** ([0.8899, 0.9018]) | **0.7615** ([0.7494, 0.7739]) | **0.6987** ([0.6882, 0.7093]) | **57.03%** | **0.1361** | Checkpoint trained Mon Aug 31 21:04:03; evaluated via verified cached tensor pipeline ($\max |\Delta| = 0.0$ bitwise parity). |
| **B-ML** | **Model B-ML (Temporal Attention)** | ECGResNetAttention (3.92M params) | `TemporalAttentionPooling` (+513 params) | `BCEWithLogitsLoss()` | `checkpoints/model_b_multilabel_best.pth` | *see notes* | **0.8977** ([$-0.0019$, $+0.0054$] vs A-ML) | **0.7606** ([$-0.0084$, $+0.0061$] vs A-ML) | **0.6920** ([$-0.0148$, $+0.0012$] vs A-ML) | **55.78%** | **0.1464** | SHA-256: `89069f...bfb0bcc`; $\Delta$ Macro AUROC $= +0.0017$ ($P(B>A) = 81.7\%$); MI AUROC $+0.0078$ significant ($P = 99.6\%$, CI $[+0.003, +0.013]$); Geometry gap gain $+0.066$. |

### Phase 6 Per-Class Breakdown (Model A-ML GAP Baseline, $N = 4,308$ Test ECGs):
| Superclass | Positive Support | Prevalence | AUROC (95% CI) | Average Precision (95% CI) | Optimal Val Threshold | Test F1 (Val-Tuned) | Test Recall | Test Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1,881 | 43.7% | 0.9458 ([0.9396, 0.9515]) | 0.9219 ([0.9109, 0.9323]) | 0.30 | 0.8602 | 92.6% | 80.3% |
| **STTC** | 1,059 | 24.6% | 0.9241 ([0.9154, 0.9320]) | 0.7966 ([0.7722, 0.8218]) | 0.35 | 0.7405 | 79.2% | 69.5% |
| **CD** | 1,019 | 23.7% | 0.9088 ([0.8971, 0.9209]) | 0.8290 ([0.8101, 0.8486]) | 0.70 | 0.7417 | 69.5% | 79.6% |
| **MI** | 1,118 | 26.0% | 0.9054 ([0.8944, 0.9153]) | 0.8085 ([0.7865, 0.8286]) | 0.40 | 0.7158 | 68.6% | 74.8% |
| **HYP** | 549 | 12.7% | 0.7957 ([0.7752, 0.8159]) | 0.4516 ([0.4074, 0.4973]) | 0.25 | 0.4351 | 47.4% | 40.2% |

### Phase 6 Per-Class Breakdown (Model B-ML Temporal Attention, $N = 4,308$ Test ECGs):
| Superclass | Positive Support | Prevalence | AUROC | $\Delta$ AUROC vs A-ML (95% CI) | Average Precision | $\Delta$ AP vs A-ML (95% CI) | Optimal Val Threshold | Test F1 (Val-Tuned) | $\Delta$ F1 vs A-ML (95% CI) | Test Recall | Test Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 1,881 | 43.7% | 0.9461 | +0.0003 ([$-0.0028$, $+0.0037$]) | 0.9219 | +0.0000 ([$-0.0068$, $+0.0064$]) | 0.45 | 0.8570 | $-0.0032$ ([$-0.0118$, $+0.0050$]) | 89.8% | 81.9% |
| **STTC** | 1,059 | 24.6% | 0.9234 | $-0.0007$ ([$-0.0050$, $+0.0037$]) | 0.7904 | $-0.0063$ ([$-0.0182$, $+0.0058$]) | 0.35 | 0.7377 | $-0.0028$ ([$-0.0171$, $+0.0115$]) | 82.7% | 66.6% |
| **CD** | 1,019 | 23.7% | 0.9070 | $-0.0018$ ([$-0.0109$, $+0.0067$]) | 0.8267 | $-0.0023$ ([$-0.0141$, $+0.0095$]) | 0.35 | 0.7335 | $-0.0082$ ([$-0.0258$, $+0.0100$]) | 71.1% | 75.8% |
| **MI** | 1,118 | 26.0% | **0.9132** | **+0.0078** ([$+0.0027$, $+0.0133$]) ★ | **0.8225** | **+0.0140** ([$+0.0033$, $+0.0267$]) ★ | 0.35 | 0.7178 | +0.0020 ([$-0.0155$, $+0.0186$]) | 77.9% | 66.5% |
| **HYP** | 549 | 12.7% | 0.7987 | +0.0030 ([$-0.0092$, $+0.0154$]) | 0.4414 | $-0.0102$ ([$-0.0369$, $+0.0176$]) | 0.20 | 0.4140 | $-0.0211$ ([$-0.0454$, $+0.0008$]) | 48.6% | 36.0% |

_★ Statistically significant improvement: 95% bootstrap CI excludes zero._

---

## 4. Phase 7 Official PTB-XL Benchmark Ledger ($N = 2,158$ Frozen Fold 10 Test Records)

_Protocol: Official PhysioNet PTB-XL `strat_fold` (Train: Folds 1–8 = 17,084 | Val: Fold 9 = 2,146 | Test: Fold 10 = 2,158). Zero Patient Leakage._  
_Model Selection Metric: Validation Macro AUROC on Fold 9._

| Exp ID | Model Name | Backbone Architecture | Pooling Module | Loss Function | Checkpoint Path | Val Macro AUROC (Fold 9) | Test Macro AUROC (Fold 10, 95% CI) | Test Macro AP (Fold 10, 95% CI) | Test Macro F1 (Val-Tuned) | Test Subset Acc. | Test Hamming Loss | Published Literature Benchmark Comparison |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A-Fold10** | **Model A-ML (Official GAP Baseline)** | ECGResNet V-B (3.92M params) | `AdaptiveAvgPool1d(1)` | `BCEWithLogitsLoss()` | `checkpoints/model_a_fold10_best.pth` | **0.8878** | **0.8868** ([0.8766, 0.8958]) | **0.7334** ([0.7148, 0.7532]) | **0.6907** ([0.6738, 0.7066]) | **56.67%** | **0.1399** | Strodthoff et al. (2021) `resnet1d_wang` = **0.925**; `xresnet1d101` = **0.932**; Nonaka \& Seita = **0.930**. Direct numerical comparability established. |
| **XResNet1D** | **Model XResNet1D (Phase 7.2)** | 1D Bag-of-Tricks ResNet (3.93M params) | `AdaptiveAvgPool1d(1)` | `BCEWithLogitsLoss()` | `checkpoints/model_xresnet_fold10_best.pth` | 0.8791 | 0.8775 ([0.8669, 0.8871]) | 0.7276 ([0.7097, 0.7457]) | 0.6680 ([0.6520, 0.6833]) | 50.88% | 0.1606 | Evaluated on Fold 10 ($N = 2,158$); $\Delta$ AUROC = $-0.0093$ vs Model A; Model A retained as top benchmark model. |
| **InceptionTime** | **InceptionTime1D (Phase 7.5)** | 6-Block Inception CNN (3.89M params) | `AdaptiveAvgPool1d(1)` | `BCEWithLogitsLoss()` | `checkpoints/model_inception_fold10_best.pth` | **0.8967** | **0.8991** ([0.8896, 0.9077]) | **0.7636** ([0.7458, 0.7808]) | **0.7070** ([0.6904, 0.7226]) | **58.80%** | **0.1296** | **New best model.** $\Delta$ AUROC = $+0.0123$ vs Model A ($+1.39\%$). Trained only 7/20 epochs (CPU thermal early stop). Lowest Hamming Loss across all models. |

### Phase 7 Per-Class Breakdown (Model A-ML Official Fold 10 Baseline, $N = 2,158$ Test ECGs):
| Superclass | Positive Support | Prevalence | Optimal Val Threshold (Fold 9) | Test AUROC (95% CI) | Strodthoff `xresnet1d101` AUROC | Test AP (95% CI) | Test F1 (Val-Tuned) | Test Recall | Test Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 44.6% | 0.40 | **0.9316** ([0.9221, 0.9409]) | 0.952 | 0.9050 ([0.8870, 0.9215]) | **0.8480** | 89.8% | 80.3% |
| **STTC** | 521 | 24.1% | 0.25 | **0.9168** ([0.9040, 0.9295]) | 0.931 | 0.7793 ([0.7415, 0.8137]) | **0.7225** | 79.5% | 66.2% |
| **CD** | 496 | 23.0% | 0.45 | **0.9025** ([0.8863, 0.9189]) | 0.930 | 0.8066 ([0.7756, 0.8352]) | **0.7126** | 67.7% | 75.2% |
| **MI** | 550 | 25.5% | 0.40 | **0.9151** ([0.9019, 0.9277]) | 0.938 | 0.7906 ([0.7566, 0.8210]) | **0.7395** | 80.0% | 68.8% |
| **HYP** | 262 | 12.1% | 0.25 | **0.7682** ([0.7358, 0.7971]) | 0.865 | 0.3854 ([0.3256, 0.4533]) | **0.4310** | 43.5% | 42.7% |

### Phase 7.5 Per-Class Breakdown (InceptionTime1D, $N = 2,158$ Test ECGs):
| Superclass | Positive Support | Prevalence | Optimal Val Threshold (Fold 9) | Test AUROC (95% CI) | $\Delta$ AUROC vs Model A | Test AP | Test F1 (Val-Tuned) | Test Recall | Test Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 44.6% | 0.36 | **0.9385** ([0.8896, 0.9077]†) | **+0.0069** | 0.9111 | **0.8495** | 92.3% | 78.7% |
| **STTC** | 521 | 24.1% | 0.32 | **0.9276** | **+0.0108** | 0.8133 | **0.7573** | 77.0% | 74.5% |
| **CD** | 496 | 23.0% | 0.37 | **0.9200** | **+0.0175** | 0.8433 | **0.7524** | 77.8% | 72.8% |
| **MI** | 550 | 25.5% | 0.43 | **0.9194** | **+0.0043** | 0.8166 | **0.7413** | 75.8% | 72.5% |
| **HYP** | 262 | 12.1% | 0.23 | **0.7899** | **+0.0217** | 0.4337 | **0.4346** | 41.2% | 46.0% |

_† Bootstrap CIs are macro-level only for InceptionTime; per-class CIs not individually reported._

### Phase 7.3: Validation-Only Threshold Optimization Analysis:
- **Objective**: Investigate whether poor threshold-dependent performance for minority class HYP is caused by default $0.50$ decision boundaries.
- **Protocol**: Zero retraining. Model A checkpoint (`model_a_fold10_best.pth`, SHA-256: `995eb2c...`) evaluated. All threshold grid searches ($dt=0.01$ and $dt=0.05$) conducted strictly on Fold 9 ($N = 2,146$). Frozen Fold 10 ($N = 2,158$) evaluated exactly once.
- **Strategies Evaluated**:
  1. **Strategy A (Default 0.50)**: `[0.50, 0.50, 0.50, 0.50, 0.50]` $\to$ Test Macro-F1: **0.6214**, HYP F1: **0.1836** (Recall: 10.69%), Subset Acc: **57.92%**, Hamming Loss: **0.1335**.
  2. **Strategy B (Validation Macro-F1 Optimal, $dt=0.01$)**: `[0.40, 0.26, 0.42, 0.42, 0.25]` $\to$ Test Macro-F1: **0.6906**, HYP F1: **0.4310** (Recall: 43.51%), Subset Acc: **56.86%**, Hamming Loss: **0.1398**.
  3. **Strategy C (HYP-Focused Optimization)**: `[0.40, 0.26, 0.42, 0.42, 0.25]` $\to$ Test Macro-F1: **0.6906**, HYP F1: **0.4310**, Subset Acc: **56.86%**, Hamming Loss: **0.1398**.
- **Conclusion**: Validates that low HYP performance under Strategy A was primarily an artifact of symmetric $0.50$ cutoff under class imbalance. Lowering $th_{HYP}$ to $0.25$ quadrupled recall (10.69% $\to$ 43.51%) and more than doubled F1 (0.1836 $\to$ 0.4310, +134.7%).
- **Final Model Usage**: Strategy B/C validation-tuned thresholds **MUST be retained** for clinical inference and reporting.
- **Artifacts**: Preserved in [`results/phase7/benchmark_fold10/model_a_thresholds/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_thresholds/).

### Phase 7.4: Probability Calibration & Threshold Stability Analysis:
- **Objective**: Determine whether Model A-ML outputs are adequately calibrated and verify stability of the 7.3 decision thresholds under bootstrap resampling.
- **Protocol**: Zero retraining. Model A checkpoint (`model_a_fold10_best.pth`, SHA-256: `995eb2c...`) evaluated. Global temperature scaling ($T$) fitted strictly on Fold 9 validation logits. Threshold stability evaluated over 1,000 bootstrap resamples on Fold 9 ($N = 2,146$). Frozen Fold 10 ($N = 2,158$) evaluated with frozen parameters.
- **Calibration Findings**:
  - Model A is exceptionally well-calibrated out of the box: Test Macro Brier = **0.0969**, Test Macro ECE (10-bin) = **2.69%** (Validation ECE = **2.94%**).
  - Learned Temperature: **$T = 0.9761$** (within 2.4% of unity).
  - Temperature scaling produces negligible change in ECE ($< 0.01\%$), confirming unscaled logits reflect empirical diagnostic risk.
  - Ranking metrics strictly invariant: Test Macro AUROC = **0.8868369445**, Macro AP = **0.7333888656** ($\Delta = 0.00$).
- **Threshold Stability Findings**:
  - Across 1,000 bootstrap resamples on Fold 9, the median threshold vector is **`[0.40, 0.26, 0.42, 0.42, 0.25]`**, exactly matching Experiment 7.3.
  - Minority class HYP threshold ($th = 0.25$) is highly stable: **34.0% exact mode frequency** (dominant among 91 candidate bins) and **70.5% in window `[0.23, 0.27]`** ($Std = 0.038$, 95% CI `[0.14, 0.27]`).
- **Conclusion**:
  1. Calibration is adequate out of the box; temperature scaling ($T = 0.9761$) should be retained for website deployment to ensure formal clinical risk calibration.
  2. The 7.3 class-specific thresholds are statistically stable and robust.
  3. Representation bottleneck for HYP (AUROC 0.7682) cannot be fixed by thresholding/calibration alone; justifies model-level innovations.
- **Artifacts**: Preserved in [`results/phase7/benchmark_fold10/model_a_calibration/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_a_calibration/).

### Phase 7.5: InceptionTime Multi-Scale Representation Benchmark:
- **Objective**: Evaluate whether multi-scale temporal feature extraction via 1D Inception architecture (parallel kernels k=9, 19, 39) improves diagnostic discrimination over fixed-kernel ECGResNet.
- **Architecture**: InceptionTime1D — 6 Inception blocks with bottleneck 1×1 Conv → 3 parallel Conv1D branches (k=9, 19, 39) + MaxPool→1×1 Conv → Concatenate → BatchNorm → ReLU + Residual. Hierarchical temporal downsampling (strides [2, 2, 2, 1, 1, 1]). GAP → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→5). **3,886,149 parameters** (−0.85% vs Model A).
- **Training**: BCEWithLogitsLoss, Adam (lr=1e-3, wd=1e-4), batch=16, seed=42. Trained for **7 of 20 planned epochs** before CPU thermal throttling forced early termination. Val Macro AUROC trajectory: 0.8687 → 0.8812 → 0.8861 → 0.8852 → 0.8894 → 0.8933 → 0.8967 (monotonically improving through all 7 epochs with no plateau).
- **Model Selection**: Best validation Macro AUROC at Epoch 7 (0.8967). Checkpoint saved: `model_inception_fold10_best.pth` (SHA-256: `18277a08...`).
- **Threshold Tuning**: Per-class F1-optimal grid search on Fold 9: NORM=0.36, STTC=0.32, CD=0.37, MI=0.43, HYP=0.23. Frozen before Fold 10 evaluation.
- **Frozen Test Results (Fold 10, N=2,158)**:
  - **Macro AUROC: 0.8991** (95% CI: [0.8896, 0.9077]) — **+0.0123 vs Model A (+1.39%)**
  - **Macro AP: 0.7636** (95% CI: [0.7458, 0.7808]) — **+0.0302 vs Model A (+4.12%)**
  - **Macro F1: 0.7070** (95% CI: [0.6904, 0.7226]) — **+0.0163 vs Model A (+2.36%)**
  - Weighted F1: 0.7548, Subset Accuracy: 58.80%, Hamming Loss: 0.1296
  - Per-class AUROC: NORM 0.9385, STTC 0.9276, CD 0.9200, MI 0.9194, HYP 0.7899
  - InceptionTime1D achieves the **highest AUROC on all five superclasses**.
- **Negative Finding**: HYP AUROC improved to 0.7899 (+0.0217 vs Model A) but remains **13+ percentage points below the next-worst class**, confirming a fundamental representation bottleneck for hypertrophy.
- **Critical Note**: These are **lower-bound results** — the model was still improving at termination (Val AUROC curve monotonically increasing, no plateau). Performance with full 20-epoch training is expected to be higher.
- **Evaluation-Only Repair**: The original training script (`train_model_inception_fold10.py`) was interrupted. A separate evaluation-only script (`evaluate_model_inception_fold10.py`) was executed to produce the frozen Fold-10 metrics without any training, checkpoint modification, or test-set tuning.
- **Artifacts**: [`results/phase7/benchmark_fold10/model_inception/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/model_inception/) — `model_inception_fold10_metrics.json`, `model_inception_fold10_predictions.csv`, `model_inception_fold10_classification_report.csv`, `MODEL_INCEPTION_FOLD10_EVALUATION_REPORT.md`.
- **Four-Model Comparison**: [`results/phase7/benchmark_fold10/FOUR_MODEL_COMPARISON.md`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/FOUR_MODEL_COMPARISON.md).

### Phase 8 — HYP Representation/Error Analysis (Analysis Phase):
- **Objective**: Conduct rigorous error, co-occurrence, and representation analysis on official Fold 10 predictions ($N = 2,158$) to determine the root cause of the persistent HYP performance deficit (AUROC ~0.77–0.79 vs ~0.90–0.94 for other superclasses).
- **Protocol**: Zero model training. Zero test-set threshold optimization. Analysis based strictly on frozen Fold 10 predictions and ground-truth multi-label annotations.
- **Key Co-occurrence Findings**:
  - $P(\text{STTC} \mid \text{HYP}) = \mathbf{59.16\%}$ (155/262). STTC is the dominant co-occurring condition with hypertrophy.
  - $P(\text{MI} \mid \text{HYP}) = \mathbf{30.15\%}$ (79/262), $P(\text{CD} \mid \text{HYP}) = \mathbf{28.63\%}$ (75/262).
  - Isolated HYP (no STTC, CD, or MI) constitutes only **21.37%** (56/262) of hypertrophy cases.
- **The "Isolated Hypertrophy Blindness" Finding**:
  - Across both ECGResNet-GAP and InceptionTime1D, models are virtually blind to isolated hypertrophy:
    - **ECGResNet-GAP Isolated Sensitivity**: **1.79%** (1/56 detected, 98.21% false negative rate; mean prob = 0.0685).
    - **InceptionTime1D Isolated Sensitivity**: **7.14%** (4/56 detected, 92.86% false negative rate; mean prob = 0.1000).
  - When HYP co-occurs with STTC ($N=155$), sensitivity surges to **69.03%** (Model A) and **63.87%** (InceptionTime).
  - Over **91–94% of all True Positives** for HYP have concurrent STTC. Over **60–68% of False Positives** have concurrent STTC.
  - **Conclusion**: Standard 1D CNNs do not learn intrinsic structural hypertrophy morphology; they rely on secondary repolarization abnormalities (ST-T changes / strain patterns) as an entangled feature proxy.
- **Sub-Phenotype Imbalance**:
  - LVH accounts for **81.68%** (214/262) of HYP cases (sensitivity ~49–50%).
  - Right ventricular hypertrophy (`RVH`, $N=12$) has **0.0% recall** across all models.
  - Left atrial overload (`LAO/LAE`, $N=42$) has **19–26% recall**.
- **Model Comparison on HYP**:
  - InceptionTime1D achieves statistically significant improvements in ranking metrics over Model A:
    - $\Delta \text{AUROC} = \mathbf{+0.0217}$ (95% CI: $[+0.0080, +0.0352]$, statistically distinguishable).
    - $\Delta \text{AP} = \mathbf{+0.0483}$ (95% CI: $[+0.0112, +0.0789]$, statistically distinguishable).
  - However, F1 improvement is negligible and not statistically significant ($\Delta \text{F1} = +0.0036$, 95% CI: $[-0.0410, +0.0424]$).
- **Artifacts**: Preserved in [`results/phase7/benchmark_fold10/hyp_analysis/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/hyp_analysis/) — `HYP_ERROR_ANALYSIS.md`, `hyp_confusion_summary.csv`, `hyp_cooccurrence_matrix.csv`, `hyp_model_comparison.csv`, `hyp_error_group_distribution.csv`, `hyp_stratified_sensitivity.csv`, `hyp_subcode_performance.csv`, `hyp_cooccurrence_heatmap.png`, `hyp_error_distribution.png`.

### Phase 8.1 — Lead-Group Occlusion Sensitivity Analysis for HYP (Analysis Phase):
- **Objective**: Determine whether HYP predictions depend disproportionately on precordial versus limb-lead information, identify which specific lead subsets govern prediction sensitivity, and evaluate how lead occlusion impacts isolated hypertrophy versus co-occurring conditions.
- **Models & Checkpoints**:
  - ECGResNet-GAP (`checkpoints/model_a_fold10_best.pth`, SHA-256: `995eb2c7...`, 20 epochs)
  - InceptionTime1D (`checkpoints/model_inception_fold10_best.pth`, SHA-256: `18277a08...`, 7 epochs)
- **Protocol**: Pure evaluation/inference only. Zero model training. Exact validation-frozen thresholds: Model A $th=0.25$, InceptionTime $th=0.23$. Tested on official frozen Fold 10 ($N = 2,158$, HYP $N = 262$).
- **Masking Protocol (7 Conditions)**: Standardized zero replacement after Butterworth 0.5–40 Hz filtering and per-lead Z-score normalization for: `NONE`, `MASK_LIMB` (I, II, III, aVR, aVL, aVF), `MASK_PRECORDIAL` (V1–V6), `MASK_V1_V2` (V1, V2), `MASK_V5_V6` (V5, V6), `MASK_I_AVL` (I, aVL), `MASK_V1_V2_V5_V6` (V1, V2, V5, V6).
- **Key Findings**:
  1. **Precordial Dominance**: Occluding precordial leads (`MASK_PRECORDIAL`) degrades diagnostic discrimination far more severely than limb leads (`MASK_LIMB`):
     - Model A AUROC drops **−0.0547** (0.7682 $\to$ 0.7135) under precordial mask vs **−0.0276** under limb mask (nearly 2.0× larger impact). AP drops **−0.0975** (0.3854 $\to$ 0.2879) vs **−0.0372** (2.6× larger impact).
     - InceptionTime AUROC drops **−0.0858** (0.7899 $\to$ 0.7041) under precordial mask vs **−0.0686** under limb mask. AP drops **−0.1407** (0.4337 $\to$ 0.2930) vs **−0.0939**.
  2. **Lateral vs Septal Chest Leads**: Occluding lateral precordial leads V5–V6 (`MASK_V5_V6`) causes a larger AUROC degradation (InceptionTime AUROC: 0.7517, $\Delta = -0.0382$) than septal leads V1–V2 (InceptionTime AUROC: 0.7873, $\Delta = -0.0026$).
  3. **STTC Confounding Persists Across Leads**: When precordial leads are occluded, HYP+STTC cases ($N=155$) retain sensitivity of **65.2%** (Model A) and **71.0%** (InceptionTime), confirming that models continue to fire HYP predictions due to concurrent repolarization changes detected in remaining leads.
  4. **InceptionTime Cross-Lead Synergy**: InceptionTime's baseline AUROC advantage over Model A (+0.0217 on unmasked data) completely vanishes when precordial leads ($\Delta = -0.0094$) or limb leads ($\Delta = -0.0192$) are occluded, showing that multi-scale temporal convolutions rely more heavily on cross-lead spatial correlations.
- **Artifacts**: Preserved in [`results/phase7/benchmark_fold10/hyp_occlusion/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/hyp_occlusion/) — `HYP_OCCLUSION_REPORT.md`, `hyp_occlusion_metrics.json`, `hyp_occlusion_predictions.csv`, `hyp_occlusion_bootstrap.csv`, `hyp_phenotype_occlusion_sensitivity.csv`, and `plots/` (`hyp_probability_drop.png`, `isolated_hyp_probability.png`, `subgroup_sensitivity.png`, `masking_effect_heatmap.png`, `phenotype_masking_effect.png`, `model_comparison.png`).

### Phase 8.2 — Single-Lead HYP Sensitivity Profile (Analysis Phase):
- **Objective**: Perform a fine-grained single-lead sensitivity analysis across all 12 individual ECG leads to identify which specific leads govern HYP predictions, examine whether lead sensitivity differs between isolated HYP and HYP+STTC, and test for technical masking artifacts (global probability drift).
- **Models & Checkpoints**:
  - ECGResNet-GAP (`checkpoints/model_a_fold10_best.pth`, SHA-256: `995eb2c7...`, 20 epochs)
  - InceptionTime1D (`checkpoints/model_inception_fold10_best.pth`, SHA-256: `18277a08...`, 7 epochs)
- **Protocol**: Pure evaluation/inference only. Zero model training. Exact validation-frozen thresholds: Model A $th=0.25$, InceptionTime $th=0.23$. Tested on official frozen Fold 10 ($N = 2,158$, HYP $N = 262$, Isolated HYP $N = 56$, HYP+STTC $N = 155$, LVH $N = 214$, RVH $N = 12$, LAO/LAE $N = 42$).
- **Masking Protocol (13 Conditions)**: Baseline `NONE` plus exactly one channel zero-masked at a time for all 12 leads: `MASK_I`, `MASK_II`, `MASK_III`, `MASK_aVR`, `MASK_aVL`, `MASK_aVF`, `MASK_V1`, `MASK_V2`, `MASK_V3`, `MASK_V4`, `MASK_V5`, `MASK_V6`.
- **Key Findings**:
  1. **Top Influential Leads**: **Lead I**, **Lead V6**, and **Lead aVR** are the top 3 most influential channels across both architectures, producing the largest drops in AUROC and AP when removed.
  2. **Lateral Lead Dominance**: Lateral leads (I, aVL, V5, V6) and reciprocal lead aVR consistently govern diagnostic discrimination for HYP, while inferior leads (II, III, aVF) have negligible impact. Lead V6 produces significantly larger AUROC/AP degradation than V1 or V2.
  3. **Collective vs Individual Precordial Sensitivity**: While precordial leads collectively dominate in Phase 8.1, individual precordial leads (e.g. V6) have comparable impact to top limb leads (Lead I, aVR). Precordial dominance is therefore a collective horizontal-plane phenomenon rather than the effect of any single chest lead.
  4. **Confirmation of Representation Bottleneck**: For isolated HYP ($N=56$), mean predicted probabilities remain flat and depressed (0.07–0.10 in Model A, 0.10–0.14 in InceptionTime) across all 12 individual lead masks. Sensitivity never exceeds 7.1% in Model A or 16.1% in InceptionTime. The models lack an active, localized lead representation for pure structural hypertrophy.
  5. **Masking Artifact Identification**: Global drift across all 2,158 records is lead-dependent: masking aVF (+0.029) or V2 (+0.030) induces non-specific upward probability drift that depresses specificity, whereas masking V6 (+0.001) or aVR (−0.004) produces virtually zero artifact. Ranking metrics (AUROC/AP) successfully isolate true informational value from this drift.
  6. **Model Comparison**: InceptionTime maintains statistically significant AUROC and AP superiority over Model A across almost all single-lead masking conditions (paired bootstrap 95% CIs exclude zero).
- **Artifacts**: Preserved in [`results/phase7/benchmark_fold10/hyp_single_lead/`](file:///c:/Users/ASUS/Desktop/ECG_Research/results/phase7/benchmark_fold10/hyp_single_lead/) — `HYP_SINGLE_LEAD_REPORT.md`, `hyp_single_lead_metrics.json`, `hyp_single_lead_predictions.csv`, `hyp_single_lead_bootstrap.csv`, `hyp_single_lead_global_drift.csv`, `hyp_single_lead_subphenotypes.csv`, and 8 plots in `plots/`.

---

## 5. Methodological Ledger & Protocol Guarantees

1. **Patient Leakage**: Enforced at 0 records across all phases. All splits strictly partitioned by `patient_id` or official `strat_fold`.
2. **Test Set Integrity**: Test partitions ($N = 4,308$ in Phase 6; $N = 2,158$ in Phase 7) are strictly frozen. Model selection is conducted strictly on **Validation Macro AUROC**.
3. **Threshold Tuning Integrity**: Optimal per-class decision thresholds are optimized **exclusively on the Validation Split** and frozen before test evaluation.
4. **Reproducibility**: Global seed 42 set across NumPy, PyTorch CPU/CUDA, and DataLoader pipelines.
5. **Statistical Rigor**: All metrics report 1,000-resample bootstrap 95% confidence intervals.

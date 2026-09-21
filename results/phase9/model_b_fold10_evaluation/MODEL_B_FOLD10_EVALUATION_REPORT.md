# Phase 9.2 — Official Fold-10 Evaluation Report: ECGResNet-Attention

_Generated: 2026-09-04 22:57:07_  
_Architecture: ECGResNetAttention (TemporalAttentionPooling)_  
_Dataset: PTB-XL v1.0.3 (Official strat_fold Protocol)_  
_Test Partition: Fold 10 ($N = 2,158$ frozen records, zero patient overlap)_  

---

## 1. Executive Summary

This report provides the **official Fold-10 benchmark evaluation** of **ECGResNet-Attention (Model B)**. The model was evaluated using the newly trained checkpoint (`checkpoints/model_b_fold10_best.pth`), establishing a fully controlled, apples-to-apples evaluation under the official PTB-XL protocol.

### Checkpoint & Provenance Verification:
- **Checkpoint Path**: `C:\Users\ASUS\Desktop\ECG_Research\checkpoints\model_b_fold10_best.pth`
- **SHA-256 Checksum**: `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa`
- **Load Status**: `strict=True` (Verification PASSED)
- **Total Parameters**: 3,920,006
- **Test Set N**: 2,158 ECGs (Folds 1–8 Train, Fold 9 Val, Fold 10 Frozen Test)
- **Patient Overlap**: Exactly **0** shared patients across partitions
- **Optimization in Evaluation**: **NONE** (`model.eval()`, `torch.no_grad()`, zero parameter updates)
- **Threshold Optimization**: Tuned strictly on Fold 9 validation predictions; **Fold 10 labels were never used for threshold tuning**

---

## 2. Benchmark Results on Fold 10 ($N = 2,158$)

### A. Primary Summary Metrics (with 1,000-Resample 95% Bootstrap CIs)

| Metric | Fold 10 Score | 95% Bootstrap Confidence Interval | Metric Category |
|:---|:---:|:---:|:---|
| **Macro AUROC** | **0.8979** | `[0.8886, 0.9062]` | Ranking (Continuous Probabilities) |
| **Macro AP (PR-AUC)** | **0.7644** | `[0.7477, 0.7807]` | Ranking (Continuous Probabilities) |
| **Macro F1 (Val-Tuned)** | **0.7052** | `[0.6887, 0.7203]` | Threshold-Dependent (Fold-9 Tuned) |
| **Macro F1 (Default 0.5)** | **0.6830** | — | Threshold-Dependent (Standard 0.5) |
| **Weighted F1** | **0.7507** | `[0.7371, 0.7638]` | Threshold-Dependent (Prevalence Weighted) |
| **Subset Accuracy** | **58.71%** | `[56.72%, 60.66%]` | Exact Match Multi-Label |
| **Hamming Loss** | **0.1338** | `[0.1266, 0.1411]` | Average Per-Label Error Rate |

---

## 3. Per-Class Diagnostic Performance on Fold 10

Thresholds were optimized exclusively on Fold 9 validation predictions and frozen before test evaluation:

| Superclass | Support | Optimal Val Thresh | AUROC (95% CI) | AP (95% CI) | F1 (95% CI) | Recall | Precision | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM** | 963 | 0.47 | **0.9408** `[0.9312, 0.9495]` | 0.9198 `[0.9060, 0.9332]` | 0.8505 `[0.8359, 0.8656]` | 94.5% | 77.3% | 77.7% |
| **STTC** | 521 | 0.36 | **0.9284** `[0.9156, 0.9396]` | 0.8226 `[0.7902, 0.8525]` | 0.7476 `[0.7155, 0.7756]` | 75.6% | 73.9% | 91.5% |
| **CD** | 496 | 0.38 | **0.9060** `[0.8906, 0.9223]` | 0.8210 `[0.7929, 0.8513]` | 0.7363 `[0.7049, 0.7699]` | 69.0% | 79.0% | 94.5% |
| **MI** | 550 | 0.34 | **0.9190** `[0.9060, 0.9307]` | 0.8167 `[0.7857, 0.8448]` | 0.7295 `[0.6980, 0.7574]` | 70.4% | 75.7% | 92.3% |
| **HYP** | 262 | 0.26 | **0.7951** `[0.7647, 0.8237]` | 0.4417 `[0.3835, 0.5029]` | 0.4620 `[0.4106, 0.5101]` | 53.4% | 40.7% | 89.2% |

---

## 4. Computational & Inference Complexity

- **Parameter Count**: 3,920,006
- **Fold-10 Inference Runtime**: 34.49 s
- **Latency per ECG**: 15.98 ms
- **Throughput**: 62.6 ECGs/sec
- **Hardware/Environment**: CPU (PyTorch 14 threads)

---

## 5. Protocol Integrity & Compliance Audit

1. **Test Set Isolation**: Fold 10 was strictly kept frozen until this post-training evaluation. Zero training updates occurred.
2. **Threshold Isolation**: Thresholds were selected purely on Fold 9 validation predictions and frozen before Fold 10 inference.
3. **Reproducibility**: Checkpoint SHA-256 `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa` verified against the training artifact.

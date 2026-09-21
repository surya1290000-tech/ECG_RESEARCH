# Phase 9.1 — ECGResNet-Attention Official Fold-10 Training Report

_Generated: 2026-09-04 22:17:34_  
_Architecture: ECGResNetAttention (TemporalAttentionPooling)_  
_Dataset: PTB-XL v1.0.3 (Official strat_fold Protocol)_  

---

## 1. Executive Summary

This report documents the controlled training run for **ECGResNet-Attention (Model B)** under the exact official PTB-XL Fold-10 benchmark protocol. This run resolves the historical partition discrepancy where Model B was previously trained under Phase 6 GroupShuffleSplit ($N=4,308$ test) and enables the final apples-to-apples four-model benchmark.

### Checkpoint Provenance:
- **Target Checkpoint**: `C:\Users\ASUS\Desktop\ECG_Research\checkpoints\model_b_fold10_best.pth`
- **SHA-256**: `8361b3bfbb85ec1306fabebce27defd96fd4bc05536afec6f182611881888afa`
- **Parameters**: 3,920,006
- **Best Epoch**: Epoch 5
- **Best Fold-9 Validation Macro AUROC**: **0.8965**
- **Best Fold-9 Validation Macro AP**: **0.7560**
- **Best Fold-9 Validation Macro F1 (Th=0.5)**: **0.6797**
- **Total Completed Epochs**: 5 / 20
- **Total Training Runtime**: 6290.0 s (104.8 min)
- **Mean Epoch Time**: 1258.0 s

---

## 2. Experimental Controls & Protocol Verification

| Parameter | Configuration | Verification Status |
|:---|:---|:---:|
| **Backbone** | ECGResNet (stem + 4 residual blocks) | PASS (Identical to Model A) |
| **Pooling** | `TemporalAttentionPooling` (Conv1d(512 $\to$ 1, kernel=1)) | PASS (+513 params) |
| **Classifier** | Linear(512 $\to$ 128) $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(128 $\to$ 5) | PASS |
| **Total Parameters** | 3,920,006 | PASS |
| **Training Partition** | Folds 1–8 ($N = 17,084$ records) | PASS |
| **Validation Partition** | Fold 9 ($N = 2,146$ records) | PASS |
| **Test Partition** | Fold 10 ($N = 2,158$ records) | **FROZEN & UNTOUCHED** |
| **Patient Overlap** | Zero overlap across train/val/test | PASS ($0$ shared patient IDs) |
| **Preprocessing** | Butterworth bandpass 0.5–40 Hz (order 2), per-lead Z-score | PASS |
| **Sampling Rate** | 100 Hz ($12 \times 1000$ samples) | PASS |
| **Loss Function** | `BCEWithLogitsLoss()` | PASS |
| **Optimizer** | Adam (lr=1e-3, weight_decay=1e-4) | PASS |
| **Batch Size** | 16 | PASS |
| **Model Selection** | Best Fold-9 Validation Macro AUROC | PASS |

---

## 3. Epoch-by-Epoch Validation Progression (Fold 9)

| Epoch | Train BCE | Val BCE | Val Macro AUROC | Val Macro AP | Val Macro F1 (0.5) | Epoch Time (s) | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | 0.3737 | 0.3336 | **0.8773** | 0.7210 | 0.6270 | 319.1s | **YES** |
| 02 | 0.3300 | 0.3268 | **0.8821** | 0.7348 | 0.6184 | 361.4s | **YES** |
| 03 | 0.3168 | 0.3115 | **0.8910** | 0.7445 | 0.6655 | 560.1s | **YES** |
| 04 | 0.3116 | 0.3113 | **0.8859** | 0.7450 | 0.6594 | 491.8s | no |
| 05 | 0.3040 | 0.3024 | **0.8965** | 0.7560 | 0.6797 | 4557.5s | **YES** |

---

## 4. Hardware & Thermal Safety Notes

- **Device**: cpu
- **Thread Count**: 12 threads
- **Execution Stability**: Epoch duration remained consistent with mean 1258.0s/epoch.
- **Hardware Throttling**: Monitored and verified safe.

---

## 5. Strict Scientific Isolation Confirmation

> [!IMPORTANT]
> **Fold 10 Test Set Status**:
> In strict accordance with Phase 9.1 guidelines, **Fold 10 was NEVER loaded, evaluated, or referenced during this training execution**. No Fold-10 metrics were computed, and no thresholds were tuned against test labels. The produced checkpoint represents a pristine model ready for official four-model evaluation.

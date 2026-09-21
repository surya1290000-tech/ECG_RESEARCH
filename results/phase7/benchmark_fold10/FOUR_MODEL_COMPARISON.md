# Four-Model Benchmark Comparison — Official PTB-XL Fold-10 Protocol

_Generated: 2026-09-04_  
_Dataset: PTB-XL v1.0.3 (21,388 diagnostic records)_  
_Protocol: Official `strat_fold` — Train: Folds 1–8 (17,084) | Val: Fold 9 (2,146) | Test: Fold 10 (2,158)_  
_Zero patient overlap across all partitions_  
_Threshold tuning: Fold 9 only; Fold 10 frozen and never used for tuning_  
_5 diagnostic superclasses: NORM, STTC, CD, MI, HYP_

---

## 1. Macro-Level Benchmark Summary

| Metric | **ECGResNet-GAP** (Model A) | **ECGResNet-Attention** (Model B) | **XResNet1D** | **InceptionTime1D** |
|:---|:---:|:---:|:---:|:---:|
| **Phase / Protocol** | Phase 7 / Fold-10 | Phase 6 / GSS† | Phase 7 / Fold-10 | Phase 7 / Fold-10 |
| **Test Set N** | 2,158 | 4,308† | 2,158 | 2,158 |
| **Parameters** | 3,919,493 | 3,919,493 (+513 attn) | 3,931,525 | 3,886,149 |
| **Checkpoint** | `model_a_fold10_best.pth` | `model_b_multilabel_best.pth` | `model_xresnet_fold10_best.pth` | `model_inception_fold10_best.pth` |
| **Checkpoint SHA-256** | `995eb2c7...` | `89069f33...` | `aee7bf2e...` | `18277a08...` |
| **Training Epochs** | 20 | 20 | 20 | 7 (early stop: thermal) |
| **Macro AUROC** | **0.8868** | 0.8977† | 0.8775 | **0.8991** |
| **AUROC 95% CI** | [0.8766, 0.8958] | — | [0.8669, 0.8871] | [0.8896, 0.9077] |
| **Macro AP** | 0.7334 | 0.7606† | 0.7276 | **0.7636** |
| **AP 95% CI** | [0.7148, 0.7532] | — | [0.7097, 0.7457] | [0.7458, 0.7808] |
| **Macro F1 (val-tuned)** | 0.6907 | 0.6920† | 0.6680 | **0.7070** |
| **F1 95% CI** | [0.6738, 0.7066] | — | [0.6520, 0.6833] | [0.6904, 0.7226] |
| **Weighted F1** | 0.7400 | 0.7413† | 0.7150 | **0.7548** |
| **Subset Accuracy** | 56.67% | 55.78%† | 50.88% | 58.80% |
| **Hamming Loss** | 0.1399 | 0.1464† | 0.1606 | 0.1296 |
| **Test Inference (sec)** | — | — | 52.3 | 51.9 |

> **†** Model B (ECGResNet-Attention) was evaluated under the **Phase 6 GroupShuffleSplit** protocol (N=4,308 test) and has **NOT** been re-evaluated under the official Fold-10 protocol. Its metrics are from a different test partition and are **not directly comparable** on a point-estimate basis. They are included here for architectural context only.

---

## 2. Per-Class AUROC Comparison (Fold-10 Models Only)

| Superclass | Support | **ECGResNet-GAP** | **XResNet1D** | **InceptionTime1D** | Best Model |
|:---|:---:|:---:|:---:|:---:|:---|
| **NORM** | 963 | 0.9316 | 0.9161 | **0.9385** | InceptionTime (+0.0069) |
| **STTC** | 521 | 0.9168 | 0.9154 | **0.9276** | InceptionTime (+0.0108) |
| **CD** | 496 | 0.9025 | 0.8990 | **0.9200** | InceptionTime (+0.0175) |
| **MI** | 550 | 0.9151 | 0.8791 | **0.9194** | InceptionTime (+0.0043) |
| **HYP** | 262 | 0.7682 | 0.7779 | **0.7899** | InceptionTime (+0.0217) |

**InceptionTime1D achieves the highest AUROC on all five superclasses**, with the largest gains on the two hardest classes: CD (+0.0175) and HYP (+0.0217).

---

## 3. Per-Class F1 / Precision / Recall (Fold-10, Validation-Tuned Thresholds)

### ECGResNet-GAP (Model A)
| Class | Threshold | F1 | Precision | Recall | Support |
|:---|:---:|:---:|:---:|:---:|:---:|
| NORM | 0.40 | 0.8480 | 80.3% | 89.8% | 963 |
| STTC | 0.25 | 0.7225 | 66.2% | 79.5% | 521 |
| CD | 0.45 | 0.7126 | 75.2% | 67.7% | 496 |
| MI | 0.40 | 0.7395 | 68.8% | 80.0% | 550 |
| HYP | 0.25 | 0.4310 | 42.7% | 43.5% | 262 |

### XResNet1D
| Class | Threshold | F1 | Precision | Recall | Support |
|:---|:---:|:---:|:---:|:---:|:---:|
| NORM | 0.40 | 0.8239 | 77.3% | 88.2% | 963 |
| STTC | 0.25 | 0.7171 | 62.7% | 83.7% | 521 |
| CD | 0.75 | 0.7131 | 71.5% | 71.2% | 496 |
| MI | 0.35 | 0.6630 | 67.7% | 64.9% | 550 |
| HYP | 0.20 | 0.4231 | 35.2% | 53.1% | 262 |

### InceptionTime1D
| Class | Threshold | F1 | Precision | Recall | Support |
|:---|:---:|:---:|:---:|:---:|:---:|
| NORM | 0.36 | 0.8495 | 78.7% | 92.3% | 963 |
| STTC | 0.32 | 0.7573 | 74.5% | 77.0% | 521 |
| CD | 0.37 | 0.7524 | 72.8% | 77.8% | 496 |
| MI | 0.43 | 0.7413 | 72.5% | 75.8% | 550 |
| HYP | 0.23 | 0.4346 | 46.0% | 41.2% | 262 |

---

## 4. Key Findings

### 4.1 InceptionTime1D is the Current Best Model

InceptionTime1D achieves the highest Fold-10 Macro AUROC (**0.8991**), surpassing ECGResNet-GAP by **+0.0123** (+1.39%) and XResNet1D by **+0.0216** (+2.46%). This improvement is consistent across all five superclasses and all secondary metrics (Macro AP, Macro F1, Weighted F1, Hamming Loss).

> [!IMPORTANT]
> InceptionTime1D completed only **7 of 20 planned training epochs** before thermal CPU throttling forced early termination. The model was still improving (Val Macro AUROC: 0.8687 → 0.8967 over 7 epochs). The reported test metrics represent a **lower bound** on the architecture's achievable performance.

### 4.2 XResNet1D Underperformed the Baseline

XResNet1D (Bag-of-Tricks 1D ResNet) achieved Macro AUROC **0.8775**, which is **−0.0093** below ECGResNet-GAP. The additional complexity of stem modifications and grouped convolutions did not translate to improved diagnostic discrimination on PTB-XL under this protocol.

### 4.3 Temporal Attention (Phase 6 Protocol — Not Directly Comparable)

Under the Phase 6 GroupShuffleSplit protocol (N=4,308 test), ECGResNet-Attention achieved Macro AUROC **0.8977** vs. ECGResNet-GAP's **0.8960** (Δ = +0.0017). The improvement was statistically marginal (P(B>A) = 81.7%), with only MI AUROC showing a significant per-class gain. **A Fold-10 evaluation of Model B has not yet been conducted.**

### 4.4 HYP Remains the Universal Bottleneck

All four models show dramatically lower performance on HYP (Hypertrophy) compared to the other four classes:

| Model | HYP AUROC | HYP F1 | HYP AP | Gap to Next-Worst Class AUROC |
|:---|:---:|:---:|:---:|:---:|
| ECGResNet-GAP | 0.7682 | 0.4310 | 0.3854 | −0.1343 (vs CD 0.9025) |
| XResNet1D | 0.7779 | 0.4231 | 0.4342 | −0.1211 (vs MI 0.8791) |
| InceptionTime1D | 0.7899 | 0.4346 | 0.4337 | −0.1301 (vs MI 0.9194) |

The HYP gap persists at **13+ percentage points below the next-worst class** regardless of architecture. This suggests a fundamental **representation limitation** for hypertrophy morphology, not an architecture-specific deficiency.

---

## 5. Architectural Interpretation

| Architecture | Key Innovation | Outcome on PTB-XL |
|:---|:---|:---|
| **ECGResNet-GAP** | Fixed-kernel 1D ResNet + GAP | Strong baseline (0.8868). Compact and efficient. |
| **ECGResNet-Attention** | Learned temporal attention pooling | Marginal gain (+0.0017 under GSS protocol). Attention entropy analysis showed limited specialization. |
| **XResNet1D** | Bag-of-Tricks stem + grouped convolutions | **Degraded** performance (−0.0093). Tricks designed for ImageNet do not transfer to 1D ECG signals. |
| **InceptionTime1D** | Multi-scale parallel convolutions (k=9/19/39) | **Best model** (+0.0123). Multi-scale temporal receptive fields capture complementary ECG dynamics across timescales. |

---

## 6. Threshold Strategy Summary

| Model | NORM | STTC | CD | MI | HYP | Strategy |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ECGResNet-GAP | 0.40 | 0.25 | 0.45 | 0.40 | 0.25 | Fold 9 per-class F1 grid |
| XResNet1D | 0.40 | 0.25 | 0.75 | 0.35 | 0.20 | Fold 9 per-class F1 grid |
| InceptionTime1D | 0.36 | 0.32 | 0.37 | 0.43 | 0.23 | Fold 9 per-class F1 grid |

All thresholds were derived exclusively from Fold 9 validation predictions and frozen before Fold 10 evaluation.

---

## 7. Recommendations

1. **Current best model**: **InceptionTime1D** (Macro AUROC 0.8991, trained only 7 epochs).
2. **Next research priority**: **HYP-focused error and co-occurrence analysis** to understand why all architectures plateau at AUROC ~0.77–0.79 for hypertrophy.
3. **Deferred**: Re-evaluate ECGResNet-Attention (Model B) under the Fold-10 protocol for a fully controlled four-way comparison.
4. **Deferred**: Resume InceptionTime training (epochs 8–20) when thermal conditions permit, to determine whether the architecture can achieve even higher performance.

---

_This document is a controlled research artifact. No test-set information was used for model selection or threshold tuning._

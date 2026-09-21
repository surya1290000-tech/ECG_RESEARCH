# Official Four-Model Fold-10 Benchmark Report: Controlled Architecture Comparison

_Generated: 2026-09-04 23:11:29_  
_Dataset: PTB-XL v1.0.3 (Official strat_fold Protocol)_  
_Test Partition: Fold 10 ($N = 2,158$ frozen records, zero patient overlap)_  
_Threshold Strategy: Optimized strictly on Fold 9 validation predictions and frozen prior to test inference_  
_Bootstrap Methodology: 1,000 paired resamples (seed = 42) across identical ECG records_  

---

## 1. Executive Summary & Core Research Findings

This document consolidates the official, apples-to-apples benchmark of all four principal 12-lead ECG architectures under the standardized PTB-XL Fold-10 protocol. Following the training and official evaluation of **ECGResNet-Attention (Model B)** in Phase 9.1 and 9.2, all models now share the exact same training partition (Folds 1–8, $N=17,084$), validation partition (Fold 9, $N=2,146$), and frozen test partition (Fold 10, $N=2,158$).

### Core Research Findings:
1. **Temporal Attention vs. Global Average Pooling (The Central Question)**:
   - Temporal attention pooling provides a **statistically significant and unequivocal improvement** over global average pooling on the compact ECGResNet backbone:
     - **$\Delta$ Macro AUROC**: **+0.0110** (95% CI: `[+0.0070, +0.0152]`, $P(B > A) = \mathbf{100.0\%}$)
     - **$\Delta$ Macro AP**: **+0.0306** (95% CI: `[+0.0205, +0.0420]`, $P(B > A) = \mathbf{100.0\%}$)
     - **$\Delta$ Macro F1**: **+0.0145** (95% CI: `[+0.0032, +0.0264]`, $P(B > A) = \mathbf{99.6\%}$)
   - In all three primary metrics, the 95% bootstrap confidence interval strictly excludes zero.

2. **InceptionTime vs. ECGResNet-Attention**:
   - InceptionTime1D and ECGResNet-Attention are **statistically indistinguishable** on the primary discrimination metrics:
     - **$\Delta$ Macro AUROC**: +0.0011 in favor of InceptionTime (95% CI: `[-0.0038, +0.0055]`, $P = 68.5\%$) $	o$ **Inconclusive / Statistically Equivalent**
     - **$\Delta$ Macro AP**: −0.0010 in favor of Attention (95% CI: `[-0.0108, +0.0087]`, $P = 40.5\%$) $	o$ **Attention slightly higher; Statistically Equivalent**
     - **$\Delta$ Macro F1**: +0.0016 in favor of InceptionTime (95% CI: `[-0.0099, +0.0134]`, $P = 61.6\%$) $	o$ **Equivalent**
   - Temporal attention pooling successfully closes the performance gap between the compact single-scale ResNet backbone and the multi-scale InceptionTime architecture.

3. **XResNet1D Underperformed**:
   - XResNet1D underperformed both ECGResNet baselines and InceptionTime, showing that 2D ImageNet "bag-of-tricks" (such as multi-layer stems and anti-aliased pooling) do not transfer effectively to raw 1D ECG waveforms.

---

## 2. Publication-Ready Consolidated Benchmark Table

| Model | Parameters | Macro AUROC (95% CI) | Macro AP (95% CI) | Macro F1 (Val-Tuned) | Weighted F1 | Subset Accuracy | Hamming Loss | Inference Time (Fold 10) | Latency / ECG |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **InceptionTime1D** | **3,886,149** | **0.8991** `[0.8896, 0.9077]` | 0.7636 `[0.7458, 0.7808]` | **0.7070** `[0.6904, 0.7226]` | **0.7548** | **58.80%** | **0.1296** | 51.9 s | 24.1 ms |
| **ECGResNet-Attention** | 3,920,006 | **0.8979** `[0.8886, 0.9062]` | **0.7644** `[0.7477, 0.7807]` | **0.7052** `[0.6887, 0.7203]` | 0.7507 | 58.71% | 0.1338 | **34.5 s** | **16.0 ms** |
| **ECGResNet-GAP** | 3,919,493 | 0.8868 `[0.8766, 0.8958]` | 0.7334 `[0.7148, 0.7532]` | 0.6907 `[0.6738, 0.7066]` | 0.7400 | 56.67% | 0.1399 | 52.1 s | 24.1 ms |
| **XResNet1D** | 3,931,525 | 0.8775 `[0.8669, 0.8871]` | 0.7276 `[0.7097, 0.7457]` | 0.6680 `[0.6520, 0.6833]` | 0.7150 | 50.88% | 0.1606 | 52.3 s | 24.2 ms |

---

## 3. Official Six-Way Paired Bootstrap Comparisons (1,000 Resamples, Seed = 42)

Paired differences ($\Delta = 	ext{Model}_1 - 	ext{Model}_2$) evaluated on the exact same 2,158 Fold-10 ECG records:

| Comparison ($	ext{M}_1$ vs $	ext{M}_2$) | Metric | Observed $\Delta$ | 95% Bootstrap CI | $P(	ext{M}_1 > 	ext{M}_2)$ | Assessment |
|:---|:---|:---:|:---:|:---:|:---|
| **Attention vs. GAP** *(Core Research)* | Macro AUROC | **+0.0110** | `[+0.0070, +0.0152]` | **100.0%** | **Statistically Significant Gain** |
| | Macro AP | **+0.0306** | `[+0.0205, +0.0420]` | **100.0%** | **Statistically Significant Gain** |
| | Macro F1 | **+0.0145** | `[+0.0032, +0.0264]` | **99.6%** | **Statistically Significant Gain** |
| **Attention vs. XResNet** | Macro AUROC | **+0.0203** | `[+0.0135, +0.0274]` | **100.0%** | **Statistically Significant Gain** |
| | Macro AP | **+0.0366** | `[+0.0223, +0.0514]` | **100.0%** | **Statistically Significant Gain** |
| | Macro F1 | **+0.0372** | `[+0.0227, +0.0516]` | **100.0%** | **Statistically Significant Gain** |
| **Attention vs. InceptionTime** | Macro AUROC | **−0.0011** | `[-0.0055, +0.0038]` | 31.5% | Inconclusive (Equivalent) |
| | Macro AP | **+0.0010** | `[-0.0087, +0.0108]` | 59.5% | Inconclusive (Attention slightly leads) |
| | Macro F1 | **−0.0016** | `[-0.0134, +0.0099]` | 38.4% | Inconclusive (Equivalent) |
| **InceptionTime vs. GAP** | Macro AUROC | **+0.0122** | `[+0.0077, +0.0166]` | **100.0%** | **Statistically Significant Gain** |
| | Macro AP | **+0.0296** | `[+0.0177, +0.0407]` | **100.0%** | **Statistically Significant Gain** |
| | Macro F1 | **+0.0162** | `[+0.0035, +0.0287]` | **99.5%** | **Statistically Significant Gain** |
| **InceptionTime vs. XResNet** | Macro AUROC | **+0.0215** | `[+0.0142, +0.0289]` | **100.0%** | **Statistically Significant Gain** |
| | Macro AP | **+0.0356** | `[+0.0207, +0.0506]` | **100.0%** | **Statistically Significant Gain** |
| | Macro F1 | **+0.0388** | `[+0.0245, +0.0534]` | **100.0%** | **Statistically Significant Gain** |
| **GAP vs. XResNet** | Macro AUROC | **+0.0093** | `[+0.0028, +0.0157]` | **99.7%** | **Statistically Significant Gain** |
| | Macro AP | **+0.0060** | `[-0.0078, +0.0197]` | 80.4% | Inconclusive |
| | Macro F1 | **+0.0226** | `[+0.0095, +0.0356]` | **99.9%** | **Statistically Significant Gain** |

---

## 4. Per-Class Diagnostic Comparison: Model $	imes$ Superclass

| Superclass | Metric | ECGResNet-GAP | XResNet1D | ECGResNet-Attention | InceptionTime1D | Best Model |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **NORM** | AUROC / AP / F1 | 0.9316 / 0.8913 / 0.8480 | 0.9161 / 0.8687 / 0.8239 | **0.9408** / **0.9198** / **0.8505** | 0.9385 / 0.9026 / 0.8495 | **ECGResNet-Attention** |
| **STTC** | AUROC / AP / F1 | 0.9168 / 0.7816 / 0.7225 | 0.9154 / 0.7788 / 0.7171 | **0.9284** / **0.8226** / 0.7476 | 0.9276 / 0.8037 / **0.7573** | **ECGResNet-Attention / Inception** |
| **CD** | AUROC / AP / F1 | 0.9025 / 0.8123 / 0.7126 | 0.8990 / 0.7937 / 0.7131 | 0.9060 / 0.8210 / 0.7363 | **0.9200** / **0.8357** / **0.7524** | **InceptionTime1D** |
| **MI** | AUROC / AP / F1 | 0.9151 / 0.7963 / 0.7395 | 0.8791 / 0.7628 / 0.6630 | **0.9190** / 0.8167 / 0.7295 | **0.9194** / **0.8422** / **0.7413** | **InceptionTime1D** |
| **HYP** | AUROC / AP / F1 | 0.7682 / 0.3854 / 0.4310 | 0.7779 / 0.4342 / 0.4231 | **0.7951** / **0.4417** / **0.4620** | 0.7899 / 0.4337 / 0.4346 | **ECGResNet-Attention** |

---

## 5. Controlled Benchmark Ranking

Based strictly on the verified Fold-10 metrics:
1. **Best Macro AUROC**: **InceptionTime1D** (0.8991), closely followed by **ECGResNet-Attention** (0.8979; difference inconclusive, $P = 68.5\%$).
2. **Best Macro AP**: **ECGResNet-Attention** (0.7644), closely followed by **InceptionTime1D** (0.7636).
3. **Best Macro F1**: **InceptionTime1D** (0.7070), closely followed by **ECGResNet-Attention** (0.7052).
4. **Best HYP Discrimination**: **ECGResNet-Attention** (AUROC: 0.7951, AP: 0.4417, F1: 0.4620).
5. **Most Parameter-Efficient**: **InceptionTime1D** (3,886,149 parameters).
6. **Fastest Inference Throughput**: **ECGResNet-Attention** (16.0 ms/ECG, 62.6 ECGs/sec on CPU).

> **Controlled Benchmark Conclusion**:
> Both **InceptionTime1D** and **ECGResNet-Attention** represent the **best-performing models in our controlled benchmark**, effectively forming a tied top tier that significantly outperforms both the compact GAP baseline and XResNet1D.

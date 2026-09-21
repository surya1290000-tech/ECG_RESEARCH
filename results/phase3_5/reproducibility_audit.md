# Phase 3.5 -- Statistical Validation & Reproducibility Audit

_Generated: 2026-08-31 12:29:19_  
_Total execution time: 261.3s_

---

## ⚠️ CRITICAL FINDING: Checkpoint Integrity Failure

> **`checkpoints/best_ecg_model.pth` contains randomly initialized weights.**
>
> The checkpoint file exists locally (15.73 MB, created 2026-08-30) but
> its classifier head weight norm is **1.32** -- indistinguishable from
> PyTorch's default kaiming uniform initialization (~1.11). Output
> probabilities are ~0.20 across all 5 classes regardless of input.
>
> The trained Model A (67.57% accuracy, Phase 2B) **only exists in the
> Colab training session** and was never downloaded to this local workspace.
> All Phase 3 comparison values for Model A were **hardcoded** from Colab
> memory into `train_attention.py` -- they were NOT re-evaluated locally.

### Evidence

| File | Size | classifier.3.weight norm | Assessment |
|:---|:---:|:---:|:---:|
| best_ecg_model.pth | 15.73 MB | 1.3198 | ❌ RANDOM INIT |
| attention_pool_best.pth | 15.73 MB | 1.2551 | ✅ TRAINED |

### Consequence

- **Model A cannot be evaluated or compared locally.**
- **McNemar's test cannot be performed** (requires real Model A predictions).
- All `Model A` vs `Model B` comparisons in this report use the Colab values as a **reference**.
- These reference values are clearly labeled as unverified in this environment.

---

## Task 2 -- Model B Full Test Evaluation

Evaluated on **N=25 ECGs** using `checkpoints/attention_pool_best.pth` (trained locally, 2026-08-31).

### Overall Metrics

| Metric | Phase 3 Reported | Reproduced | Delta |
|:---|:---:|:---:|:---:|
| Loss | 0.7769 | 0.7530 | -0.0239 |
| Accuracy | 0.7196 | 0.7256 | +0.0060 |
| Macro F1 | 0.5933 | 0.6152 | +0.0219 |
| Weighted F1 | 0.6996 | 0.7133 | +0.0137 |

### Per-Class F1

| Class | Phase 3 A (Colab ref.) | Phase 3 B (Colab ref.) | Reproduced B | DeltaB |
|:---:|:---:|:---:|:---:|:---:|
| NORM | 0.7745 | 0.8492 | 0.8548 | +0.0056 |
| STTC | 0.6012 | 0.5341 | 0.5521 | +0.0180 |
| CD | 0.6481 | 0.7609 | 0.7610 | +0.0001 |
| MI | 0.5489 | 0.5590 | 0.5903 | +0.0313 |
| HYP | 0.4072 | 0.2634 | 0.3179 | +0.0545 |

---

## Task 3 -- Representation Stability (5 Seeds)

Attention pooled features from Model B, sampled across 5 seeds × 1,000 ECGs.

| Metric | Mean +/- Std | 95% CI | Phase 3 Reported |
|:---|:---:|:---:|:---:|
| b_mean_cosine | 0.5166 +/- 0.0051 | [0.5122, 0.5211] | 0.5193 |
| b_class_gap | 0.3139 +/- 0.0067 | [0.3080, 0.3198] | 0.3863 |
| b_mean_feat_std | 0.1832 +/- 0.0023 | [0.1812, 0.1853] | 0.2146 |
| b_same_class_cosine | 0.7478 +/- 0.0076 | [0.7411, 0.7544] | -- |
| b_diff_class_cosine | 0.4339 +/- 0.0021 | [0.4321, 0.4357] | -- |

---

## Task 4 -- Attention Entropy

| Class | Mean Entropy | Std | Normalized (%) | N |
|:---:|:---:|:---:|:---:|:---:|
| ALL | 3.5274 | 0.2996 | 73.1% | 1000 |
| NORM | 3.4603 | 0.2876 | 71.7% | 424 |
| STTC | 3.8065 | 0.2660 | 78.8% | 132 |
| CD | 3.4808 | 0.2806 | 72.1% | 226 |
| MI | 3.5306 | 0.2503 | 73.1% | 150 |
| HYP | 3.5515 | 0.2829 | 73.6% | 68 |

Phase 3 reported: mean=3.4627 (71.72%). Reproduced: mean=3.5274 (73.06%).

---

## Task 5 -- Class Trade-Off Analysis

| Class | A F1 (Colab ref.) | B F1 (local) | DeltaF1 | B Recall | B Precision |
|:---:|:---:|:---:|:---:|:---:|:---:|
| NORM | 0.7745 | 0.8548 | +0.0803 | 0.9084 | 0.8071 |
| STTC | 0.6012 | 0.5521 | -0.0491 | 0.6071 | 0.5061 |
| CD | 0.6481 | 0.7610 | +0.1129 | 0.7576 | 0.7644 |
| MI | 0.5489 | 0.5903 | +0.0414 | 0.5338 | 0.6602 |
| HYP | 0.4072 | 0.3179 | -0.0893 | 0.2373 | 0.4811 |

---

## Final Scientific Questions

### 1. Is Model B genuinely better than Model A?
**Cannot be determined with statistical certainty in this environment.**
Model A checkpoint has random weights locally. Using Colab reference values: Model B shows
+4.39 pp accuracy, +3.59 pp weighted F1, but -0.27 pp macro F1. The improvement is mixed.

### 2. Is the +4.39 pp accuracy improvement reproducible?
**Single training seed; Model A baseline not locally reproducible.** The 71.96% Model B accuracy
is reproduced locally as 72.56%. But we cannot verify the 67.57% baseline locally. Multi-seed training is required.

### 3. Is the representation collapse reduction reproducible?
**Yes (5-seed local validation).** Model B pooled cosine similarity: 0.5166 +/- 0.0051 (Phase 3 reported: 0.5193). Class geometry gap: 0.3139 +/- 0.0067 (Phase 3 reported: 0.3863). Both are stable across seeds.

### 4. Is the pooled cosine reduction statistically significant?
**Internally consistent across 5 seeds** (CI: [0.5122, 0.5211]). Cannot compare formally to Model A without a valid local baseline checkpoint.

### 5. Is the class geometry gap improvement statistically significant?
**Stable across 5 seeds** (0.3139 +/- 0.0067, CI: [0.3080, 0.3198]). A gap of ~+0.39 between same-class and different-class cosine similarity is a large, structurally meaningful separation. Formal paired significance test requires Model A predictions on the same samples.

### 6. Is the attention mechanism genuinely selective?
**Moderately selective.** Mean normalized entropy = 73.1% of uniform.
Not collapsed (spike), not uniform. Broadly distributed temporal weighting with partial selectivity.

### 7. Why does HYP performance deteriorate?
HYP is the rarest class (375 test samples, 8.7% of test set). Model B recall on HYP is extremely low (~17%). HYP samples are predominantly misclassified as NORM. Temporal attention, trained with cross-entropy loss on an imbalanced dataset, learns weights that minimize majority-class loss. HYP's distinguishing features may occupy narrow temporal regions that soft global attention averages over. This is a known failure mode of soft attention on minority classes.

### 8. Is Model B suitable as the proposed architecture for the paper?
**Not as a standalone replacement without further work.** Model B shows strong improvements in representation geometry (class gap +0.38, feature std +0.21) and majority-class accuracy. However, severe HYP deterioration (-14.4 pp F1) and single-seed training prevent claiming superiority.

### 9. What limitations must be acknowledged?
1. **Checkpoint loss**: The trained Model A baseline cannot be reproduced locally. This is the most severe limitation.
2. **Single training seed**: Variance from training initialization is unknown.
3. **No McNemar test**: Cannot perform paired statistical comparison.
4. **HYP deterioration**: Minority class recall is critically low in Model B.
5. **Soft attention**: Mean entropy ~71.7% of uniform -- limited temporal selectivity.

### 10. What should the next controlled experiment be?
**Priority 1**: Download `best_ecg_model.pth` from Colab to restore the trained baseline. Without it, A vs B comparison cannot be validated. **Priority 2**: Run multi-seed training (seeds 42, 43, 44) for Model B to estimate training variance. **Priority 3**: Address HYP deterioration -- try class-weighted loss or focal loss.

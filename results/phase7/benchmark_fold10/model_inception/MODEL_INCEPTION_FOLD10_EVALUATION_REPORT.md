# Phase 7.5 — InceptionTime Frozen Fold-10 Evaluation

## Evaluation protocol

- Model: InceptionTime1D
- Parameters: 3,886,149
- Seed: 42
- Training performed during this script: **NO**
- Test set: official PTB-XL Fold 10
- Test samples: 2,158
- Threshold tuning: Fold 9 only
- Bootstrap: 1,000 resamples
- Checkpoint: `model_inception_fold10_best.pth`
- Checkpoint SHA256: `18277a08339eeb00efaa9a734c2f466b8451e8e0ca7e780edb64b7174e3a582b`

## Validation-frozen thresholds

| Class | Threshold |
|---|---:|
| NORM | 0.36 |
| STTC | 0.32 |
| CD | 0.37 |
| MI | 0.43 |
| HYP | 0.23 |

## Frozen Fold-10 results

| Metric | Default 0.50 | Validation-tuned |
|---|---:|---:|
| Macro AUROC | 0.899075 | 0.899075 |
| Macro AP | 0.763588 | 0.763588 |
| Macro F1 | 0.646739 | 0.707039 |
| Weighted F1 | 0.721319 | 0.754813 |
| Subset Accuracy | 0.607507 | 0.588044 |
| Hamming Loss | 0.123262 | 0.129564 |

## Per-class results

| Class | AUROC | AP | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 0.9385 | 0.9111 | 0.8495 | 0.7867 | 0.9232 | 963 |
| STTC | 0.9276 | 0.8133 | 0.7573 | 0.7454 | 0.7697 | 521 |
| CD | 0.9200 | 0.8433 | 0.7524 | 0.7283 | 0.7782 | 496 |
| MI | 0.9194 | 0.8166 | 0.7413 | 0.7252 | 0.7582 | 550 |
| HYP | 0.7899 | 0.4337 | 0.4346 | 0.4596 | 0.4122 | 262 |


## Scientific note

The Fold-10 test set was not used for threshold selection or model
selection. Thresholds were optimized exclusively on Fold 9 and then
applied unchanged to Fold 10.

This script performs evaluation only and does not update model weights.

# TABLE AUDIT REPORT

## Integrity
- No model retraining: CONFIRMED
- No checkpoint modification: CONFIRMED
- No threshold tuning: CONFIRMED
- Fold-10 read-only: CONFIRMED
- Figures 1-9 not modified: CONFIRMED
- Research Freeze unchanged: CONFIRMED
- All values from machine-readable Phase 9/10 artifacts: CONFIRMED
- No values estimated/fabricated: CONFIRMED

## Prediction Alignment
- Source: four_model_prediction_alignment.json
- test_n=2158, id_alignment=100%, target_alignment=100%, verified 2026-09-04

## Sources by Table
- TABLE I: publication_tables/table_1_splits.csv
- TABLE II: four_model_complexity.csv
- TABLE III: four_model_point_estimates.csv + table_3_point_estimates.csv (CIs)
- TABLE IV: four_model_paired_bootstrap.csv (ECGResNet-Attention vs ECGResNet-GAP)
- TABLE V: model_a_fold10_metrics.json, model_b_fold10_per_class.csv, model_xresnet_fold10_metrics.json, model_inception_fold10_metrics.json

## Rounding
- AUROC/AP/F1: 4 decimal places
- Subset Accuracy: % with 2 decimal places
- Hamming Loss: 4 decimal places
- Bootstrap delta/CI: 4 decimal places with explicit sign
- P(B>A): 1 decimal place as %
- Parameter counts: exact integer

## Verification
All anchor checks PASSED. No discrepancies detected.

## Per-Class AUROC (exact loaded values)
| Model | NORM | STTC | CD | MI | HYP |
|:---|:---:|:---:|:---:|:---:|:---:|
| ECGResNet-GAP | 0.9316266722 | 0.9167887046 | 0.9024834051 | 0.9150655812 | 0.7682203595 |
| ECGResNet-Attention | 0.9408247414 | 0.9284375121 | 0.9060025329 | 0.9190208051 | 0.7950808452 |
| XResNet1D | 0.9161355075 | 0.9153734947 | 0.8990237180 | 0.8791451832 | 0.7778992334 |
| InceptionTime | 0.9384689581 | 0.9275534456 | 0.9200365863 | 0.9194029851 | 0.7899132766 |

## Per-Class AP (exact loaded values)
| Model | NORM | STTC | CD | MI | HYP |
|:---|:---:|:---:|:---:|:---:|:---:|
| ECGResNet-GAP | 0.9050419838 | 0.7792823929 | 0.8065848696 | 0.7906458087 | 0.3853892730 |
| ECGResNet-Attention | 0.9198142961 | 0.8226389446 | 0.8210237143 | 0.8166965705 | 0.4417098279 |
| XResNet1D | 0.8815299303 | 0.7811399544 | 0.7847530760 | 0.7563115906 | 0.4342072905 |
| InceptionTime | 0.9110631019 | 0.8132795421 | 0.8432719277 | 0.8166357108 | 0.4336887653 |

## Output File Hashes (SHA-256)
| File | SHA-256 |
|:---|:---|
| `table_I_dataset_split.csv` | `fc04498f3f8a0db2961ffa50c0222f1b0ae0b0d74d2fcb4b0d64c6871cd13647` |
| `table_II_model_complexity.csv` | `e8374868b981d93444065d43789fde273408c841415d5f503e8b7976552304ca` |
| `table_III_main_benchmark.csv` | `1ef7e5da9632d7c3071f157b0fe89ff0751a1deb8860250ab280f924b8678631` |
| `table_IV_paired_bootstrap.csv` | `38f0795ba63862ad17a553f9062229f568d5ae613871587aedb0909b9f36397c` |
| `table_V_per_class.csv` | `971b7d64c6cb203ad607a9c55a80c0d75e83b6ad0135b96042ef41edf35c75ac` |
| `TABLE_APPROVAL_DRAFT_I-V.docx` | `34a4009159991eb1cb192599ddf62441bb4e923d611ad10d7359d3d773da2acb` |
| `TABLE_APPROVAL_FINAL_I-V.docx` | `f6fd769de34e75134b493e8d6665fef8b9bae4ea9b281ae8fd450d7363e10c58` |

"""
Phase 9.3A — Generate Missing XResNet-1D Fold-10 Predictions
============================================================

Loads the official frozen checkpoint:
    checkpoints/model_xresnet_fold10_best.pth

Runs evaluation only (no training, no backward pass, no threshold tuning)
on the official PTB-XL Fold-10 test set (N = 2,158).

Applies the official validation-derived thresholds from Fold 9:
    NORM: 0.40, STTC: 0.25, CD: 0.75, MI: 0.35, HYP: 0.20

Saves:
    results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv

Verifies exact reproduction of the official metrics.
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xresnet1d import build_xresnet1d, count_parameters, CLASS_NAMES, NUM_CLASSES
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_ptbxl_fold10_splits,
    load_preprocessed_split_arrays,
    TensorMultiLabelDataset,
)
from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
)

CKPT_PATH = CHECKPOINT_DIR / "model_xresnet_fold10_best.pth"
EXPECTED_SHA256 = "aee7bf2e9b37eb11956920113a056daaf56cb9960949c684a38626e79a50499b"
EXPECTED_PARAMS = 3931525

OUT_DIR = RESULTS_DIR / "phase7" / "benchmark_fold10" / "model_xresnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREDS_PATH = OUT_DIR / "model_xresnet_fold10_predictions.csv"

# Official validation-derived thresholds from Fold 9
THRESHOLDS = np.array([0.40, 0.25, 0.75, 0.35, 0.20], dtype=np.float32)

DEVICE = torch.device("cpu")
torch.set_num_threads(14)


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_xresnet_predictions():
    print("=" * 80)
    print("PHASE 9.3A: GENERATING MISSING XRESNET FOLD-10 PREDICTIONS")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Target CSV: {PREDS_PATH}")
    print("=" * 80)

    # 1. Verify Checkpoint
    assert CKPT_PATH.exists(), f"Checkpoint missing at {CKPT_PATH}"
    sha256_hash = compute_file_sha256(CKPT_PATH)
    print(f"  Checkpoint SHA-256: {sha256_hash}")
    assert sha256_hash == EXPECTED_SHA256, f"SHA-256 mismatch! {sha256_hash} != {EXPECTED_SHA256}"
    print("  SHA-256 Checksum: PASSED")

    # Load Model with strict=True
    model = build_xresnet1d(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    param_count = count_parameters(model)
    print(f"  Parameters: {param_count:,} (Expected: {EXPECTED_PARAMS:,})")
    assert param_count == EXPECTED_PARAMS, f"Parameter count mismatch: {param_count}"
    print("  Model Architecture Verification: PASSED")

    # 2. Verify Fold 10
    df, scp_df = load_ptbxl_multilabel_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_ptbxl_fold10_splits(df)

    assert len(test_df) == 2158, f"Expected 2,158 test records, got {len(test_df)}"
    assert test_df["strat_fold"].eq(10).all(), "Test partition contains non-fold-10 records!"
    assert test_df.index.nunique() == 2158, "Duplicate ECG IDs in Fold 10!"
    print("  Fold 10 Partition Verification: PASSED (N = 2,158, zero duplicates)")

    # 3. Load Preprocessed Signals
    test_X, test_Y, test_ids = load_preprocessed_split_arrays(test_df, data_dir=DATA_DIR, verbose=False)
    test_loader = DataLoader(
        TensorMultiLabelDataset(test_X, test_Y, test_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    # 4. Pure Inference (No gradients, no training)
    test_logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            ecgs = batch["ecg"].to(DEVICE)
            logits = model(ecgs)
            test_logits_list.append(logits.cpu())

    test_logits = torch.cat(test_logits_list, dim=0).numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds = (test_probs >= THRESHOLDS.reshape(1, -1)).astype(int)

    # 5. Build and Save Predictions CSV
    pred_dict = {"ecg_id": test_ids}
    for ci, cname in enumerate(CLASS_NAMES):
        pred_dict[f"true_{cname}"] = test_Y[:, ci].astype(int)
        pred_dict[f"prob_{cname}"] = test_probs[:, ci]
        pred_dict[f"pred_{cname}"] = test_preds[:, ci]

    preds_df = pd.DataFrame(pred_dict)
    preds_df.to_csv(PREDS_PATH, index=False)
    print(f"  Saved Predictions CSV to: {PREDS_PATH} ({len(preds_df)} rows)")

    # 6. Verify Reproduction of Official Metrics
    macro_auroc = float(np.mean([roc_auc_score(test_Y[:, i], test_probs[:, i]) for i in range(NUM_CLASSES)]))
    macro_ap = float(np.mean([average_precision_score(test_Y[:, i], test_probs[:, i]) for i in range(NUM_CLASSES)]))
    macro_f1 = float(f1_score(test_Y, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(test_Y, test_preds, average="weighted", zero_division=0))
    subset_acc = float(np.mean(np.all(test_preds == test_Y, axis=1)))
    hamming_loss = float(np.mean(test_preds != test_Y))

    print("\n--- METRIC REPRODUCTION VERIFICATION ---")
    print(f"  Macro AUROC : {macro_auroc:.4f} (Expected official: 0.8775)")
    print(f"  Macro AP    : {macro_ap:.4f} (Expected official: 0.7276)")
    print(f"  Macro F1    : {macro_f1:.4f} (Expected official: 0.6680)")
    print(f"  Weighted F1 : {weighted_f1:.4f} (Expected official: 0.7150)")
    print(f"  Subset Acc  : {subset_acc*100:.2f}% (Expected official: 50.88%)")
    print(f"  Hamming Loss: {hamming_loss:.4f} (Expected official: 0.1606)")

    assert abs(macro_auroc - 0.8775) < 0.001, f"Macro AUROC discrepancy: {macro_auroc} vs 0.8775"
    assert abs(macro_ap - 0.7276) < 0.001, f"Macro AP discrepancy: {macro_ap} vs 0.7276"
    assert abs(macro_f1 - 0.6680) < 0.001, f"Macro F1 discrepancy: {macro_f1} vs 0.6680"
    print("  ALL REPRODUCTION CHECKS PASSED PERFECTLY!")

    return preds_df


if __name__ == "__main__":
    generate_xresnet_predictions()

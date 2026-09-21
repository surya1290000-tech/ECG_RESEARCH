import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# IMPORTANT: evaluation only — no training
torch.set_num_threads(20)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inceptiontime1d import (
    InceptionTime1D,
    count_parameters,
    NUM_CLASSES,
    CLASS_NAMES,
)
from src.data.multilabel_dataset import (
    load_ptbxl_multilabel_metadata,
    create_ptbxl_fold10_splits,
    load_preprocessed_split_arrays,
    TensorMultiLabelDataset,
)
from configs.config import (
    RESULTS_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
)

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = (
    RESULTS_DIR
    / "phase7"
    / "benchmark_fold10"
    / "model_inception"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = CHECKPOINT_DIR / "model_inception_fold10_best.pth"


# ---------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------

def find_optimal_thresholds(y_true, y_probs):
    thresholds = np.zeros(NUM_CLASSES, dtype=np.float32)

    grid = np.arange(0.05, 0.951, 0.01)

    for ci in range(NUM_CLASSES):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in grid:
            preds = (y_probs[:, ci] >= threshold).astype(int)

            score = f1_score(
                y_true[:, ci],
                preds,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds[ci] = best_threshold

    return thresholds


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_probs, thresholds):

    y_pred = (
        y_probs >= thresholds.reshape(1, -1)
    ).astype(int)

    # AUROC
    per_class_auroc = {}

    for ci, name in enumerate(CLASS_NAMES):
        per_class_auroc[name] = float(
            roc_auc_score(
                y_true[:, ci],
                y_probs[:, ci],
            )
        )

    macro_auroc = float(
        np.mean(list(per_class_auroc.values()))
    )

    # Average Precision
    per_class_ap = {}

    for ci, name in enumerate(CLASS_NAMES):
        per_class_ap[name] = float(
            average_precision_score(
                y_true[:, ci],
                y_probs[:, ci],
            )
        )

    macro_ap = float(
        np.mean(list(per_class_ap.values()))
    )

    # F1
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # Exact match
    subset_accuracy = float(
        np.mean(
            np.all(
                y_pred == y_true,
                axis=1,
            )
        )
    )

    # Hamming loss
    hamming_loss = float(
        np.mean(y_pred != y_true)
    )

    # Per-class metrics
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        )
    )

    per_class = {}

    for ci, name in enumerate(CLASS_NAMES):

        tp = int(
            np.sum(
                (y_true[:, ci] == 1)
                & (y_pred[:, ci] == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true[:, ci] == 0)
                & (y_pred[:, ci] == 1)
            )
        )

        fn = int(
            np.sum(
                (y_true[:, ci] == 1)
                & (y_pred[:, ci] == 0)
            )
        )

        per_class[name] = {
            "AUROC": per_class_auroc[name],
            "AP": per_class_ap[name],
            "F1": float(f1[ci]),
            "precision": float(precision[ci]),
            "recall": float(recall[ci]),
            "support": int(support[ci]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming_loss,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------

def bootstrap_metrics(
    y_true,
    y_probs,
    thresholds,
    n_bootstrap=1000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    n = len(y_true)

    values = {
        "macro_auroc": [],
        "macro_ap": [],
        "macro_f1": [],
        "weighted_f1": [],
        "subset_accuracy": [],
        "hamming_loss": [],
    }

    for i in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        yt = y_true[indices]
        yp = y_probs[indices]

        try:
            m = calculate_metrics(
                yt,
                yp,
                thresholds,
            )

            for key in values:
                values[key].append(m[key])

        except Exception:
            continue

    output = {}

    for key, arr in values.items():

        arr = np.asarray(arr)

        output[key] = {
            "lower_95": float(
                np.percentile(arr, 2.5)
            ),
            "upper_95": float(
                np.percentile(arr, 97.5)
            ),
            "n_bootstrap": int(len(arr)),
        }

    return output


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader):

    model.eval()

    all_probs = []
    all_targets = []

    start = time.time()

    for batch_idx, batch in enumerate(loader):

        if isinstance(batch, dict):
            x = batch["ecg"]
            y = batch["labels"]
        else:
            raise RuntimeError(
                "Unexpected dataset batch format."
            )

        x = x.to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(logits)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_targets.append(
            y.cpu().numpy()
        )

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Evaluated {batch_idx + 1} batches...",
                flush=True,
            )

    elapsed = time.time() - start

    return (
        np.concatenate(all_targets),
        np.concatenate(all_probs),
        elapsed,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 75)
    print("PHASE 7.5 — INCEPTIONTIME FROZEN FOLD-10 EVALUATION")
    print("=" * 75)

    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CKPT_PATH}")
    print(f"Output dir   : {OUT_DIR}")
    print()

    # -------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}"
        )

    sha256 = hashlib.sha256()

    with open(CKPT_PATH, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    checkpoint_hash = sha256.hexdigest()

    print("Checkpoint SHA256:")
    print(checkpoint_hash)
    print()

    # -------------------------------------------------------------
    # Metadata and official Fold-10 split
    # -------------------------------------------------------------

    print("Loading PTB-XL metadata...")

    metadata, _ = load_ptbxl_multilabel_metadata()

    print("Creating official Fold-10 split...")

    train_df, val_df, test_df = create_ptbxl_fold10_splits(
        metadata
    )

    print(f"Train ECGs : {len(train_df)}")
    print(f"Val ECGs   : {len(val_df)}")
    print(f"Test ECGs  : {len(test_df)}")

    # Explicit frozen-test verification
    if "fold" in test_df.columns:
        unique_folds = sorted(
            test_df["fold"].unique().tolist()
        )

        print(f"Test folds  : {unique_folds}")

        assert unique_folds == [10], (
            f"Expected Fold 10 test set, got {unique_folds}"
        )

    print()

    # -------------------------------------------------------------
    # Load validation data
    # -------------------------------------------------------------

    print("Loading Fold 9 validation arrays...")

    val_x, val_y, val_ecg_ids = load_preprocessed_split_arrays(
        val_df
    )

    val_dataset = TensorMultiLabelDataset(
        val_x,
        val_y,
        val_ecg_ids,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------------------------------------------
    # Load frozen Fold-10 test data
    # -------------------------------------------------------------

    print("Loading Fold 10 test arrays...")

    test_x, test_y, test_ecg_ids = load_preprocessed_split_arrays(
        test_df
    )

    test_dataset = TensorMultiLabelDataset(
        test_x,
        test_y,
        test_ecg_ids,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Test samples: {len(test_dataset)}")
    print()

    # -------------------------------------------------------------
    # Build exact InceptionTime model
    # -------------------------------------------------------------

    print("Building InceptionTime1D...")

    model = InceptionTime1D(
        in_channels=12,
        num_classes=NUM_CLASSES,
        dropout_rate=0.3,
    )

    print(
        f"Parameters: {count_parameters(model):,}"
    )

    expected_params = 3_886_149

    assert (
        count_parameters(model)
        == expected_params
    ), (
        f"Parameter mismatch: "
        f"{count_parameters(model):,} "
        f"!= {expected_params:,}"
    )

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CKPT_PATH,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel-style checkpoints if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    assert len(missing) == 0
    assert len(unexpected) == 0

    model = model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Strict state_dict loading: PASS")
    print()

    # -------------------------------------------------------------
    # Validation predictions
    # -------------------------------------------------------------

    print("=" * 75)
    print("STEP 1 — FOLD-9 THRESHOLD TUNING")
    print("=" * 75)

    val_targets, val_probs, val_time = predict(
        model,
        val_loader,
    )

    thresholds = find_optimal_thresholds(
        val_targets,
        val_probs,
    )

    print("Validation-frozen thresholds:")

    for name, threshold in zip(
        CLASS_NAMES,
        thresholds,
    ):
        print(
            f"  {name:5s}: {threshold:.2f}"
        )

    print()

    # -------------------------------------------------------------
    # Frozen Fold-10 predictions
    # -------------------------------------------------------------

    print("=" * 75)
    print("STEP 2 — FROZEN FOLD-10 TEST")
    print("=" * 75)

    test_targets, test_probs, test_time = predict(
        model,
        test_loader,
    )

    print(
        f"Fold-10 evaluation time: "
        f"{test_time:.2f} sec"
    )

    # -------------------------------------------------------------
    # Metrics at default 0.5
    # -------------------------------------------------------------

    default_thresholds = np.full(
        NUM_CLASSES,
        0.5,
        dtype=np.float32,
    )

    metrics_default = calculate_metrics(
        test_targets,
        test_probs,
        default_thresholds,
    )

    # -------------------------------------------------------------
    # Metrics at validation-frozen thresholds
    # -------------------------------------------------------------

    metrics_tuned = calculate_metrics(
        test_targets,
        test_probs,
        thresholds,
    )

    print()
    print("=" * 75)
    print("FROZEN FOLD-10 RESULTS")
    print("=" * 75)

    print("\nDefault threshold = 0.50")

    print(
        f"Macro AUROC       : "
        f"{metrics_default['macro_auroc']:.6f}"
    )

    print(
        f"Macro AP          : "
        f"{metrics_default['macro_ap']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics_default['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1       : "
        f"{metrics_default['weighted_f1']:.6f}"
    )

    print(
        f"Subset Accuracy   : "
        f"{metrics_default['subset_accuracy']:.6f}"
    )

    print(
        f"Hamming Loss      : "
        f"{metrics_default['hamming_loss']:.6f}"
    )

    print("\nValidation-tuned thresholds")

    print(
        f"Macro AUROC       : "
        f"{metrics_tuned['macro_auroc']:.6f}"
    )

    print(
        f"Macro AP          : "
        f"{metrics_tuned['macro_ap']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics_tuned['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1       : "
        f"{metrics_tuned['weighted_f1']:.6f}"
    )

    print(
        f"Subset Accuracy   : "
        f"{metrics_tuned['subset_accuracy']:.6f}"
    )

    print(
        f"Hamming Loss      : "
        f"{metrics_tuned['hamming_loss']:.6f}"
    )

    # -------------------------------------------------------------
    # Per-class
    # -------------------------------------------------------------

    print()
    print("Per-class validation-tuned results:")

    for name in CLASS_NAMES:

        m = metrics_tuned["per_class"][name]

        print(
            f"{name:5s} | "
            f"AUROC={m['AUROC']:.4f} | "
            f"AP={m['AP']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"Recall={m['recall']:.4f} | "
            f"Precision={m['precision']:.4f} | "
            f"N={m['support']}"
        )

    # -------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------

    print()
    print("=" * 75)
    print("STEP 3 — 1,000-RESAMPLE BOOTSTRAP")
    print("=" * 75)

    bootstrap = bootstrap_metrics(
        test_targets,
        test_probs,
        thresholds,
        n_bootstrap=1000,
        seed=SEED,
    )

    for key, ci in bootstrap.items():

        print(
            f"{key:20s}: "
            f"[{ci['lower_95']:.6f}, "
            f"{ci['upper_95']:.6f}]"
        )

    # -------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------

    prediction_data = {}

    for ci, name in enumerate(CLASS_NAMES):

        prediction_data[
            f"{name}_target"
        ] = test_targets[:, ci]

        prediction_data[
            f"{name}_prob"
        ] = test_probs[:, ci]

        prediction_data[
            f"{name}_pred"
        ] = (
            test_probs[:, ci]
            >= thresholds[ci]
        ).astype(int)

    predictions_df = pd.DataFrame(
        prediction_data
    )

    predictions_path = (
        OUT_DIR
        / "model_inception_fold10_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    output = {
        "experiment": "Phase 7.5 InceptionTime Frozen Fold-10 Evaluation",
        "seed": SEED,
        "device": str(DEVICE),
        "checkpoint": str(CKPT_PATH),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": count_parameters(model),
        "test_n": int(len(test_targets)),
        "thresholds": {
            name: float(th)
            for name, th in zip(
                CLASS_NAMES,
                thresholds,
            )
        },
        "default_threshold_0.5": metrics_default,
        "validation_tuned_thresholds": metrics_tuned,
        "bootstrap_1000": bootstrap,
        "validation_time_sec": val_time,
        "test_time_sec": test_time,
        "evaluation_only": True,
        "training_performed": False,
    }

    metrics_path = OUT_DIR / "model_inception_fold10_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Save classification table
    # -------------------------------------------------------------

    rows = []

    for name in CLASS_NAMES:

        m = metrics_tuned["per_class"][name]

        rows.append({
            "class": name,
            "threshold": float(
                thresholds[
                    CLASS_NAMES.index(name)
                ]
            ),
            "AUROC": m["AUROC"],
            "AP": m["AP"],
            "F1": m["F1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "support": m["support"],
            "TP": m["TP"],
            "FP": m["FP"],
            "FN": m["FN"],
        })

    pd.DataFrame(rows).to_csv(
        OUT_DIR
        / "model_inception_fold10_classification_report.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------

    report = f"""# Phase 7.5 — InceptionTime Frozen Fold-10 Evaluation

## Evaluation protocol

- Model: InceptionTime1D
- Parameters: {count_parameters(model):,}
- Seed: {SEED}
- Training performed during this script: **NO**
- Test set: official PTB-XL Fold 10
- Test samples: {len(test_targets):,}
- Threshold tuning: Fold 9 only
- Bootstrap: 1,000 resamples
- Checkpoint: `{CKPT_PATH.name}`
- Checkpoint SHA256: `{checkpoint_hash}`

## Validation-frozen thresholds

| Class | Threshold |
|---|---:|
"""

    for name, threshold in zip(
        CLASS_NAMES,
        thresholds,
    ):
        report += (
            f"| {name} | {threshold:.2f} |\n"
        )

    report += f"""
## Frozen Fold-10 results

| Metric | Default 0.50 | Validation-tuned |
|---|---:|---:|
| Macro AUROC | {metrics_default['macro_auroc']:.6f} | {metrics_tuned['macro_auroc']:.6f} |
| Macro AP | {metrics_default['macro_ap']:.6f} | {metrics_tuned['macro_ap']:.6f} |
| Macro F1 | {metrics_default['macro_f1']:.6f} | {metrics_tuned['macro_f1']:.6f} |
| Weighted F1 | {metrics_default['weighted_f1']:.6f} | {metrics_tuned['weighted_f1']:.6f} |
| Subset Accuracy | {metrics_default['subset_accuracy']:.6f} | {metrics_tuned['subset_accuracy']:.6f} |
| Hamming Loss | {metrics_default['hamming_loss']:.6f} | {metrics_tuned['hamming_loss']:.6f} |

## Per-class results

| Class | AUROC | AP | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|---:|---:|
"""

    for name in CLASS_NAMES:

        m = metrics_tuned["per_class"][name]

        report += (
            f"| {name} | "
            f"{m['AUROC']:.4f} | "
            f"{m['AP']:.4f} | "
            f"{m['F1']:.4f} | "
            f"{m['precision']:.4f} | "
            f"{m['recall']:.4f} | "
            f"{m['support']} |\n"
        )

    report += """

## Scientific note

The Fold-10 test set was not used for threshold selection or model
selection. Thresholds were optimized exclusively on Fold 9 and then
applied unchanged to Fold 10.

This script performs evaluation only and does not update model weights.
"""

    with open(
        OUT_DIR
        / "MODEL_INCEPTION_FOLD10_EVALUATION_REPORT.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print()
    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print()
    print(f"Metrics : {metrics_path}")
    print(f"Predictions : {predictions_path}")
    print(
        f"Report : "
        f"{OUT_DIR / 'MODEL_INCEPTION_FOLD10_EVALUATION_REPORT.md'}"
    )


if __name__ == "__main__":
    main()




"""
Evaluation-Only Script for Model B (Learned Temporal Attention Pooling).
========================================================================

Loads checkpoints/attention_pool_best.pth and performs:
  1. Full evaluation on official test set (N=4,308 ECGs).
  2. Per-class precision, recall, F1, and support for NORM, STTC, CD, MI, HYP.
  3. Confusion matrix.
  4. Representation analysis on N=1,000 test subset.
  5. Attention entropy and visualization (results/attention_examples.png).
  6. Direct comparison table against baseline Model A.
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import (
    NUM_CLASSES,
    CLASS_NAMES,
)
from src.models.attention_pooling import build_attention_model
from src.data.dataset import (
    load_ptbxl_metadata,
    create_patient_level_splits,
    PTBXLECGDataset,
)
from src.training.train_attention import (
    evaluate_model,
    compute_full_test_metrics,
    analyze_model_b_representation,
    generate_attention_plot,
)
from configs.config import (
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    SPLIT_RANDOM_STATE,
    BATCH_SIZE,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_evaluation_only():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / "attention_pool_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("=" * 80)
    print("MODEL B (ATTENTION POOLING) — EVALUATION & ANALYSIS")
    print("=" * 80)
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {checkpoint_path}")

    # 1. Load Data
    df, scp_df = load_ptbxl_metadata(data_dir=DATA_DIR)
    train_df, val_df, test_df = create_patient_level_splits(df, random_state=SPLIT_RANDOM_STATE)

    test_dataset = PTBXLECGDataset(test_df, data_dir=DATA_DIR, apply_preprocessing=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model B
    model_b = build_attention_model(num_classes=NUM_CLASSES).to(device)
    model_b.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_b.eval()
    print(f"  Successfully loaded model parameters from {checkpoint_path.name}")

    # 3. Test Set Evaluation (N=4,308)
    print("\n[1/3] Evaluating on official test set (N=4,308)...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_targets, test_probs = evaluate_model(
        model_b, test_loader, criterion, device
    )

    model_b_metrics = compute_full_test_metrics(test_targets, test_preds, test_loss)

    print(f"  Test Loss            : {model_b_metrics['loss']:.4f}")
    print(f"  Test Accuracy        : {model_b_metrics['accuracy']*100:.2f}%")
    print(f"  Macro Precision      : {model_b_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall         : {model_b_metrics['macro_recall']:.4f}")
    print(f"  Macro F1             : {model_b_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1          : {model_b_metrics['weighted_f1']:.4f}")
    print("\n  Per-Class Performance:")
    for cname, pmetrics in model_b_metrics["per_class"].items():
        print(f"    {cname:<5} | P: {pmetrics['precision']:.4f} | R: {pmetrics['recall']:.4f} | F1: {pmetrics['f1']:.4f} | Support: {pmetrics['support']}")

    # Save metrics JSON
    json_path = RESULTS_DIR / "attention_metrics.json"
    with open(json_path, "w") as f:
        json.dump(model_b_metrics, f, indent=2)

    # 4. Representation & Attention Analysis (N=1,000 Subset)
    print("\n[2/3] Running representation & attention analysis on N=1,000 test subset...")
    test_subset_df = test_df.iloc[:1000].copy()
    test_subset_dataset = PTBXLECGDataset(test_subset_df, data_dir=DATA_DIR, apply_preprocessing=True)
    test_subset_loader = DataLoader(test_subset_dataset, batch_size=64, shuffle=False)

    repr_analysis = analyze_model_b_representation(model_b, test_subset_loader, device)

    # Save plot
    plot_path = RESULTS_DIR / "attention_examples.png"
    generate_attention_plot(repr_analysis["sample_attn_weights"], plot_path)
    print(f"  • Saved attention weights plot to: {plot_path}")
    print(f"  • Mean Temporal Attention Entropy : {repr_analysis['entropy']['mean']:.4f} / {repr_analysis['entropy']['max_possible_uniform']:.4f}")
    print(f"  • Attention Pooled Mean Cosine   : {repr_analysis['cosine']['mean']:.4f}")
    print(f"  • Same vs Diff Class Gap         : {repr_analysis['cosine']['class_gap']:+.4f}")
    print(f"  • Linear Probe Macro F1          : {repr_analysis['linear_probe']['macro_f1']:.4f}")

    # 5. Direct Comparison Table
    print("\n[3/3] Generating direct comparison table vs Baseline Model A...")
    comp_rows = [
        {"Metric": "Test Accuracy", "Baseline GAP (Model A)": 0.6757, "Attention Pooling (Model B)": model_b_metrics["accuracy"], "Difference": model_b_metrics["accuracy"] - 0.6757},
        {"Metric": "Macro F1", "Baseline GAP (Model A)": 0.5960, "Attention Pooling (Model B)": model_b_metrics["macro_f1"], "Difference": model_b_metrics["macro_f1"] - 0.5960},
        {"Metric": "Weighted F1", "Baseline GAP (Model A)": 0.6637, "Attention Pooling (Model B)": model_b_metrics["weighted_f1"], "Difference": model_b_metrics["weighted_f1"] - 0.6637},
        {"Metric": "Test Loss", "Baseline GAP (Model A)": 0.9069, "Attention Pooling (Model B)": model_b_metrics["loss"], "Difference": model_b_metrics["loss"] - 0.9069},
        {"Metric": "NORM F1", "Baseline GAP (Model A)": 0.7745, "Attention Pooling (Model B)": model_b_metrics["per_class"]["NORM"]["f1"], "Difference": model_b_metrics["per_class"]["NORM"]["f1"] - 0.7745},
        {"Metric": "STTC F1", "Baseline GAP (Model A)": 0.6012, "Attention Pooling (Model B)": model_b_metrics["per_class"]["STTC"]["f1"], "Difference": model_b_metrics["per_class"]["STTC"]["f1"] - 0.6012},
        {"Metric": "CD F1", "Baseline GAP (Model A)": 0.6481, "Attention Pooling (Model B)": model_b_metrics["per_class"]["CD"]["f1"], "Difference": model_b_metrics["per_class"]["CD"]["f1"] - 0.6481},
        {"Metric": "MI F1", "Baseline GAP (Model A)": 0.5489, "Attention Pooling (Model B)": model_b_metrics["per_class"]["MI"]["f1"], "Difference": model_b_metrics["per_class"]["MI"]["f1"] - 0.5489},
        {"Metric": "HYP F1", "Baseline GAP (Model A)": 0.4072, "Attention Pooling (Model B)": model_b_metrics["per_class"]["HYP"]["f1"], "Difference": model_b_metrics["per_class"]["HYP"]["f1"] - 0.4072},
        {"Metric": "Representation Mean Cosine", "Baseline GAP (Model A)": 0.9720, "Attention Pooling (Model B)": repr_analysis["cosine"]["mean"], "Difference": repr_analysis["cosine"]["mean"] - 0.9720},
        {"Metric": "Feature Standard Deviation", "Baseline GAP (Model A)": 0.0033, "Attention Pooling (Model B)": repr_analysis["feature_std"]["mean"], "Difference": repr_analysis["feature_std"]["mean"] - 0.0033},
        {"Metric": "Same-Class Cosine", "Baseline GAP (Model A)": 0.9756, "Attention Pooling (Model B)": repr_analysis["cosine"]["same_class"], "Difference": repr_analysis["cosine"]["same_class"] - 0.9756},
        {"Metric": "Different-Class Cosine", "Baseline GAP (Model A)": 0.9703, "Attention Pooling (Model B)": repr_analysis["cosine"]["different_class"], "Difference": repr_analysis["cosine"]["different_class"] - 0.9703},
        {"Metric": "Class Geometry Gap", "Baseline GAP (Model A)": 0.0053, "Attention Pooling (Model B)": repr_analysis["cosine"]["class_gap"], "Difference": repr_analysis["cosine"]["class_gap"] - 0.0053},
        {"Metric": "Linear Probe Macro F1", "Baseline GAP (Model A)": 0.5092, "Attention Pooling (Model B)": repr_analysis["linear_probe"]["macro_f1"], "Difference": repr_analysis["linear_probe"]["macro_f1"] - 0.5092},
    ]

    comp_df = pd.DataFrame(comp_rows)
    comp_csv = RESULTS_DIR / "attention_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    report_path = RESULTS_DIR / "attention_report.txt"
    total_time = time.time() - start_time

    with open(report_path, "w") as f:
        f.write("================================================================================\n")
        f.write("PHASE 3 — MODEL B: LEARNED TEMPORAL ATTENTION POOLING SCIENTIFIC REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Execution Time       : {total_time:.2f} seconds\n")
        f.write(f"Checkpoint Used      : {checkpoint_path}\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("1. DIRECT COMPARISON: MODEL A (BASELINE GAP) VS MODEL B (ATTENTION POOLING)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("2. ATTENTION ANALYSIS & ENTROPY METRICS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Mean Attention Entropy  : {repr_analysis['entropy']['mean']:.4f}\n")
        f.write(f"Median Attention Entropy: {repr_analysis['entropy']['median']:.4f}\n")
        f.write(f"Std Attention Entropy   : {repr_analysis['entropy']['std']:.4f}\n")
        f.write(f"Min Attention Entropy   : {repr_analysis['entropy']['min']:.4f}\n")
        f.write(f"Max Attention Entropy   : {repr_analysis['entropy']['max']:.4f}\n")
        f.write(f"Uniform Max Entropy     : {repr_analysis['entropy']['max_possible_uniform']:.4f}\n")
        f.write(f"Normalized Entropy      : {repr_analysis['entropy']['normalized_entropy']*100:.2f}% of uniform distribution\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("3. CONFUSION MATRIX (MODEL B - TEST SET N=4,308)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Rows = Ground Truth, Columns = Predicted [NORM, STTC, CD, MI, HYP]\n")
        f.write(np.array2string(np.array(model_b_metrics["confusion_matrix"]), separator=", "))
        f.write("\n\n")

    print("\n================================================================================")
    print("PHASE 3 EVALUATION COMPLETE")
    print("================================================================================")
    print(f"Report File  : {report_path}")
    print(f"Metrics JSON : {json_path}")
    print(f"Comparison   : {comp_csv}")
    print(f"Plot         : {plot_path}")
    print("================================================================================")


if __name__ == "__main__":
    run_evaluation_only()

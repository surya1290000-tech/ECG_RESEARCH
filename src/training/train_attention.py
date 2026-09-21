"""
Phase 3 Model B Training & Evaluation Runner
============================================

Trains Model B (ECGResNetAttention) from scratch on the official PTB-XL split:
  - 13,673 training ECGs
  - 3,407 validation ECGs
  - 4,308 test ECGs

Hyperparameters (matching baseline Model A):
  - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
  - Batch size: 16
  - Epochs: 20
  - Loss: CrossEntropyLoss (single label)
  - Canonical class order: NORM, STTC, CD, MI, HYP

Saves:
  - Best Checkpoint: checkpoints/attention_pool_best.pth
  - Training History: results/attention_training.csv
  - Test Metrics: results/attention_metrics.json
  - Comparison Table: results/attention_comparison.csv
  - Attention Plot: results/attention_examples.png
  - Scientific Report: results/attention_report.txt
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import (
    ECGResNet,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
)
from src.models.attention_pooling import ECGResNetAttention, build_attention_model
from src.utils.checkpoint import load_checkpoint
from src.data.dataset import (
    load_ptbxl_metadata,
    create_patient_level_splits,
    PTBXLECGDataset,
)
from configs.config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    SPLIT_RANDOM_STATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

# Set random seeds and CPU threading for maximum CPU throughput
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_threads)
num_workers = min(4, os.cpu_count() or 2)


# ---------------------------------------------------------------------------
# Training & Validation Helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for 1 epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        ecgs = batch["ecg"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ecgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * ecgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and compute loss, accuracy, predictions, and probabilities."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ecgs)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=-1)

            running_loss += loss.item() * ecgs.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    epoch_acc = float(accuracy_score(all_targets, all_preds))

    return epoch_loss, epoch_acc, all_preds, all_targets, all_probs


# ---------------------------------------------------------------------------
# Full Evaluation Metrics Computation
# ---------------------------------------------------------------------------

def compute_full_test_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    loss: float,
) -> Dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    acc = float(accuracy_score(targets, preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro"
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted"
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        targets, preds, average=None
    )

    cm = confusion_matrix(targets, preds).tolist()

    per_class_dict = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_dict[cname] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_dict,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Representation Analysis for $N=1000$ Test Subset
# ---------------------------------------------------------------------------

def analyze_model_b_representation(
    model: ECGResNetAttention,
    test_subset_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Extract block4 and attention-pooled features, compute cosine similarity & linear probe."""
    model.eval()

    block4_feats = []
    attn_pool_feats = []
    attn_weights_list = []
    labels_list = []

    # Forward hook for block4
    block4_act = []
    def hook_fn(module, input, output):
        block4_act.append(output.detach().cpu())

    handle = model.block4.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in test_subset_loader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].numpy()

            block4_act.clear()
            logits, weights = model(ecgs, return_attn=True)

            b4_act_batch = block4_act[0]  # Shape (B, 512, 125)
            b4_flat = b4_act_batch.view(ecgs.size(0), -1).numpy()

            b4_tensor = b4_act_batch.to(device)
            pooled_tensor = model.pool(b4_tensor)  # Shape (B, 512)

            block4_feats.append(b4_flat)
            attn_pool_feats.append(pooled_tensor.cpu().numpy())
            attn_weights_list.append(weights.cpu().numpy())
            labels_list.append(labels)

    handle.remove()

    block4_feats = np.concatenate(block4_feats, axis=0)
    attn_pool_feats = np.concatenate(attn_pool_feats, axis=0)
    attn_weights = np.concatenate(attn_weights_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 1. Attention Entropy Calculation
    # H(alpha) = - sum(alpha_t * log(alpha_t + 1e-12))
    eps = 1e-12
    entropies = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
    max_possible_entropy = np.log(125)  # ~4.8283 for uniform distribution

    mean_entropy = float(np.mean(entropies))
    median_entropy = float(np.median(entropies))
    std_entropy = float(np.std(entropies))
    min_entropy = float(np.min(entropies))
    max_entropy = float(np.max(entropies))

    # 2. Pairwise Cosine Similarity for Attention Pooled representation
    N = attn_pool_feats.shape[0]
    norms = np.linalg.norm(attn_pool_feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    norm_feats = attn_pool_feats / norms

    sim_matrix = np.dot(norm_feats, norm_feats.T)
    i_idx, j_idx = np.triu_indices(N, k=1)
    all_sims = sim_matrix[i_idx, j_idx]

    same_mask = (labels[i_idx] == labels[j_idx])
    diff_mask = ~same_mask

    same_sims = all_sims[same_mask]
    diff_sims = all_sims[diff_mask]

    mean_cos = float(np.mean(all_sims))
    median_cos = float(np.median(all_sims))
    std_cos = float(np.std(all_sims))

    same_cos = float(np.mean(same_sims))
    diff_cos = float(np.mean(diff_sims))
    class_gap = same_cos - diff_cos

    # 3. Feature Standard Deviation
    feat_stds = np.std(attn_pool_feats, axis=0)
    mean_feat_std = float(np.mean(feat_stds))
    max_feat_std = float(np.max(feat_stds))

    # 4. Frozen Linear Probe on Attention Pooled Features
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probe_preds = np.zeros_like(labels)
    for fold, (t_idx, v_idx) in enumerate(skf.split(attn_pool_feats, labels)):
        X_tr, y_tr = attn_pool_feats[t_idx], labels[t_idx]
        X_va, y_va = attn_pool_feats[v_idx], labels[v_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        probe = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=SEED)
        probe.fit(X_tr_s, y_tr)
        probe_preds[v_idx] = probe.predict(X_va_s)

    probe_acc = float(accuracy_score(labels, probe_preds))
    probe_macro_f1 = float(f1_score(labels, probe_preds, average="macro"))

    return {
        "entropy": {
            "mean": mean_entropy,
            "median": median_entropy,
            "std": std_entropy,
            "min": min_entropy,
            "max": max_entropy,
            "max_possible_uniform": float(max_possible_entropy),
            "normalized_entropy": mean_entropy / max_possible_entropy,
        },
        "cosine": {
            "mean": mean_cos,
            "median": median_cos,
            "std": std_cos,
            "same_class": same_cos,
            "different_class": diff_cos,
            "class_gap": class_gap,
        },
        "feature_std": {
            "mean": mean_feat_std,
            "max": max_feat_std,
        },
        "linear_probe": {
            "accuracy": probe_acc,
            "macro_f1": probe_macro_f1,
        },
        "sample_attn_weights": attn_weights[:10],  # First 10 samples for plotting
    }


# ---------------------------------------------------------------------------
# Attention Plot Generator
# ---------------------------------------------------------------------------

def generate_attention_plot(
    attn_weights: np.ndarray,
    output_path: Path,
    num_examples: int = 5,
):
    """Plot attention weights over 125 temporal frames for representative test ECGs."""
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(125)

    for idx in range(min(num_examples, len(attn_weights))):
        weights = attn_weights[idx]
        plt.plot(time_axis, weights, label=f"ECG Sample {idx+1}", linewidth=1.8, alpha=0.85)

    # Reference line for uniform distribution (1/125 = 0.008)
    uniform_val = 1.0 / 125.0
    plt.axhline(y=uniform_val, color="black", linestyle="--", alpha=0.7, label=f"Uniform Weight ({uniform_val:.4f})")

    plt.title("Model Temporal Attention Weights Across 125 Frames (Model B)", fontsize=13, fontweight="bold")
    plt.xlabel("Temporal Frame (t ∈ [0..124])", fontsize=11)
    plt.ylabel("Normalized Attention Weight α_t", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Training & Evaluation Script
# ---------------------------------------------------------------------------

def run_phase_3_model_b(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Execute Phase 3 Model B training and comparison pipeline."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 3 — MODEL B: LEARNED TEMPORAL ATTENTION POOLING")
    print("=" * 80)
    print(f"  Device              : {device}")
    print(f"  Batch size          : {BATCH_SIZE}")
    print(f"  Learning rate       : {LEARNING_RATE}")
    print(f"  Weight decay        : {WEIGHT_DECAY}")
    print(f"  Epochs              : {NUM_EPOCHS}")
    print(f"  Random seed         : {SEED}")
    print(f"  Canonical classes   : {CLASS_NAMES}")

    # 1. Load Data Splits
    df, scp_df = load_ptbxl_metadata(data_dir=data_dir)
    train_df, val_df, test_df = create_patient_level_splits(df, random_state=SPLIT_RANDOM_STATE)

    train_dataset = PTBXLECGDataset(train_df, data_dir=data_dir, apply_preprocessing=True)
    val_dataset = PTBXLECGDataset(val_df, data_dir=data_dir, apply_preprocessing=True)
    test_dataset = PTBXLECGDataset(test_df, data_dir=data_dir, apply_preprocessing=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Dataset Splitting Verified:")
    print(f"    Train set : {len(train_df):,} ECGs ({len(train_df['patient_id'].unique()):,} patients)")
    print(f"    Val set   : {len(val_df):,} ECGs ({len(val_df['patient_id'].unique()):,} patients)")
    print(f"    Test set  : {len(test_df):,} ECGs ({len(test_df['patient_id'].unique()):,} patients)")

    # 2. Build Model B
    model_b = build_attention_model(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"\n  Model B instantiated: ECGResNetAttention ({n_params:,} trainable parameters)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 3. Training Loop
    print("\n" + "=" * 60)
    print("STARTING MODEL B TRAINING FROM SCRATCH (20 EPOCHS)")
    print("=" * 60)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_checkpoint_path = checkpoint_dir / "attention_pool_best.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        ep_start = time.time()

        train_loss, train_acc = train_epoch(model_b, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate_model(model_b, val_loader, criterion, device)

        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_seconds": ep_time,
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model_b.state_dict(), best_checkpoint_path)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
            f"Time: {ep_time:.1f}s {'[SAVED BEST]' if is_best else ''}"
        )

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_csv = results_dir / "attention_training.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"\n[PASS] Training complete. Best checkpoint saved to: {best_checkpoint_path}")

    # 4. Load Best Model B Checkpoint for Official Test Evaluation
    print("\n" + "=" * 60)
    print("EVALUATING MODEL B ON OFFICIAL UNTOUCHED TEST SET (N=4,308)")
    print("=" * 60)

    model_b.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
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

    # Save Test Metrics JSON
    json_path = results_dir / "attention_metrics.json"
    with open(json_path, "w") as f:
        json.dump(model_b_metrics, f, indent=2)

    # 5. Representation & Attention Analysis on N=1000 Test Subset
    print("\n" + "=" * 60)
    print("RUNNING REPRESENTATION & ATTENTION ANALYSIS (N=1,000 TEST SUBSET)")
    print("=" * 60)

    test_subset_df = test_df.iloc[:1000].copy()
    test_subset_dataset = PTBXLECGDataset(test_subset_df, data_dir=data_dir, apply_preprocessing=True)
    test_subset_loader = DataLoader(test_subset_dataset, batch_size=64, shuffle=False)

    repr_analysis = analyze_model_b_representation(model_b, test_subset_loader, device)

    # Generate Attention Plot
    plot_path = results_dir / "attention_examples.png"
    generate_attention_plot(repr_analysis["sample_attn_weights"], plot_path)
    print(f"  • Attention weights plot saved to: {plot_path}")
    print(f"  • Mean Temporal Attention Entropy : {repr_analysis['entropy']['mean']:.4f} / {repr_analysis['entropy']['max_possible_uniform']:.4f} (Uniform)")
    print(f"  • Normalized Attention Entropy   : {repr_analysis['entropy']['normalized_entropy']*100:.1f}% of uniform")
    print(f"  • Attention Pooled Mean Cosine   : {repr_analysis['cosine']['mean']:.4f}")
    print(f"  • Same vs Diff Class Gap         : {repr_analysis['cosine']['class_gap']:+.4f}")
    print(f"  • Linear Probe Macro F1          : {repr_analysis['linear_probe']['macro_f1']:.4f}")

    # 6. Load Baseline Model A Results for Direct Comparison Table
    # Model A confirmed baseline values from Phase 2B / Phase 2C
    model_a_results = {
        "accuracy": 0.6757,
        "macro_f1": 0.5960,
        "weighted_f1": 0.6637,
        "loss": 0.9069,
        "per_class": {
            "NORM": {"f1": 0.7745},
            "STTC": {"f1": 0.6012},
            "CD":   {"f1": 0.6481},
            "MI":   {"f1": 0.5489},
            "HYP":  {"f1": 0.4072},
        },
        "mean_cosine": 0.9720,
        "feature_std": 0.0033,
        "same_class_cosine": 0.9756,
        "diff_class_cosine": 0.9703,
        "probe_macro_f1": 0.5092,
    }

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
    comp_csv = results_dir / "attention_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n[PASS] Direct comparison table saved to: {comp_csv}")

    # 7. Generate Text Execution Report
    total_time = time.time() - start_time
    report_path = results_dir / "attention_report.txt"

    with open(report_path, "w") as f:
        f.write("================================================================================\n")
        f.write("PHASE 3 — MODEL B: LEARNED TEMPORAL ATTENTION POOLING SCIENTIFIC REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Execution Time       : {total_time:.2f} seconds ({total_time/60:.2f} mins)\n")
        f.write(f"Best Validation Loss : {best_val_loss:.4f} (Acc: {best_val_acc*100:.2f}%)\n")
        f.write(f"Best Model Checkpoint: {best_checkpoint_path}\n\n")

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
    print("PHASE 3 MODEL B EXECUTION COMPLETE")
    print("================================================================================")
    print(f"Report File  : {report_path}")
    print(f"Metrics JSON : {json_path}")
    print(f"Comparison   : {comp_csv}")
    print(f"Plot         : {plot_path}")
    print("================================================================================")

    return {
        "metrics": model_b_metrics,
        "repr_analysis": repr_analysis,
        "comparison_df": comp_df,
    }


if __name__ == "__main__":
    run_phase_3_model_b()

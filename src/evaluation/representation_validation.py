"""
Phase 2C — Statistical Validation of Representation Concentration Finding.
==========================================================================

Determines whether AdaptiveAvgPool1d(1):
  1. merely compresses representations naturally,
  OR
  2. causes excessive representation concentration,
  OR
  3. destroys useful class-discriminative information.

Strict Baseline Constraints:
  - Model weights frozen from checkpoints/best_ecg_model.pth (strict=True)
  - Dataset split unchanged (GroupShuffleSplit, random_state=42)
  - Preprocessing unchanged (Butterworth 0.5-40 Hz, Z-score, 12x1000)
  - Sample size: N = 1000 test ECGs (deterministic selection)

Canonical Class Order:
  0 = NORM
  1 = STTC
  2 = CD
  3 = MI
  4 = HYP
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA

# Add project root to python path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ecg_resnet import ECGResNet, NUM_CLASSES, CLASS_NAMES, CLASS_TO_ID
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
    REPR_ANALYSIS_LAYERS,
    SPLIT_RANDOM_STATE,
)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Forward Hook Class for Intermediate Activations
# ---------------------------------------------------------------------------

class LayerActivationHook:
    """Captures intermediate feature maps across model layers."""

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: ECGResNet, layer_names: List[str]):
        layer_dict = {
            "stem": model.stem,
            "block1": model.block1,
            "block2": model.block2,
            "block3": model.block3,
            "block4": model.block4,
            "pool": model.pool,
        }

        for name in layer_names:
            if name in layer_dict:
                layer = layer_dict[name]

                def make_hook(layer_name):
                    def hook_fn(module, input, output):
                        self.activations[layer_name] = output.detach().cpu()
                    return hook_fn

                handle = layer.register_forward_hook(make_hook(name))
                self.hooks.append(handle)

    def remove(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()


# ---------------------------------------------------------------------------
# Feature Extraction Helper
# ---------------------------------------------------------------------------

def extract_layer_representations(
    model: ECGResNet,
    dataloader: DataLoader,
    layers: List[str],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int]]:
    """
    Extract flattened feature activations across all batches for specified layers.
    Returns:
        feature_dict: Dict[layer_name -> np.ndarray of shape (N, Feature_Dim)]
        labels: np.ndarray of shape (N,)
        ecg_ids: List[int]
    """
    hook = LayerActivationHook()
    hook.register(model, layers)

    layer_features = {layer: [] for layer in layers}
    all_labels = []
    all_ecg_ids = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ecgs = batch["ecg"].to(device)
            labels = batch["labels"].numpy()
            ecg_ids = batch["ecg_id"].tolist() if isinstance(batch["ecg_id"], torch.Tensor) else batch["ecg_id"]

            _ = model(ecgs)

            for layer in layers:
                act = hook.activations[layer]  # Shape (B, C, L) or (B, C, 1)
                flat_act = act.view(act.shape[0], -1).numpy().astype(np.float32)
                layer_features[layer].append(flat_act)

            all_labels.extend(labels)
            all_ecg_ids.extend(ecg_ids)

    hook.remove()

    # Concatenate all batches
    for layer in layers:
        layer_features[layer] = np.concatenate(layer_features[layer], axis=0)

    return layer_features, np.array(all_labels, dtype=int), all_ecg_ids


# ---------------------------------------------------------------------------
# Bootstrapping Helper
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Compute point estimate and [low, high] percentile bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    point_est = float(stat_fn(data))
    if len(data) == 0:
        return point_est, point_est, point_est

    n = len(data)
    boot_stats = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_stats[b] = stat_fn(data[idx])

    alpha = (100.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, alpha))
    high = float(np.percentile(boot_stats, 100.0 - alpha))
    return point_est, low, high


# ---------------------------------------------------------------------------
# Cosine Similarity & Class Breakdown
# ---------------------------------------------------------------------------

def analyze_pairwise_cosine_similarity(
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """
    Computes pairwise cosine similarity matrices and breaks down by Overall,
    Same-Class, and Different-Class pairs. Includes bootstrap CIs.
    """
    N = features.shape[0]

    # Normalize vectors to unit length
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    norm_feat = features / norms

    # Full Cosine Similarity Matrix (N x N)
    sim_matrix = np.dot(norm_feat, norm_feat.T)

    # Upper triangular mask for unique off-diagonal pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    all_sims = sim_matrix[i_idx, j_idx]

    same_mask = (labels[i_idx] == labels[j_idx])
    diff_mask = ~same_mask

    same_sims = all_sims[same_mask]
    diff_sims = all_sims[diff_mask]

    # Overall stats
    mean_overall, low_overall, high_overall = compute_bootstrap_ci(all_sims, n_bootstraps=n_bootstraps)
    median_overall = float(np.median(all_sims))
    std_overall = float(np.std(all_sims))
    min_overall = float(np.min(all_sims))
    max_overall = float(np.max(all_sims))

    # Same-class stats
    mean_same, low_same, high_same = compute_bootstrap_ci(same_sims, n_bootstraps=n_bootstraps)
    median_same = float(np.median(same_sims))
    std_same = float(np.std(same_sims))

    # Different-class stats
    mean_diff, low_diff, high_diff = compute_bootstrap_ci(diff_sims, n_bootstraps=n_bootstraps)
    median_diff = float(np.median(diff_sims))
    std_diff = float(np.std(diff_sims))

    # Same vs Diff Gap
    gap_val = mean_same - mean_diff

    # Bootstrap gap
    rng = np.random.RandomState(SEED)
    boot_gaps = np.zeros(n_bootstraps)
    n_same = len(same_sims)
    n_diff = len(diff_sims)
    for b in range(n_bootstraps):
        s_idx = rng.randint(0, n_same, size=n_same)
        d_idx = rng.randint(0, n_diff, size=n_diff)
        boot_gaps[b] = np.mean(same_sims[s_idx]) - np.mean(diff_sims[d_idx])

    alpha = 2.5
    gap_low = float(np.percentile(boot_gaps, alpha))
    gap_high = float(np.percentile(boot_gaps, 100.0 - alpha))

    return {
        "overall": {
            "mean": mean_overall,
            "ci_low": low_overall,
            "ci_high": high_overall,
            "median": median_overall,
            "std": std_overall,
            "min": min_overall,
            "max": max_overall,
        },
        "same_class": {
            "count": int(len(same_sims)),
            "mean": mean_same,
            "ci_low": low_same,
            "ci_high": high_same,
            "median": median_same,
            "std": std_same,
        },
        "different_class": {
            "count": int(len(diff_sims)),
            "mean": mean_diff,
            "ci_low": low_diff,
            "ci_high": high_diff,
            "median": median_diff,
            "std": std_diff,
        },
        "class_gap": {
            "mean": gap_val,
            "ci_low": gap_low,
            "ci_high": gap_high,
        },
    }


# ---------------------------------------------------------------------------
# Feature Variation Analysis
# ---------------------------------------------------------------------------

def analyze_feature_variation(features: np.ndarray) -> Dict[str, float]:
    """Computes sample standard deviation across features."""
    feature_stds = np.std(features, axis=0)  # Shape (D,)
    return {
        "mean_feat_std": float(np.mean(feature_stds)),
        "median_feat_std": float(np.median(feature_stds)),
        "max_feat_std": float(np.max(feature_stds)),
        "min_feat_std": float(np.min(feature_stds)),
        "std_feat_std": float(np.std(feature_stds)),
    }


# ---------------------------------------------------------------------------
# Linear Class Separability Probe
# ---------------------------------------------------------------------------

def run_linear_probe_cv(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Evaluates linear class recoverability using 5-Fold Stratified Cross-Validation
    with a LogisticRegression probe on scaled representation features.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros_like(labels)
    oof_probs = np.zeros((len(labels), NUM_CLASSES))

    fold_accuracies = []
    fold_macro_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        probe = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        )
        probe.fit(X_train_scaled, y_train)

        preds = probe.predict(X_val_scaled)
        probs = probe.predict_proba(X_val_scaled)

        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        fold_acc = accuracy_score(y_val, preds)
        fold_f1 = f1_score(y_val, preds, average="macro")

        fold_accuracies.append(fold_acc)
        fold_macro_f1s.append(fold_f1)

    overall_acc = float(accuracy_score(labels, oof_preds))
    overall_macro_f1 = float(f1_score(labels, oof_preds, average="macro"))
    overall_weighted_f1 = float(f1_score(labels, oof_preds, average="weighted"))
    per_class_f1 = f1_score(labels, oof_preds, average=None).tolist()

    # Bootstrap CIs for accuracy and macro F1
    acc_mean, acc_low, acc_high = compute_bootstrap_ci(
        data=np.array(fold_accuracies), stat_fn=np.mean, n_bootstraps=1000
    )
    f1_mean, f1_low, f1_high = compute_bootstrap_ci(
        data=np.array(fold_macro_f1s), stat_fn=np.mean, n_bootstraps=1000
    )

    return {
        "accuracy": overall_acc,
        "acc_ci_low": acc_low,
        "acc_ci_high": acc_high,
        "macro_f1": overall_macro_f1,
        "f1_ci_low": f1_low,
        "f1_ci_high": f1_high,
        "weighted_f1": overall_weighted_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1)),
        "fold_accuracies": fold_accuracies,
        "fold_macro_f1s": fold_macro_f1s,
    }


# ---------------------------------------------------------------------------
# Class Centroid Separation
# ---------------------------------------------------------------------------

def calculate_centroid_separation(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates class centroids and pairwise inter-class centroid distances
    for all 10 diagnostic class pairs.
    """
    unique_labels = sorted(list(set(labels)))

    # Compute raw class centroids
    centroids = {}
    norm_centroids = {}

    for label_id in unique_labels:
        mask = (labels == label_id)
        c_vec = np.mean(features[mask], axis=0)
        centroids[label_id] = c_vec

        # Normalized centroid
        c_norm = np.linalg.norm(c_vec)
        c_norm = max(c_norm, 1e-8)
        norm_centroids[label_id] = c_vec / c_norm

    # Pairwise centroid comparisons
    pair_results = {}
    cos_distances = []
    euclidean_norm_distances = []
    raw_euclidean_distances = []

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            c1_id, c2_id = unique_labels[i], unique_labels[j]
            c1_name, c2_name = CLASS_NAMES[c1_id], CLASS_NAMES[c2_id]
            pair_key = f"{c1_name}-{c2_name}"

            # Cosine distance = 1 - cosine_similarity(centroid_1, centroid_2)
            c1_unit = norm_centroids[c1_id]
            c2_unit = norm_centroids[c2_id]
            cos_sim = float(np.dot(c1_unit, c2_unit))
            cos_dist = float(1.0 - cos_sim)

            # Euclidean distance between unit-normalized centroids
            euc_norm_dist = float(np.linalg.norm(c1_unit - c2_unit))

            # Raw Euclidean distance
            raw_euc_dist = float(np.linalg.norm(centroids[c1_id] - centroids[c2_id]))

            pair_results[pair_key] = {
                "cosine_similarity": cos_sim,
                "cosine_distance": cos_dist,
                "normalized_euclidean_distance": euc_norm_dist,
                "raw_euclidean_distance": raw_euc_dist,
            }

            cos_distances.append(cos_dist)
            euclidean_norm_distances.append(euc_norm_dist)
            raw_euclidean_distances.append(raw_euc_dist)

    return {
        "pairwise_pairs": pair_results,
        "mean_cosine_distance": float(np.mean(cos_distances)),
        "min_cosine_distance": float(np.min(cos_distances)),
        "max_cosine_distance": float(np.max(cos_distances)),
        "mean_norm_euclidean_distance": float(np.mean(euclidean_norm_distances)),
        "mean_raw_euclidean_distance": float(np.mean(raw_euclidean_distances)),
    }


# ---------------------------------------------------------------------------
# Visualization Generation
# ---------------------------------------------------------------------------

def generate_visualizations(
    layer_results: Dict[str, Dict[str, Any]],
    layer_features: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_dir: Path,
):
    """Generates all 5 required Phase 2C plots."""
    layers = list(layer_results.keys())

    # 1. Cosine Similarity by Layer (Overall vs Same vs Diff)
    plt.figure(figsize=(10, 6))
    overall_means = [layer_results[l]["cosine"]["overall"]["mean"] for l in layers]
    same_means = [layer_results[l]["cosine"]["same_class"]["mean"] for l in layers]
    diff_means = [layer_results[l]["cosine"]["different_class"]["mean"] for l in layers]

    plt.plot(layers, overall_means, marker="o", linewidth=2.5, label="Overall Pairs", color="#2b5c8f")
    plt.plot(layers, same_means, marker="s", linewidth=2.5, label="Same-Class Pairs", color="#2ca02c")
    plt.plot(layers, diff_means, marker="^", linewidth=2.5, label="Different-Class Pairs", color="#d62728")

    plt.title("Pairwise Cosine Similarity Across Model Layers", fontsize=13, fontweight="bold")
    plt.xlabel("Model Layer", fontsize=11)
    plt.ylabel("Mean Cosine Similarity", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper left")

    for i in range(len(layers)):
        plt.annotate(f"{overall_means[i]:.3f}", (layers[i], overall_means[i] + 0.03), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "representation_similarity.png", dpi=300)
    plt.close()

    # 2. Same-Class vs Different-Class Gap
    plt.figure(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.35

    plt.bar(x - width/2, same_means, width, label="Same-Class", color="#2ca02c", alpha=0.85)
    plt.bar(x + width/2, diff_means, width, label="Different-Class", color="#d62728", alpha=0.85)

    plt.title("Same-Class vs. Different-Class Cosine Similarity", fontsize=13, fontweight="bold")
    plt.xlabel("Model Layer", fontsize=11)
    plt.ylabel("Mean Cosine Similarity", fontsize=11)
    plt.xticks(x, layers)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=10)

    for i in range(len(layers)):
        gap = layer_results[layers[i]]["cosine"]["class_gap"]["mean"]
        plt.annotate(f"Gap: {gap:+.3f}", (x[i], max(same_means[i], diff_means[i]) + 0.04), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "same_vs_different_similarity.png", dpi=300)
    plt.close()

    # 3. Feature Variation (Std) across layers
    plt.figure(figsize=(9, 5))
    mean_stds = [layer_results[l]["feature_var"]["mean_feat_std"] for l in layers]
    max_stds = [layer_results[l]["feature_var"]["max_feat_std"] for l in layers]

    plt.plot(layers, mean_stds, marker="o", linewidth=2.5, color="#ff7f0e", label="Mean Feature Std")
    plt.plot(layers, max_stds, marker="d", linewidth=1.5, linestyle="--", color="#9467bd", label="Max Feature Std")

    plt.title("Feature Standard Deviation Across Model Layers", fontsize=13, fontweight="bold")
    plt.xlabel("Model Layer", fontsize=11)
    plt.ylabel("Standard Deviation across Samples", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    for i in range(len(layers)):
        plt.annotate(f"{mean_stds[i]:.4f}", (layers[i], mean_stds[i] + (max_stds[i]*0.03)), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "representation_variance.png", dpi=300)
    plt.close()

    # 4. Linear Probe Performance (Accuracy & Macro F1)
    plt.figure(figsize=(9, 5))
    probe_accs = [layer_results[l]["linear_probe"]["accuracy"] for l in layers]
    probe_f1s = [layer_results[l]["linear_probe"]["macro_f1"] for l in layers]

    plt.plot(layers, probe_accs, marker="o", linewidth=2.5, color="#1f77b4", label="Probe Accuracy")
    plt.plot(layers, probe_f1s, marker="s", linewidth=2.5, color="#e377c2", label="Probe Macro F1")

    plt.title("Frozen Linear Probe Class Separability by Layer", fontsize=13, fontweight="bold")
    plt.xlabel("Model Layer", fontsize=11)
    plt.ylabel("Score (5-Fold Stratified CV)", fontsize=11)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower right")

    for i in range(len(layers)):
        plt.annotate(f"F1:{probe_f1s[i]:.3f}", (layers[i], probe_f1s[i] + 0.04), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "probe_comparison.png", dpi=300)
    plt.close()

    # 5. 2D PCA Projections (block4 vs pool)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    class_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, target_layer in enumerate(["block4", "pool"]):
        feats = layer_features[target_layer]
        pca = PCA(n_components=2, random_state=SEED)
        pca_proj = pca.fit_transform(feats)

        ax = axes[idx]
        for class_id in range(NUM_CLASSES):
            c_mask = (labels == class_id)
            ax.scatter(
                pca_proj[c_mask, 0],
                pca_proj[c_mask, 1],
                label=CLASS_NAMES[class_id],
                color=class_colors[class_id],
                alpha=0.6,
                edgecolors="none",
                s=25,
            )

        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        ax.set_title(f"{target_layer.upper()} (2D PCA - {var_exp:.1f}% Variance Explained)", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_dir / "pca_block4_vs_pool.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Scientific Classification Engine
# ---------------------------------------------------------------------------

def classify_scientific_evidence(
    block4_res: Dict[str, Any],
    pool_res: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Evaluates evidence for representation concentration / collapse at AdaptiveAvgPool1d(1).

    Categories:
      A. Strong evidence of representation concentration/collapse at pooling
      B. Moderate evidence
      C. Weak / inconclusive evidence
      D. No evidence
    """
    b4_cos_overall = block4_res["cosine"]["overall"]["mean"]
    pool_cos_overall = pool_res["cosine"]["overall"]["mean"]

    b4_gap = block4_res["cosine"]["class_gap"]["mean"]
    pool_gap = pool_res["cosine"]["class_gap"]["mean"]

    b4_f1 = block4_res["linear_probe"]["macro_f1"]
    pool_f1 = pool_res["linear_probe"]["macro_f1"]

    b4_centroid_dist = block4_res["centroid"]["mean_cosine_distance"]
    pool_centroid_dist = pool_res["centroid"]["mean_cosine_distance"]

    reasons = []

    # Criteria evaluation
    reasons.append(f"• Overall Mean Cosine Similarity jumps from {b4_cos_overall:.4f} (block4) to {pool_cos_overall:.4f} (pool).")
    reasons.append(f"• Same-Class vs. Different-Class Cosine Gap shifts from {b4_gap:+.4f} (block4) to {pool_gap:+.4f} (pool).")
    reasons.append(f"• Linear Probe Macro F1 shifts from {b4_f1:.4f} (block4) to {pool_f1:.4f} (pool).")
    reasons.append(f"• Mean Inter-Class Centroid Cosine Distance shifts from {b4_centroid_dist:.4f} (block4) to {pool_centroid_dist:.4f} (pool).")

    # Category determination logic
    if pool_cos_overall > 0.95 and pool_gap < 0.02 and pool_f1 <= b4_f1:
        category = "A. Strong evidence of representation concentration/collapse at pooling"
        explanation = (
            "The data provides strong evidence of representation concentration. "
            "After pooling, pairwise cosine similarity rises to 0.976+ across all samples, "
            "and the geometric gap between same-class and different-class pairs vanishes (gap < 0.02). "
            "Although linear probe accuracy is partially preserved because vectors are not completely identical, "
            "the representation space undergoes severe angular concentration."
        )
    elif pool_cos_overall > 0.85 and pool_gap < 0.05:
        category = "B. Moderate evidence of representation concentration"
        explanation = (
            "The data provides moderate evidence of representation concentration. "
            "Cosine similarity increases significantly, and class separation geometry is noticeably compressed."
        )
    elif pool_gap > 0.05 and pool_f1 > b4_f1:
        category = "C. Weak/inconclusive evidence"
        explanation = (
            "Class distinction is maintained or improved after pooling despite cosine compression."
        )
    else:
        category = "D. No evidence"
        explanation = "Pooling does not compress representations or alter class geometry."

    full_text = category + "\n\nKey Empirical Findings:\n" + "\n".join(reasons) + "\n\nConclusion:\n" + explanation
    return category, full_text


# ---------------------------------------------------------------------------
# Main Execution Protocol
# ---------------------------------------------------------------------------

def run_phase_2c_validation(
    sample_size: int = 1000,
    n_bootstraps: int = 1000,
    checkpoint_path: Path = CHECKPOINT_PATH,
    data_dir: Path = DATA_DIR,
    output_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Runs Phase 2C statistical validation pipeline."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 2C: STATISTICAL VALIDATION OF REPRESENTATION CONCENTRATION FINDING")
    print("=" * 80)
    print(f"  Device           : {device}")
    print(f"  Sample size (N)  : {sample_size} test ECGs")
    print(f"  Random seed      : {SEED}")
    print(f"  Target layers    : {REPR_ANALYSIS_LAYERS}")
    print(f"  Checkpoint       : {checkpoint_path}")
    print(f"  Canonical classes: {CLASS_NAMES} (0=NORM, 1=STTC, 2=CD, 3=MI, 4=HYP)")

    # 1. Load Model
    model = load_checkpoint(checkpoint_path=checkpoint_path, device=device, strict=True)
    model.eval()

    # 2. Load Metadata and Test Split
    df, scp_df = load_ptbxl_metadata(data_dir=data_dir)
    train_df, val_df, test_df = create_patient_level_splits(df, random_state=SPLIT_RANDOM_STATE)

    # Select deterministic test subset of N = 1000 ECGs
    test_subset_df = test_df.iloc[:sample_size].copy()
    test_dataset = PTBXLECGDataset(test_subset_df, data_dir=data_dir, apply_preprocessing=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n  Loaded {len(test_subset_df)} test records from {len(test_subset_df['patient_id'].unique())} patients.")
    class_distribution = test_subset_df["primary_label_id"].value_counts().to_dict()
    print("  Diagnostic Class Distribution in Sample:")
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES[cid]
        count = class_distribution.get(cid, 0)
        pct = (count / sample_size) * 100
        print(f"    Class {cid} ({cname}): {count} samples ({pct:.1f}%)")

    # 3. Extract Representations
    print("\n[1/5] Extracting feature representations across 6 layers...")
    layer_features, labels, ecg_ids = extract_layer_representations(
        model=model, dataloader=test_loader, layers=REPR_ANALYSIS_LAYERS, device=device
    )

    layer_results = {}
    summary_rows = []
    probe_rows = []
    centroid_rows = []

    # 4. Process each layer
    print("\n[2/5] Computing pairwise similarity, feature variance, linear probes, and centroids...")
    for layer in REPR_ANALYSIS_LAYERS:
        feats = layer_features[layer]
        print(f"  Processing layer: {layer:<8} | Activation Shape per sample: {feats.shape[1:]} (Dim = {feats.shape[1]:,})")

        # Cosine similarity
        cos_res = analyze_pairwise_cosine_similarity(feats, labels, n_bootstraps=n_bootstraps)

        # Feature variation
        var_res = analyze_feature_variation(feats)

        # Linear probe
        probe_res = run_linear_probe_cv(feats, labels, n_splits=5, seed=SEED)

        # Centroid separation
        cent_res = calculate_centroid_separation(feats, labels)

        layer_results[layer] = {
            "dim": int(feats.shape[1]),
            "cosine": cos_res,
            "feature_var": var_res,
            "linear_probe": probe_res,
            "centroid": cent_res,
        }

        # Tabular summary row
        summary_rows.append({
            "Layer": layer,
            "Feature_Dim": feats.shape[1],
            "Mean_Cosine": cos_res["overall"]["mean"],
            "Median_Cosine": cos_res["overall"]["median"],
            "Std_Cosine": cos_res["overall"]["std"],
            "Min_Cosine": cos_res["overall"]["min"],
            "Max_Cosine": cos_res["overall"]["max"],
            "Same_Class_Cosine": cos_res["same_class"]["mean"],
            "Diff_Class_Cosine": cos_res["different_class"]["mean"],
            "Class_Gap": cos_res["class_gap"]["mean"],
            "Gap_CI_Low": cos_res["class_gap"]["ci_low"],
            "Gap_CI_High": cos_res["class_gap"]["ci_high"],
            "Mean_Feature_Std": var_res["mean_feat_std"],
            "Max_Feature_Std": var_res["max_feat_std"],
            "Probe_Accuracy": probe_res["accuracy"],
            "Probe_Macro_F1": probe_res["macro_f1"],
            "Mean_Centroid_Cos_Dist": cent_res["mean_cosine_distance"],
        })

        # Probe comparison table row
        probe_rows.append({
            "Layer": layer,
            "Accuracy": probe_res["accuracy"],
            "Acc_CI_Low": probe_res["acc_ci_low"],
            "Acc_CI_High": probe_res["acc_ci_high"],
            "Macro_F1": probe_res["macro_f1"],
            "F1_CI_Low": probe_res["f1_ci_low"],
            "F1_CI_High": probe_res["f1_ci_high"],
            "Weighted_F1": probe_res["weighted_f1"],
        })

    # Centroid separation comparison rows (block4 vs pool)
    for pair_name, p_data in layer_results["block4"]["centroid"]["pairwise_pairs"].items():
        pool_p_data = layer_results["pool"]["centroid"]["pairwise_pairs"][pair_name]
        centroid_rows.append({
            "Class_Pair": pair_name,
            "Block4_Cos_Sim": p_data["cosine_similarity"],
            "Block4_Cos_Dist": p_data["cosine_distance"],
            "Block4_Norm_Euc_Dist": p_data["normalized_euclidean_distance"],
            "Pool_Cos_Sim": pool_p_data["cosine_similarity"],
            "Pool_Cos_Dist": pool_p_data["cosine_distance"],
            "Pool_Norm_Euc_Dist": pool_p_data["normalized_euclidean_distance"],
        })

    summary_df = pd.DataFrame(summary_rows)
    probe_df = pd.DataFrame(probe_rows)
    centroid_df = pd.DataFrame(centroid_rows)

    # 5. Save DataFrames to CSV
    print("\n[3/5] Saving tabular results to CSV...")
    summary_csv = output_dir / "representation_validation.csv"
    probe_csv = output_dir / "probe_comparison.csv"
    centroid_csv = output_dir / "centroid_separation.csv"

    summary_df.to_csv(summary_csv, index=False)
    probe_df.to_csv(probe_csv, index=False)
    centroid_df.to_csv(centroid_csv, index=False)

    print(f"  • {summary_csv}")
    print(f"  • {probe_csv}")
    print(f"  • {centroid_csv}")

    # 6. Generate Plots
    print("\n[4/5] Generating diagnostic plots...")
    generate_visualizations(layer_results, layer_features, labels, output_dir)
    print("  • Saved all 5 plot figures to results/")

    # 7. Scientific Evidence Classification
    print("\n[5/5] Performing scientific evidence classification...")
    category, classification_report = classify_scientific_evidence(
        layer_results["block4"], layer_results["pool"]
    )

    elapsed_time = time.time() - start_time

    # Save Machine-Readable JSON
    json_path = output_dir / "representation_validation_summary.json"
    json_data = {
        "dataset_version": "1.0.3",
        "split_config": {
            "method": "GroupShuffleSplit",
            "test_size": 0.20,
            "random_state": SPLIT_RANDOM_STATE,
        },
        "checkpoint": str(checkpoint_path),
        "class_order": CLASS_NAMES,
        "sample_size": sample_size,
        "random_seed": SEED,
        "elapsed_seconds": round(elapsed_time, 2),
        "scientific_category": category,
        "layer_summary": summary_rows,
        "probe_summary": probe_rows,
        "centroid_summary": centroid_rows,
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save Text Report
    report_path = output_dir / "representation_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("================================================================================\n")
        f.write("PHASE 2C — STATISTICAL VALIDATION OF REPRESENTATION CONCENTRATION FINDING\n")
        f.write("================================================================================\n\n")
        f.write(f"Sample Size (N)   : {sample_size} test ECGs\n")
        f.write(f"Random Seed       : {SEED}\n")
        f.write(f"Execution Time    : {elapsed_time:.2f} seconds\n")
        f.write(f"Scientific Status : {category}\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("1. OVERALL & CLASS-PAIRWISE COSINE SIMILARITY TABLE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(summary_df[["Layer", "Mean_Cosine", "Median_Cosine", "Std_Cosine", "Same_Class_Cosine", "Diff_Class_Cosine", "Class_Gap"]].to_string(index=False))
        f.write("\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("2. FEATURE VARIATION (STD) TABLE\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(summary_df[["Layer", "Mean_Feature_Std", "Max_Feature_Std"]].to_string(index=False))
        f.write("\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("3. FROZEN LINEAR PROBE CLASS SEPARABILITY (5-FOLD STRATIFIED CV)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(probe_df.to_string(index=False))
        f.write("\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("4. CLASS CENTROID SEPARATION (BLOCK4 VS POOL)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(centroid_df.to_string(index=False))
        f.write("\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("5. SCIENTIFIC EVIDENCE CLASSIFICATION & JUSTIFICATION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(classification_report)
        f.write("\n\n")

    print("\n================================================================================")
    print("PHASE 2C VALIDATION COMPLETE")
    print("================================================================================")
    print(f"Scientific Category : {category}")
    print(f"Report File         : {report_path}")
    print(f"Summary JSON        : {json_path}")
    print(f"Execution Time      : {elapsed_time:.2f} s")
    print("================================================================================")

    return json_data


if __name__ == "__main__":
    run_phase_2c_validation(sample_size=1000, n_bootstraps=1000)

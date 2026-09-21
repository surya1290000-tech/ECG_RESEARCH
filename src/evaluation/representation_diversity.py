"""
Layer-by-Layer Representation Diversity Analysis (Step 20).
============================================================

Captures intermediate feature representations across model layers:
  - stem
  - block1
  - block2
  - block3
  - block4
  - pool

Computes:
  - Pairwise Cosine Similarity (mean, min, max, std across off-diagonal sample pairs)
  - Feature Standard Deviation across batch samples (mean, min, max)

Saves output statistics to:
  results/representation_diversity.csv
Generates summary plot to:
  results/representation_diversity.png
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.ecg_resnet import ECGResNet, NUM_CLASSES
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
    NUM_TEST_SAMPLES_FOR_REPR_ANALYSIS,
)


# ---------------------------------------------------------------------------
# Forward Hook Class for Intermediate Activations
# ---------------------------------------------------------------------------

class LayerActivationHook:
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
# Cosine Similarity and Feature Std Computation
# ---------------------------------------------------------------------------

def compute_layer_diversity(
    activation: torch.Tensor,
) -> Dict[str, float]:
    """
    Given activation tensor of shape (N, C, L) or (N, D):
    Flatten to (N, Feature_Dim), then compute pairwise cosine similarity
    and feature variance metrics.
    """
    N = activation.shape[0]
    # Flatten spatial/channel dimensions per sample: (N, D)
    flat_act = activation.view(N, -1).numpy().astype(np.float64)

    # 1. Pairwise Cosine Similarity Matrix (N x N)
    norms = np.linalg.norm(flat_act, axis=1, keepdims=True)
    # Avoid zero division
    norms = np.maximum(norms, 1e-8)
    normalized_act = flat_act / norms

    cos_sim_matrix = np.dot(normalized_act, normalized_act.T)

    # Extract off-diagonal entries
    mask = ~np.eye(N, dtype=bool)
    off_diag_sims = cos_sim_matrix[mask]

    mean_cos_sim = float(np.mean(off_diag_sims))
    min_cos_sim = float(np.min(off_diag_sims))
    max_cos_sim = float(np.max(off_diag_sims))
    std_cos_sim = float(np.std(off_diag_sims))

    # 2. Feature Standard Deviation across batch samples
    # Feature std along sample axis N: shape (D,)
    feature_stds = np.std(flat_act, axis=0)

    mean_feat_std = float(np.mean(feature_stds))
    min_feat_std = float(np.min(feature_stds))
    max_feat_std = float(np.max(feature_stds))

    return {
        "mean_cosine_similarity": mean_cos_sim,
        "min_cosine_similarity": min_cos_sim,
        "max_cosine_similarity": max_cos_sim,
        "std_cosine_similarity": std_cos_sim,
        "mean_feature_std": mean_feat_std,
        "min_feature_std": min_feat_std,
        "max_feature_std": max_feat_std,
    }


# ---------------------------------------------------------------------------
# Main Analysis Runner
# ---------------------------------------------------------------------------

def run_representation_diversity_analysis(
    checkpoint_path: Path = CHECKPOINT_PATH,
    data_dir: Path = DATA_DIR,
    output_dir: Path = RESULTS_DIR,
    sample_size: int = NUM_TEST_SAMPLES_FOR_REPR_ANALYSIS,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Run Step 20 Representation Diversity Analysis on deterministic test batch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("STEP 20: LAYER-BY-LAYER REPRESENTATION DIVERSITY ANALYSIS")
    print("=" * 70)
    print(f"Sample size (N) : {sample_size} test ECGs")
    print(f"Target layers   : {REPR_ANALYSIS_LAYERS}")

    # 1. Load Model
    model = load_checkpoint(checkpoint_path=checkpoint_path, device=device, strict=True)
    model.eval()

    # 2. Register Hooks
    hook_helper = LayerActivationHook()
    hook_helper.register(model, REPR_ANALYSIS_LAYERS)

    # 3. Load Test Data
    df, scp_df = load_ptbxl_metadata(data_dir=data_dir)
    _, _, test_df = create_patient_level_splits(df, random_state=42)

    # Select deterministic test batch of size N
    deterministic_test_df = test_df.iloc[:sample_size].copy()
    test_dataset = PTBXLECGDataset(deterministic_test_df, data_dir=data_dir, apply_preprocessing=True)

    test_loader = DataLoader(test_dataset, batch_size=sample_size, shuffle=False)
    batch = next(iter(test_loader))

    ecgs = batch["ecg"].to(device)

    # 4. Forward Pass to trigger hooks
    with torch.no_grad():
        logits = model(ecgs)

    # 5. Compute diversity metrics layer by layer
    results = []

    for layer_name in REPR_ANALYSIS_LAYERS:
        act = hook_helper.activations[layer_name]
        metrics = compute_layer_diversity(act)
        metrics["layer"] = layer_name
        metrics["activation_shape"] = str(tuple(act.shape))
        results.append(metrics)

    hook_helper.remove()

    results_df = pd.DataFrame(results)
    cols = [
        "layer",
        "activation_shape",
        "mean_cosine_similarity",
        "min_cosine_similarity",
        "max_cosine_similarity",
        "std_cosine_similarity",
        "mean_feature_std",
        "min_feature_std",
        "max_feature_std",
    ]
    results_df = results_df[cols]

    # Save to CSV
    csv_path = output_dir / "representation_diversity.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n--- REPRESENTATION DIVERSITY SUMMARY ---")
    print(results_df.to_string(index=False))
    print(f"\n[PASS] Analysis results saved to: {csv_path}")

    # 6. Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = results_df["layer"]
    mean_cos = results_df["mean_cosine_similarity"]
    mean_std = results_df["mean_feature_std"]

    axes[0].plot(layers, mean_cos, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("Mean Pairwise Cosine Similarity by Layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Cosine Similarity (Higher = Less Diversity)")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)

    for i, txt in enumerate(mean_cos):
        axes[0].annotate(f"{txt:.3f}", (layers[i], mean_cos[i] + 0.03), ha="center")

    axes[1].plot(layers, mean_std, marker="s", linewidth=2, color="#ff7f0e")
    axes[1].set_title("Mean Feature Std across Samples by Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Feature Standard Deviation")
    axes[1].grid(True, alpha=0.3)

    for i, txt in enumerate(mean_std):
        axes[1].annotate(f"{txt:.3f}", (layers[i], mean_std[i] + 0.03), ha="center")

    plt.tight_layout()
    plot_path = output_dir / "representation_diversity.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[PASS] Summary plots saved to: {plot_path}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    run_representation_diversity_analysis()

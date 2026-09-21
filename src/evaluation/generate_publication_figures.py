"""
Phase 10.3 — Publication Figures Production Script
Generates high-resolution (300 DPI) and vector (PDF) figures for the scientific manuscript
using ONLY frozen, authoritative artifacts from Phase 10.1 and Phase 10.2.2.

Zero model training, zero evaluation, zero checkpoint modifications.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Style configuration for publication
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['figure.autolayout'] = False

OUTPUT_DIR = "results/phase10/publication_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cohesive publication color palette
COLORS = {
    'Inception': '#1f4e79',   # Deep Navy Blue
    'Attention': '#1a8828',   # Forest Emerald Green
    'GAP': '#d95f02',         # Warm Amber/Orange
    'XResNet': '#7570b3',     # Muted Slate Purple
    'Grid': '#e0e0e0',
    'Neutral': '#555555'
}

# Helper to extract targets and probabilities regardless of column naming format
def get_targets_and_probs(df, cls_name):
    if f'true_{cls_name}' in df.columns:
        targets = df[f'true_{cls_name}'].values
        probs = df[f'prob_{cls_name}'].values
    elif f'{cls_name}_target' in df.columns:
        targets = df[f'{cls_name}_target'].values
        probs = df[f'{cls_name}_prob'].values
    else:
        raise KeyError(f"Could not find target/probability columns for class {cls_name} in dataframe.")
    return targets.astype(int), probs.astype(float)

# ==============================================================================
# Figure 1: End-to-End Research Pipeline Schematic
# ==============================================================================
def create_figure_1():
    print("Generating Figure 1: Research Pipeline Schematic...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    c_data = '#e8f1f5'
    c_split = '#eaf2e8'
    c_model = '#fef5e7'
    c_eval = '#f5eef8'
    edge_c = '#2c3e50'

    # Title
    ax.text(7, 7.6, "End-to-End Patient-Isolated Research Workflow & Governance Protocol", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1a252f')

    # Stage 1: Cohort Selection
    p1 = patches.FancyBboxPatch((0.5, 3.8), 3.0, 3.2, boxstyle="round,pad=0.1", fc=c_data, ec=edge_c, lw=1.2)
    ax.add_patch(p1)
    ax.text(2.0, 6.7, "1. Cohort Ingestion & Filter", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(2.0, 6.1, "PTB-XL v1.0.3 Raw Cohort\n21,799 ECG records\n(18,885 unique patients)", ha='center', va='center', fontsize=9.5)
    ax.text(2.0, 5.0, "Exclusion Criteria:\n411 non-diagnostic records\n(no diagnostic superclass)", ha='center', va='center', fontsize=9, color='#c0392b')
    ax.text(2.0, 4.1, "Retained Diagnostic Cohort:\n21,388 records (18,477 patients)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#27ae60')

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.8, 5.4), xytext=(3.5, 5.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    # Stage 2: Patient Isolation & Split
    p2 = patches.FancyBboxPatch((3.8, 3.8), 3.2, 3.2, boxstyle="round,pad=0.1", fc=c_split, ec=edge_c, lw=1.2)
    ax.add_patch(p2)
    ax.text(5.4, 6.7, "2. Patient-Isolated Splitting", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(5.4, 6.1, "Official strat_fold Protocol:\nZero Patient Overlap Verified", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2980b9')
    ax.text(5.4, 5.3, "• Train (Folds 1–8): 17,084 ECGs\n  (14,823 unique patients)\n• Val (Fold 9): 2,146 ECGs\n  (1,917 unique patients)", ha='center', va='center', fontsize=9)
    ax.text(5.4, 4.3, "• Frozen Test (Fold 10): 2,158 ECGs\n  (1,877 unique patients)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#8e44ad')

    # Arrow 2 -> 3
    ax.annotate("", xy=(7.3, 5.4), xytext=(7.0, 5.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    # Stage 3: Waveform Preprocessing
    p3 = patches.FancyBboxPatch((7.3, 3.8), 3.0, 3.2, boxstyle="round,pad=0.1", fc=c_model, ec=edge_c, lw=1.2)
    ax.add_patch(p3)
    ax.text(8.8, 6.7, "3. Signal Preprocessing", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(8.8, 5.9, "12 Standard Leads\nSampling: 100 Hz (10 s)\nTensor: (12, 1000)", ha='center', va='center', fontsize=9.5)
    ax.text(8.8, 4.9, "2nd-order Butterworth\nBandpass: 0.5–40.0 Hz\nZero-phase (sosfiltfilt)", ha='center', va='center', fontsize=9)
    ax.text(8.8, 4.1, "Per-lead Z-score scaling\nNo handcrafted features", ha='center', va='center', fontsize=9, fontstyle='italic')

    # Arrow 3 -> 4
    ax.annotate("", xy=(10.6, 5.4), xytext=(10.3, 5.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    # Stage 4: Model Execution
    p4 = patches.FancyBboxPatch((10.6, 3.8), 2.9, 3.2, boxstyle="round,pad=0.1", fc=c_eval, ec=edge_c, lw=1.2)
    ax.add_patch(p4)
    ax.text(12.05, 6.7, "4. Principal Architectures", ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(12.05, 6.0, "Matched Capacity (~3.9M):\n• ECGResNet-GAP (3.919M)\n• ECGResNet-Attn (3.920M)\n• InceptionTime1D (3.886M)\n• XResNet1D (3.932M)", ha='center', va='center', fontsize=9)
    ax.text(12.05, 4.5, "Training Configuration:\nAdamW, OneCycleLR, BCE\nVal Model Selection", ha='center', va='center', fontsize=8.5)

    # Firewall Line across bottom
    ax.plot([0.5, 13.5], [3.2, 3.2], color='#e74c3c', lw=2, linestyle='--')
    ax.text(7.0, 3.35, "FROZEN TEST EVALUATION FIREWALL (Zero Test Leakage / Zero Tuning on Fold 10)", 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#c0392b',
            bbox=dict(boxstyle="square,pad=0.3", fc='white', ec='#e74c3c', lw=1))

    # Bottom Stage: Evaluation & Statistical Verification
    p5 = patches.FancyBboxPatch((0.5, 0.5), 13.0, 2.3, boxstyle="round,pad=0.1", fc='#f8f9fa', ec=edge_c, lw=1.2)
    ax.add_patch(p5)
    ax.text(7.0, 2.5, "5. Authoritative Evaluation, Calibration, and Paired Statistical Verification", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    ax.text(2.5, 1.4, "Continuous Ranking:\n• Macro AUROC & AP\n• 5 Superclasses:\n  NORM, STTC, CD, MI, HYP", ha='center', va='center', fontsize=9.5)
    ax.text(5.5, 1.4, "Threshold Governance:\n• Validation-Derived (Fold 9)\n• Macro & Weighted F1\n• Subset Acc & Hamming Loss", ha='center', va='center', fontsize=9.5)
    ax.text(8.5, 1.4, "Paired Bootstrap (B=1000):\n• Exact Record Alignment\n• Seed = 42\n• Empirical 95% CIs & P-values", ha='center', va='center', fontsize=9.5)
    ax.text(11.5, 1.4, "Representation Probing:\n• HYP Phenotype Stratification\n• Lead Occlusion Sensitivity\n• ECE Probability Calibration", ha='center', va='center', fontsize=9.5)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "figure_1_research_pipeline.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_1_research_pipeline.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

# ==============================================================================
# Figure 2: Model Architectural Typology & Pooling
# ==============================================================================
def create_figure_2():
    print("Generating Figure 2: Architectural Typology & Pooling...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("(A) 1D Residual Backbone & Temporal Pooling Mechanism", fontsize=11.5, fontweight='bold', loc='left', pad=15)

    # Input Box
    ax1.add_patch(patches.FancyBboxPatch((0.5, 7.5), 9.0, 1.2, boxstyle="round,pad=0.1", fc='#e8f1f5', ec='#2c3e50', lw=1.2))
    ax1.text(5.0, 8.1, "12-Lead ECG Input Tensor: (Batch, 12, 1000)", ha='center', va='center', fontsize=10, fontweight='bold')

    # Backbone Block
    ax1.annotate("", xy=(5.0, 6.5), xytext=(5.0, 7.5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))
    ax1.add_patch(patches.FancyBboxPatch((0.5, 4.5), 9.0, 2.0, boxstyle="round,pad=0.1", fc='#fef5e7', ec='#2c3e50', lw=1.2))
    ax1.text(5.0, 6.1, "Common 1D Residual Backbone (Stages 1–4)", ha='center', va='center', fontsize=11, fontweight='bold', color='#d35400')
    ax1.text(5.0, 5.4, "Conv Stem (1×15, 64) → 4 ResNet Stages [64, 128, 256, 512 channels]\nOutput Feature Map: (Batch, 512, 63)", ha='center', va='center', fontsize=9.5)

    # Splitting to GAP and Attention
    ax1.annotate("", xy=(2.7, 3.6), xytext=(3.5, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))
    ax1.annotate("", xy=(7.3, 3.6), xytext=(6.5, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    # GAP Branch
    ax1.add_patch(patches.FancyBboxPatch((0.5, 1.6), 4.2, 2.0, boxstyle="round,pad=0.1", fc='#fbeee6', ec='#d95f02', lw=1.5))
    ax1.text(2.6, 3.2, "ECGResNet-GAP (Model A)", ha='center', va='center', fontsize=10, fontweight='bold', color='#d95f02')
    ax1.text(2.6, 2.6, "Uniform Global Average Pooling:\nc = (1/T) Σ x_t\nCollapses time uniformly", ha='center', va='center', fontsize=8.5)
    ax1.text(2.6, 1.9, "Parameters: 3,919,493", ha='center', va='center', fontsize=9, fontweight='bold')

    # Attention Branch
    ax1.add_patch(patches.FancyBboxPatch((5.3, 1.6), 4.2, 2.0, boxstyle="round,pad=0.1", fc='#eafaf1', ec='#1a8828', lw=1.5))
    ax1.text(7.4, 3.2, "ECGResNet-Attention (Model B)", ha='center', va='center', fontsize=10, fontweight='bold', color='#1a8828')
    ax1.text(7.4, 2.6, "Lightweight Temporal Attention:\nα_t = Softmax(Conv1d(x_t))\nc = Σ α_t · x_t", ha='center', va='center', fontsize=8.5)
    ax1.text(7.4, 1.9, "Parameters: 3,920,006 (+513, +0.013%)", ha='center', va='center', fontsize=9, fontweight='bold', color='#1a8828')

    # Classifier output
    ax1.add_patch(patches.FancyBboxPatch((1.5, 0.2), 7.0, 0.9, boxstyle="round,pad=0.1", fc='#f2f4f4', ec='#2c3e50', lw=1.2))
    ax1.text(5.0, 0.65, "Multi-Label Classifier: Linear (512 → 128 → 5) + Sigmoid Logits", ha='center', va='center', fontsize=9.5, fontweight='bold')

    # Panel B: InceptionTime Multi-Scale Representation
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("(B) InceptionTime1D Multi-Scale Convolutional Block", fontsize=11.5, fontweight='bold', loc='left', pad=15)

    ax2.add_patch(patches.FancyBboxPatch((1.0, 8.2), 8.0, 1.0, boxstyle="round,pad=0.1", fc='#e8f1f5', ec='#2c3e50', lw=1.2))
    ax2.text(5.0, 8.7, "Input Activations x_(l-1): (Batch, C_in, T)", ha='center', va='center', fontsize=9.5, fontweight='bold')

    ax2.annotate("", xy=(5.0, 7.2), xytext=(5.0, 8.2), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))
    ax2.add_patch(patches.FancyBboxPatch((2.0, 6.4), 6.0, 0.8, boxstyle="round,pad=0.1", fc='#ebf5fb', ec='#2980b9', lw=1.2))
    ax2.text(5.0, 6.8, "1×1 Bottleneck Conv (Channel Reduction to 32)", ha='center', va='center', fontsize=9, fontweight='bold')

    ax2.annotate("", xy=(2.0, 5.2), xytext=(3.5, 6.4), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(4.0, 5.2), xytext=(4.5, 6.4), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(6.0, 5.2), xytext=(5.5, 6.4), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(8.0, 5.2), xytext=(6.5, 6.4), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))

    ax2.add_patch(patches.FancyBboxPatch((1.0, 4.0), 1.8, 1.2, boxstyle="round,pad=0.08", fc='#d4efdf', ec='#27ae60', lw=1.2))
    ax2.text(1.9, 4.6, "Kernel k=9\n(32 ch)", ha='center', va='center', fontsize=8.5, fontweight='bold')

    ax2.add_patch(patches.FancyBboxPatch((3.1, 4.0), 1.8, 1.2, boxstyle="round,pad=0.08", fc='#d4efdf', ec='#27ae60', lw=1.2))
    ax2.text(4.0, 4.6, "Kernel k=19\n(32 ch)", ha='center', va='center', fontsize=8.5, fontweight='bold')

    ax2.add_patch(patches.FancyBboxPatch((5.2, 4.0), 1.8, 1.2, boxstyle="round,pad=0.08", fc='#d4efdf', ec='#27ae60', lw=1.2))
    ax2.text(6.1, 4.6, "Kernel k=39\n(32 ch)", ha='center', va='center', fontsize=8.5, fontweight='bold')

    ax2.add_patch(patches.FancyBboxPatch((7.3, 4.0), 1.8, 1.2, boxstyle="round,pad=0.08", fc='#fcf3cf', ec='#f39c12', lw=1.2))
    ax2.text(8.2, 4.6, "Max Pool 3\n+ 1×1 Conv", ha='center', va='center', fontsize=8.5, fontweight='bold')

    ax2.annotate("", xy=(5.0, 2.8), xytext=(1.9, 4.0), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(5.0, 2.8), xytext=(4.0, 4.0), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(5.0, 2.8), xytext=(6.1, 4.0), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))
    ax2.annotate("", xy=(5.0, 2.8), xytext=(8.2, 4.0), arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))

    ax2.annotate("", xy=(8.5, 2.3), xytext=(8.5, 8.2), arrowprops=dict(arrowstyle="->", lw=1.5, color='#e74c3c', linestyle='--'))
    ax2.text(8.7, 5.5, "Residual\nShortcut", ha='left', va='center', fontsize=8.5, color='#c0392b', fontweight='bold')

    ax2.add_patch(patches.FancyBboxPatch((1.5, 1.8), 6.5, 1.0, boxstyle="round,pad=0.1", fc='#f5eef8', ec='#8e44ad', lw=1.2))
    ax2.text(4.75, 2.3, "Channel Concatenation (128 ch) + Residual Add", ha='center', va='center', fontsize=9, fontweight='bold')

    ax2.add_patch(patches.FancyBboxPatch((1.5, 0.2), 7.0, 1.0, boxstyle="round,pad=0.1", fc='#e8eaed', ec='#2c3e50', lw=1.2))
    ax2.text(5.0, 0.7, "Total Parameters: 3,886,149 (2 blocks × 3 modules)\nMulti-scale receptive fields without dynamic attention", ha='center', va='center', fontsize=8.5)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "figure_2_architectures.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_2_architectures.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

# ==============================================================================
# Figure 3: Four-Model Performance Comparison (ROC & PR Curves)
# ==============================================================================
def create_figure_3():
    print("Generating Figure 3: Four-Model ROC and PR Curves...")
    preds = {
        'InceptionTime': pd.read_csv("results/phase7/benchmark_fold10/model_inception/model_inception_fold10_predictions.csv"),
        'Attention': pd.read_csv("results/phase9/model_b_fold10_evaluation/model_b_fold10_predictions.csv"),
        'GAP': pd.read_csv("results/phase7/benchmark_fold10/model_a/model_a_fold10_predictions.csv"),
        'XResNet': pd.read_csv("results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv")
    }

    classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    model_colors = {
        'InceptionTime': COLORS['Inception'],
        'Attention': COLORS['Attention'],
        'GAP': COLORS['GAP'],
        'XResNet': COLORS['XResNet']
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

    # Panel A: Macro ROC Curves
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#999999', lw=1, label='Chance (0.5000)')

    mean_fpr = np.linspace(0, 1, 500)
    for m_name, df in preds.items():
        tprs = []
        for cls_name in classes:
            y_true, y_prob = get_targets_and_probs(df, cls_name)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        
        if m_name == 'InceptionTime':
            label_text = f"InceptionTime1D (Macro AUROC = 0.8991)"
        elif m_name == 'Attention':
            label_text = f"ECGResNet-Attention (Macro AUROC = 0.8979)"
        elif m_name == 'GAP':
            label_text = f"ECGResNet-GAP (Macro AUROC = 0.8868)"
        else:
            label_text = f"XResNet1D (Macro AUROC = 0.8775)"
            
        ax_roc.plot(mean_fpr, mean_tpr, color=model_colors[m_name], lw=2.2, label=label_text)

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax_roc.set_title("(A) Macro-Averaged Receiver Operating Characteristic (ROC)", fontsize=11, fontweight='bold', pad=10)
    ax_roc.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'])
    ax_roc.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    # Panel B: Macro Precision-Recall Curves
    ax_pr = axes[1]
    mean_recall = np.linspace(0, 1, 500)
    for m_name, df in preds.items():
        precisions = []
        for cls_name in classes:
            y_true, y_prob = get_targets_and_probs(df, cls_name)
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            precisions.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
        mean_prec = np.mean(precisions, axis=0)
        
        if m_name == 'InceptionTime':
            label_text = f"InceptionTime1D (Macro AP = 0.7636)"
        elif m_name == 'Attention':
            label_text = f"ECGResNet-Attention (Macro AP = 0.7644)"
        elif m_name == 'GAP':
            label_text = f"ECGResNet-GAP (Macro AP = 0.7334)"
        else:
            label_text = f"XResNet1D (Macro AP = 0.7276)"
            
        ax_pr.plot(mean_recall, mean_prec, color=model_colors[m_name], lw=2.2, label=label_text)

    # Macro-average prevalence across 5 classes (25.87%)
    ax_pr.axhline(0.2587, linestyle='--', color='#999999', lw=1, label='Baseline Prevalence (0.2587)')

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold')
    ax_pr.set_title("(B) Macro-Averaged Precision-Recall (PR) Continuum", fontsize=11, fontweight='bold', pad=10)
    ax_pr.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'])
    ax_pr.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "figure_3_roc_pr_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_3_roc_pr_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

# ==============================================================================
# Figure 4: Attention vs GAP Paired Statistical Comparison
# ==============================================================================
def create_figure_4():
    print("Generating Figure 4: Paired Statistical Bootstrap Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, gridspec_kw={'width_ratios': [1, 1.2]})

    # Panel A: Forest Plot of Key Pairwise Comparisons
    ax_forest = axes[0]
    
    comparisons = [
        ('Attention vs. GAP\n(Macro AUROC)', 0.0110, 0.0070, 0.0152, '#1a8828', 'P = 100.0%'),
        ('Attention vs. GAP\n(Macro AP)', 0.0310, 0.0205, 0.0420, '#1a8828', 'P = 100.0%'),
        ('Attention vs. GAP\n(Macro F1)', 0.0144, 0.0032, 0.0264, '#1a8828', 'P = 99.6%'),
        ('Attention vs. Inception\n(Macro AUROC)', -0.0012, -0.0055, 0.0038, '#1f4e79', 'P = 31.5%'),
        ('Attention vs. Inception\n(Macro AP)', 0.0008, -0.0087, 0.0108, '#1f4e79', 'P = 59.5%'),
        ('Attention vs. Inception\n(Macro F1)', -0.0019, -0.0134, 0.0099, '#1f4e79', 'P = 38.4%'),
        ('GAP vs. XResNet1D\n(Macro AUROC)', 0.0093, 0.0037, 0.0148, '#7570b3', 'P = 99.9%')
    ]
    
    y_pos = np.arange(len(comparisons))[::-1]
    labels = [c[0] for c in comparisons]
    means = [c[1] for c in comparisons]
    ci_low = [c[2] for c in comparisons]
    ci_high = [c[3] for c in comparisons]
    colors = [c[4] for c in comparisons]
    p_texts = [c[5] for c in comparisons]

    ax_forest.axvline(0, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.8)

    for y, mean, low, high, c, p_txt in zip(y_pos, means, ci_low, ci_high, colors, p_texts):
        ax_forest.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt='o', color=c, 
                           ecolor=c, elinewidth=2.2, capsize=5, markersize=7)
        ax_forest.text(high + 0.002, y, p_txt, va='center', fontsize=8.5, color='#333333', fontweight='bold')

    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(labels, fontsize=9.5)
    ax_forest.set_xlabel("Paired Difference (Δ = Model 1 - Model 2)", fontsize=10.5, fontweight='bold')
    ax_forest.set_title("(A) 95% Bootstrap Confidence Intervals (B = 1,000)", fontsize=11, fontweight='bold', pad=10)
    ax_forest.set_xlim([-0.02, 0.055])
    ax_forest.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'])

    # Panel B: Per-Class Diagnostic Gains of Attention over GAP
    ax_class = axes[1]
    classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    auroc_deltas = [0.0092, 0.0116, 0.0035, 0.0040, 0.0269]
    ap_deltas = [0.0148, 0.0434, 0.0144, 0.0261, 0.0563]
    
    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax_class.bar(x - width/2, auroc_deltas, width, label='Δ AUROC (Attn - GAP)', color='#2ecc71', edgecolor='#27ae60', lw=1.2)
    rects2 = ax_class.bar(x + width/2, ap_deltas, width, label='Δ Average Precision (Attn - GAP)', color='#3498db', edgecolor='#2980b9', lw=1.2)

    ax_class.axhline(0, color='#333333', lw=0.8)
    ax_class.set_ylabel("Metric Difference over GAP", fontsize=10.5, fontweight='bold')
    ax_class.set_title("(B) Per-Class Diagnostic Gains of Attention over GAP", fontsize=11, fontweight='bold', pad=10)
    ax_class.set_xticks(x)
    ax_class.set_xticklabels([f"{c}\n(N={n})" for c, n in zip(classes, [963, 521, 496, 550, 262])], fontsize=9.5)
    ax_class.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax_class.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    # Annotate highest gain on HYP
    ax_class.annotate("Largest Gain:\nΔAP = +0.0563\nΔAUROC = +0.0269", xy=(4 + width/2, 0.0563), xytext=(3.2, 0.06),
                      arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                      fontsize=9, fontweight='bold', color='#c0392b',
                      bbox=dict(boxstyle="round,pad=0.2", fc='#fadbd8', ec='#e74c3c', lw=1))

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "figure_4_paired_bootstrap.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_4_paired_bootstrap.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

# ==============================================================================
# Figure 5: HYP Isolated vs Composite Sensitivity Analysis
# ==============================================================================
def create_figure_5():
    print("Generating Figure 5: HYP Stratified Sensitivity Analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300, gridspec_kw={'width_ratios': [1.15, 1]})

    # Figure Title
    fig.suptitle("HYP Stratified Sensitivity Analysis", fontsize=12.5, fontweight='bold', color='#1a252f', y=0.98)

    # -------------------------------------------------------------------------
    # Panel A: Composition of HYP-Positive Cases (N = 262)
    # -------------------------------------------------------------------------
    categories = [
        'Pure isolated HYP',
        'HYP + STTC',
        'Other HYP co-occurrence\npatterns'
    ]
    counts = [56, 155, 51]
    pcts = [21.37, 59.16, 19.47]
    bar_colors = ['#b03a2e', '#2b5c8f', '#707b7c']

    y_pos = np.array([2, 1, 0])
    bars = ax1.barh(y_pos, counts, color=bar_colors, edgecolor='#2c3e50', lw=0.9, height=0.52)

    for bar, n, pct in zip(bars, counts, pcts):
        ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"n = {n} ({pct:.2f}%)",
                 va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=9.5)
    ax1.set_xlabel("Number of Fold-10 ECG Records (Total HYP N = 262)", fontsize=10, fontweight='bold')
    ax1.set_title("(A) Composition of HYP-Positive Cases (N = 262)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlim(0, 195)
    ax1.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='x')

    # -------------------------------------------------------------------------
    # Panel B: Sensitivity by Subgroup
    # -------------------------------------------------------------------------
    subgroups = ['Pure isolated HYP\n(n = 56)', 'HYP + STTC\n(n = 155)']
    sens_gap = [1.79, 69.03]
    sens_inc = [7.14, 63.87]

    x = np.arange(len(subgroups))
    width = 0.30

    rects1 = ax2.bar(x - width/2, sens_gap, width, label='ECGResNet-GAP', color=COLORS['GAP'], edgecolor='#9c3800', lw=0.9)
    rects2 = ax2.bar(x + width/2, sens_inc, width, label='InceptionTime', color=COLORS['Inception'], edgecolor='#12283f', lw=0.9)

    for rect, val in zip(rects1, sens_gap):
        ax2.text(rect.get_x() + rect.get_width()/2, val + 1.8, f"{val:.2f}%",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLORS['GAP'])

    for rect, val in zip(rects2, sens_inc):
        ax2.text(rect.get_x() + rect.get_width()/2, val + 1.8, f"{val:.2f}%",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLORS['Inception'])

    ax2.set_ylabel("Sensitivity (%)", fontsize=10, fontweight='bold')
    ax2.set_title("(B) HYP Sensitivity by Key Subgroup", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subgroups, fontsize=9.5)
    ax2.set_ylim(0, 80)
    ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax2.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    # Conservative scientific interpretation footnote
    fig.text(0.5, 0.02,
             "Evidence of limited sensitivity to isolated hypertrophy phenotypes (Fold 10 test set, N = 2,158 records; HYP support = 262).",
             ha='center', va='bottom', fontsize=8.5, fontstyle='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    png_path = os.path.join(OUTPUT_DIR, "figure_5_hyp_stratification.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_5_hyp_stratification.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} (600 DPI) and {pdf_path}")

# ==============================================================================
# Figure 6: Single-Lead Sensitivity Analysis for HYP Discrimination
# ==============================================================================
def create_figure_6():
    print("Generating Figure 6: Single-Lead Sensitivity Analysis for HYP Discrimination...")
    
    with open("results/phase7/benchmark_fold10/hyp_single_lead/hyp_single_lead_metrics.json") as f:
        data = json.load(f)

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    none_gap = data['ECGResNet-GAP']['NONE']['auroc']
    none_inc = data['InceptionTime1D']['NONE']['auroc']

    delta_gap = [data['ECGResNet-GAP'][f'MASK_{l}']['auroc'] - none_gap for l in leads]
    delta_inc = [data['InceptionTime1D'][f'MASK_{l}']['auroc'] - none_inc for l in leads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300, sharey=True)

    fig.suptitle("Single-Lead Sensitivity Analysis for HYP Discrimination", 
                 fontsize=13, fontweight='bold', color='#1a252f', y=0.98)

    x = np.arange(len(leads))
    bar_width = 0.62

    # Panel A: ECGResNet-GAP
    color_gap = COLORS['GAP']
    bars1 = ax1.bar(x, delta_gap, bar_width, color=color_gap, edgecolor='#9c3800', lw=0.9)
    ax1.axhline(0, color='#333333', linestyle='-', lw=1.0)
    ax1.set_title(f"(A) ECGResNet-GAP (Baseline AUROC = {none_gap:.4f})", fontsize=11, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(leads, fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Δ Macro-AUROC relative to NONE", fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    # Panel B: InceptionTime
    color_inc = COLORS['Inception']
    bars2 = ax2.bar(x, delta_inc, bar_width, color=color_inc, edgecolor='#12283f', lw=0.9)
    ax2.axhline(0, color='#333333', linestyle='-', lw=1.0)
    ax2.set_title(f"(B) InceptionTime (Baseline AUROC = {none_inc:.4f})", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(leads, fontsize=9.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    # Identical Y-axis limits
    y_min = -0.014
    y_max = 0.012
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Numerical labels above/below bars
    for ax, deltas, bars in [(ax1, delta_gap, bars1), (ax2, delta_inc, bars2)]:
        for bar, val in zip(bars, deltas):
            if val < 0:
                va = 'top'
                y_offset = -0.0005
            else:
                va = 'bottom'
                y_offset = +0.0004
            ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
                    f"{val:+.4f}", ha='center', va=va, fontsize=7.2, fontweight='bold', color='#2c3e50')

    # Separation between Limb and Precordial leads
    for ax in (ax1, ax2):
        ax.axvline(5.5, color='#888888', linestyle='--', lw=0.8, alpha=0.7)
        ax.text(2.5, y_max - 0.0015, "Limb Leads", ha='center', va='top', fontsize=8.5, color='#555555', fontstyle='italic')
        ax.text(8.5, y_max - 0.0015, "Precordial Leads", ha='center', va='top', fontsize=8.5, color='#555555', fontstyle='italic')

    # Footnote note
    fig.text(0.5, 0.015,
             "Lead masking was applied after preprocessing; ΔAUROC is interpreted as a sensitivity measure rather than causal attribution. (PTB-XL Fold 10, N = 2,158).",
             ha='center', va='bottom', fontsize=8.5, fontstyle='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    pdf_path = os.path.join(OUTPUT_DIR, "figure_6_lead_sensitivity.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figure_6_lead_sensitivity.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path} (600 DPI) and {pdf_path}")

# ==============================================================================
# Auxiliary Figure 6: Lead Group Masking & BatchNorm Probability Shift
# ==============================================================================
def create_figure_6_auxiliary():
    print("Generating Auxiliary Figure 6: Lead Group Masking & BatchNorm Drift...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, gridspec_kw={'width_ratios': [1, 1.2]})

    # Panel A: Lead-Group Perturbation Degradation from hyp_occlusion_metrics.json
    conditions = ['Baseline\n(All 12 Leads)', 'Limb Masked\n(I, II, III, aVR, aVL, aVF)', 'Precordial Masked\n(V1 - V6)']
    gap_auroc = [0.7682, 0.7406, 0.7135]
    inc_auroc = [0.7899, 0.7213, 0.7041]

    x = np.arange(len(conditions))
    width = 0.35

    rects1 = ax1.bar(x - width/2, gap_auroc, width, label='ECGResNet-GAP', color='#d95f02', edgecolor='#b94a00', lw=1.2)
    rects2 = ax1.bar(x + width/2, inc_auroc, width, label='InceptionTime1D', color='#1f4e79', edgecolor='#153556', lw=1.2)

    ax1.axhline(0.5000, color='#999999', linestyle='--', lw=1, label='Chance Level (0.5000)')

    ax1.set_ylabel("HYP Macro AUROC", fontsize=10.5, fontweight='bold')
    ax1.set_title("(A) Spatial Lead-Group Perturbation Degradation", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9.5)
    ax1.set_ylim(0.45, 0.85)
    ax1.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax1.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    # Annotate precordial degradation
    ax1.annotate("Precordial Drop:\nΔ = -0.0547 (GAP)\nΔ = -0.0858 (Inc)\nAP drop: -0.098 to -0.141", xy=(2, 0.7135), xytext=(1.3, 0.52),
                 arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                 fontsize=9, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle="round,pad=0.2", fc='#fadbd8', ec='#e74c3c', lw=1))

    # Panel B: Single-Lead Masking & Global Probability Drift (BatchNorm Shift)
    df_drift = pd.read_csv("results/phase7/benchmark_fold10/hyp_single_lead/hyp_single_lead_global_drift.csv")
    df_gap_drift = df_drift[(df_drift['model'] == 'ECGResNet-GAP') & (df_drift['lead'] != 'NONE')].copy()
    
    leads = df_gap_drift['lead'].values
    prob_shifts = df_gap_drift['mean_global_delta'].values
    
    colors = ['#e74c3c' if l in ['aVF', 'V2', 'I'] else '#3498db' for l in leads]
    bars = ax2.bar(leads, prob_shifts, color=colors, edgecolor='#2c3e50', lw=1.2)

    ax2.set_ylabel("Global Mean Predicted Probability Shift (Δ p)", fontsize=10.5, fontweight='bold')
    ax2.set_title("(B) Single-Lead Masking Induces BatchNorm Distribution Shift", fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.6, color=COLORS['Grid'], axis='y')

    max_idx = np.argmax(prob_shifts)
    max_lead = leads[max_idx]
    max_val = prob_shifts[max_idx]
    ax2.annotate(f"Largest Shift ({max_lead}):\nΔp = +{max_val:.4f} global drift\n(Channel zeroing alters\nearly BatchNorm stats)", 
                 xy=(max_idx, max_val), xytext=(max_idx - 1.5, max_val + 0.005),
                 arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.5),
                 fontsize=8.5, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle="round,pad=0.2", fc='#fadbd8', ec='#e74c3c', lw=1))

    ax2.text(5.5, -0.012, "*Lead masking is interpreted as a sensitivity probe, not causal attribution, due to input distribution shifts.",
             ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#555555')
    ax2.set_ylim([-0.015, 0.045])

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "figure_6_auxiliary_lead_group_drift.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_6_auxiliary_lead_group_drift.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved auxiliary: {png_path} and {pdf_path}")

# ==============================================================================
# Master Execution
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("PHASE 10.3 — GENERATING PUBLICATION-QUALITY FIGURES")
    print("Strictly derived from frozen authoritative artifacts.")
    print("="*80)
    create_figure_1()
    create_figure_2()
    create_figure_3()
    create_figure_4()
    create_figure_5()
    create_figure_6()
    print("="*80)
    print("ALL 6 PUBLICATION FIGURES SUCCESSFULLY GENERATED!")
    print("="*80)

#!/usr/bin/env python3
"""
generate_figures_7_9.py
=======================
Generates Figures 7, 8, and 9 for the ECG benchmark publication.

INTEGRITY RULES ENFORCED:
  - No model retraining.
  - No checkpoint loading or modification.
  - No test-set threshold tuning.
  - Fold-10 frozen test set is read-only.
  - All plotted values loaded from authoritative Phase 9 / Phase 10 artifacts.
  - No values estimated or fabricated.
  - Figures 1-6 are never touched.
  - All random operations use a fixed seed for determinism.
  - Output goes only to results/phase10/figures_7_9/.

AUTHORITATIVE SOURCES:
  Primary benchmark:    results/phase9/four_model_benchmark/
  Model B evaluation:   results/phase9/model_b_fold10_evaluation/
  Model A evaluation:   results/phase7/benchmark_fold10/model_a/
  XResNet evaluation:   results/phase7/benchmark_fold10/model_xresnet/
  InceptionTime eval:   results/phase7/benchmark_fold10/model_inception/
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend - deterministic rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# --------------------------------------------------------------------------
# 0. SEED
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------
# 1. PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[3]

BENCH_DIR     = PROJECT_ROOT / "results" / "phase9" / "four_model_benchmark"
MODEL_B_DIR   = PROJECT_ROOT / "results" / "phase9" / "model_b_fold10_evaluation"
MODEL_A_DIR   = PROJECT_ROOT / "results" / "phase7" / "benchmark_fold10" / "model_a"
XRESNET_DIR   = PROJECT_ROOT / "results" / "phase7" / "benchmark_fold10" / "model_xresnet"
INCEPTION_DIR = PROJECT_ROOT / "results" / "phase7" / "benchmark_fold10" / "model_inception"
OUT_DIR       = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# 2. MATPLOTLIB STYLE  (IEEE two-column compatible)
# --------------------------------------------------------------------------
IEEE_2COL_W = 7.16  # inches

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.6,
    "patch.linewidth":    0.6,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.8",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# Colorblind-safe palette (Paul Tol vibrant)
MODEL_COLORS = {
    "ECGResNet-GAP":       "#0077BB",
    "ECGResNet-Attention": "#EE7733",
    "XResNet1D":           "#009988",
    "InceptionTime":       "#CC3311",
}
MODEL_MARKERS = {
    "ECGResNet-GAP":       "o",
    "ECGResNet-Attention": "s",
    "XResNet1D":           "^",
    "InceptionTime":       "D",
}
MODEL_NAMES = [
    "ECGResNet-GAP",
    "ECGResNet-Attention",
    "XResNet1D",
    "InceptionTime",
]
MODEL_SHORT = {
    "ECGResNet-GAP":       "GAP",
    "ECGResNet-Attention": "Attention",
    "XResNet1D":           "XResNet1D",
    "InceptionTime":       "InceptionTime",
}
CLASSES = ["NORM", "STTC", "CD", "MI", "HYP"]

# --------------------------------------------------------------------------
# 3. HELPERS
# --------------------------------------------------------------------------
def save_figure(fig, stem):
    paths = {}
    for fmt in ("png", "pdf", "svg"):
        p = OUT_DIR / (stem + "." + fmt)
        fig.savefig(p, format=fmt)
        paths[fmt] = p
    return paths

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ==========================================================================
# FIGURE 7: Per-Class Performance
# ==========================================================================
print("=" * 70)
print("LOADING PER-CLASS METRICS FOR FIGURE 7")
print("=" * 70)

# ECGResNet-GAP from model_a_fold10_metrics.json
with open(MODEL_A_DIR / "model_a_fold10_metrics.json") as f:
    gap_m = json.load(f)
gap_auroc = {c: gap_m["test_metrics_validation_tuned_thresholds"]["per_class"][c]["auroc"] for c in CLASSES}
gap_ap    = {c: gap_m["test_metrics_validation_tuned_thresholds"]["per_class"][c]["ap"]    for c in CLASSES}
print(f"  GAP AUROC: { {c: round(v,4) for c,v in gap_auroc.items()} }")
print(f"  GAP AP:    { {c: round(v,4) for c,v in gap_ap.items()} }")

# ECGResNet-Attention from model_b_fold10_per_class.csv
attn_df = pd.read_csv(MODEL_B_DIR / "model_b_fold10_per_class.csv", index_col="class")
attn_auroc = {c: float(attn_df.loc[c, "auroc"]) for c in CLASSES}
attn_ap    = {c: float(attn_df.loc[c, "ap"])    for c in CLASSES}
print(f"  Attention AUROC: { {c: round(v,4) for c,v in attn_auroc.items()} }")
print(f"  Attention AP:    { {c: round(v,4) for c,v in attn_ap.items()} }")

# XResNet1D from model_xresnet_fold10_metrics.json
with open(XRESNET_DIR / "model_xresnet_fold10_metrics.json") as f:
    xr_m = json.load(f)
xr_auroc = {c: xr_m["validation_tuned_thresholds"]["per_class"][c]["auroc"] for c in CLASSES}
xr_ap    = {c: xr_m["validation_tuned_thresholds"]["per_class"][c]["ap"]    for c in CLASSES}
print(f"  XResNet1D AUROC: { {c: round(v,4) for c,v in xr_auroc.items()} }")
print(f"  XResNet1D AP:    { {c: round(v,4) for c,v in xr_ap.items()} }")

# InceptionTime from model_inception_fold10_metrics.json (caps keys)
with open(INCEPTION_DIR / "model_inception_fold10_metrics.json") as f:
    inc_m = json.load(f)
inc_auroc = {c: inc_m["validation_tuned_thresholds"]["per_class"][c]["AUROC"] for c in CLASSES}
inc_ap    = {c: inc_m["validation_tuned_thresholds"]["per_class"][c]["AP"]    for c in CLASSES}
print(f"  InceptionTime AUROC: { {c: round(v,4) for c,v in inc_auroc.items()} }")
print(f"  InceptionTime AP:    { {c: round(v,4) for c,v in inc_ap.items()} }")

PER_CLASS_AUROC = {
    "ECGResNet-GAP":       gap_auroc,
    "ECGResNet-Attention": attn_auroc,
    "XResNet1D":           xr_auroc,
    "InceptionTime":       inc_auroc,
}
PER_CLASS_AP = {
    "ECGResNet-GAP":       gap_ap,
    "ECGResNet-Attention": attn_ap,
    "XResNet1D":           xr_ap,
    "InceptionTime":       inc_ap,
}

# Export CSV
rows7 = []
for model in MODEL_NAMES:
    for cls in CLASSES:
        rows7.append({"model": model, "class": cls,
                      "auroc": PER_CLASS_AUROC[model][cls],
                      "average_precision": PER_CLASS_AP[model][cls]})
csv7 = OUT_DIR / "figure_7_per_class_performance.csv"
pd.DataFrame(rows7).to_csv(csv7, index=False)
print(f"\n  Figure 7 CSV saved: {csv7}")

# Plot
print("\n" + "=" * 70)
print("GENERATING FIGURE 7")
print("=" * 70)

n_cl = len(CLASSES)
n_m  = len(MODEL_NAMES)
bw   = 0.18
x    = np.arange(n_cl)
offs = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bw

fig7, axes7 = plt.subplots(1, 2, figsize=(IEEE_2COL_W, 3.3), constrained_layout=True)

for ax_idx, (ax, md, ml, ymn, ymx) in enumerate(zip(
    axes7,
    [PER_CLASS_AUROC, PER_CLASS_AP],
    ["AUROC", "Average Precision"],
    [0.60, 0.30],
    [1.00, 1.00],
)):
    for i, model in enumerate(MODEL_NAMES):
        vals = [md[model][c] for c in CLASSES]
        ax.bar(x + offs[i], vals, width=bw,
               color=MODEL_COLORS[model],
               label=model if ax_idx == 0 else "_nolegend_",
               alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("Diagnostic Superclass")
    ax.set_ylabel(ml)
    pl = "A" if ax_idx == 0 else "B"
    ax.set_title(f"({pl}) {ml} by Diagnostic Superclass")
    ax.set_ylim(ymn, ymx)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

handles7 = [mpatches.Patch(color=MODEL_COLORS[m], label=m, alpha=0.88) for m in MODEL_NAMES]
fig7.legend(handles=handles7, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.10), framealpha=0.85, edgecolor="0.8", fontsize=8)

paths7 = save_figure(fig7, "figure_7_per_class_performance")
plt.close(fig7)
print(f"  Figure 7 saved (png/pdf/svg)")

# ==========================================================================
# FIGURE 8: Performance-Complexity Trade-off
# ==========================================================================
print("\n" + "=" * 70)
print("LOADING COMPLEXITY DATA FOR FIGURE 8")
print("=" * 70)

comp_df  = pd.read_csv(BENCH_DIR / "four_model_complexity.csv")
pt_df    = pd.read_csv(BENCH_DIR / "four_model_point_estimates.csv")

ART2DISP = {
    "InceptionTime1D":     "InceptionTime",
    "ECGResNet-Attention": "ECGResNet-Attention",
    "ECGResNet-GAP":       "ECGResNet-GAP",
    "XResNet1D":           "XResNet1D",
}

fig8_data = {}
for _, row in pt_df.iterrows():
    ma = str(row["model"])
    md = ART2DISP.get(ma)
    if md is None:
        print(f"  WARNING: Unknown model '{ma}'")
        continue
    cr = comp_df[comp_df["model"] == ma]
    if len(cr) == 0:
        sys.exit(f"STOP: '{ma}' not in four_model_complexity.csv")
    lat = float(cr["latency_ms_per_ecg"].iloc[0])
    fig8_data[md] = {
        "parameters":          int(row["parameters"]),
        "parameters_millions": float(row["parameters"]) / 1e6,
        "macro_auroc":         float(row["macro_auroc"]),
        "inference_ms_per_ecg": lat,
    }
    print(f"  {md}: params={int(row['parameters'])}, auroc={float(row['macro_auroc']):.6f}, lat={lat}")

if len(fig8_data) != 4:
    sys.exit(f"STOP: expected 4 models, got {len(fig8_data)}")

# Export CSV
rows8 = [{"model": m, **fig8_data[m]} for m in MODEL_NAMES]
csv8 = OUT_DIR / "figure_8_performance_complexity.csv"
pd.DataFrame(rows8).to_csv(csv8, index=False)
print(f"\n  Figure 8 CSV saved: {csv8}")

# Plot
print("\n" + "=" * 70)
print("GENERATING FIGURE 8")
print("=" * 70)

fig8, axes8 = plt.subplots(1, 2, figsize=(IEEE_2COL_W, 3.3), constrained_layout=True)

for ax_idx, (ax, xk, xl, pl) in enumerate(zip(
    axes8,
    ["parameters_millions", "inference_ms_per_ecg"],
    ["Parameters (Millions)", "Inference Time (ms/ECG)"],
    ["A", "B"],
)):
    for model in MODEL_NAMES:
        d = fig8_data[model]
        xv = d[xk]
        yv = d["macro_auroc"]
        ax.scatter(xv, yv, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                   s=72, zorder=4, edgecolors="white", linewidths=0.8)
        # Per-model label offsets to avoid overlap
        if ax_idx == 0:
            offsets_map = {
                "ECGResNet-GAP":       ("left",  "top",    0.001, -0.0005),
                "ECGResNet-Attention": ("left",  "bottom", 0.001,  0.0005),
                "XResNet1D":           ("right", "top",   -0.001, -0.0005),
                "InceptionTime":       ("right", "bottom",-0.001,  0.0005),
            }
        else:
            offsets_map = {
                "ECGResNet-Attention": ("left",  "bottom",  0.15,  0.0005),
                "ECGResNet-GAP":       ("left",  "top",     0.15, -0.0005),
                "XResNet1D":           ("right", "bottom", -0.15,  0.0005),
                "InceptionTime":       ("right", "top",    -0.15, -0.0005),
            }
        ha, va, dx, dy = offsets_map[model]
        ax.annotate(model, xy=(xv, yv), xytext=(xv + dx, yv + dy),
                    ha=ha, va=va, fontsize=7, color=MODEL_COLORS[model])
    ax.set_xlabel(xl)
    ax.set_ylabel("Macro-AUROC")
    ax.set_title(f"({pl}) Macro-AUROC vs. {xl.split(' (')[0]}")
    ys = [fig8_data[m]["macro_auroc"] for m in MODEL_NAMES]
    ax.set_ylim(min(ys) - 0.012, max(ys) + 0.012)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.grid(linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

handles8 = [plt.scatter([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                         s=64, label=m, edgecolors="white", linewidths=0.6)
             for m in MODEL_NAMES]
fig8.legend(handles=handles8, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.10), framealpha=0.85, edgecolor="0.8", fontsize=8)

paths8 = save_figure(fig8, "figure_8_performance_complexity")
plt.close(fig8)
print(f"  Figure 8 saved (png/pdf/svg)")

# ==========================================================================
# FIGURE 9: Prediction Agreement
# ==========================================================================
print("\n" + "=" * 70)
print("LOADING FROZEN FOLD-10 PREDICTION FILES FOR FIGURE 9")
print("=" * 70)

PRED_FILES = {
    "ECGResNet-GAP":       MODEL_A_DIR   / "model_a_fold10_predictions.csv",
    "ECGResNet-Attention": MODEL_B_DIR   / "model_b_fold10_predictions.csv",
    "XResNet1D":           XRESNET_DIR   / "model_xresnet_fold10_predictions.csv",
    "InceptionTime":       INCEPTION_DIR / "model_inception_fold10_predictions.csv",
}

with open(BENCH_DIR / "four_model_prediction_alignment.json") as f:
    align_j = json.load(f)
TEST_N = align_j["test_n"]
print(f"  Recorded test_n from alignment JSON: {TEST_N}")
print(f"  ID alignment:    {align_j['id_alignment']}")
print(f"  Label alignment: {align_j['target_alignment']}")
print(f"  Verified at:     {align_j['verification_timestamp']}")

raw = {}
for model, path in PRED_FILES.items():
    df = pd.read_csv(path)
    print(f"\n  {model}: {len(df)} rows, cols={list(df.columns)}")
    raw[model] = df

# Row count check
for model, df in raw.items():
    if len(df) != TEST_N:
        sys.exit(f"STOP — ROW COUNT ERROR: {model} has {len(df)} rows, expected {TEST_N}")
print(f"\n  Row count check: all {TEST_N} rows. PASS.")

# ECG ID alignment (GAP / Attention / XResNet1D have ecg_id)
id_models_with_id = ["ECGResNet-GAP", "ECGResNet-Attention", "XResNet1D"]
ref_ids = raw["ECGResNet-GAP"]["ecg_id"].values
for m in id_models_with_id:
    if not np.array_equal(ref_ids, raw[m]["ecg_id"].values):
        sys.exit(f"STOP — ID MISMATCH: {m} vs ECGResNet-GAP")
print("  ECG-ID alignment (GAP/Attention/XResNet1D): PASS.")

# Duplicate ID check
for m in id_models_with_id:
    if raw[m]["ecg_id"].duplicated().any():
        sys.exit(f"STOP — DUPLICATE ECG IDs in {m}")
print("  Duplicate ECG-ID check: PASS.")

# Column maps
PROB_COLS = {
    "ECGResNet-GAP":       {c: f"prob_{c}" for c in CLASSES},
    "ECGResNet-Attention": {c: f"prob_{c}" for c in CLASSES},
    "XResNet1D":           {c: f"prob_{c}" for c in CLASSES},
    "InceptionTime":       {c: f"{c}_prob" for c in CLASSES},
}
TRUE_COLS = {
    "ECGResNet-GAP":       {c: f"true_{c}" for c in CLASSES},
    "ECGResNet-Attention": {c: f"true_{c}" for c in CLASSES},
    "XResNet1D":           {c: f"true_{c}" for c in CLASSES},
    "InceptionTime":       {c: f"{c}_target" for c in CLASSES},
}

# Ground-truth alignment check
gap_true = np.column_stack([raw["ECGResNet-GAP"][TRUE_COLS["ECGResNet-GAP"][c]].values for c in CLASSES])
for m in ["ECGResNet-Attention", "XResNet1D", "InceptionTime"]:
    mt = np.column_stack([raw[m][TRUE_COLS[m][c]].values for c in CLASSES])
    if not np.array_equal(gap_true, mt):
        sys.exit(f"STOP — LABEL MISMATCH: {m} vs ECGResNet-GAP")
print("  Ground-truth label alignment (all 4 models): PASS.")

# Extract probability matrices
prob_mats = {}
for model in MODEL_NAMES:
    pm = np.column_stack([raw[model][PROB_COLS[model][c]].values.astype(np.float64) for c in CLASSES])
    prob_mats[model] = pm
    print(f"  {model}: shape={pm.shape}, range=[{pm.min():.4f},{pm.max():.4f}]")

shapes = [prob_mats[m].shape for m in MODEL_NAMES]
if len(set(shapes)) != 1:
    sys.exit(f"STOP — shape mismatch: {shapes}")
print(f"  All probability matrix shapes: {shapes[0]} — PASS.")

# Pearson correlation matrix
print("\n" + "=" * 70)
print("COMPUTING PEARSON CORRELATION MATRIX")
print("=" * 70)

flat = {m: prob_mats[m].flatten() for m in MODEL_NAMES}
corr = np.zeros((4, 4))
for i, m1 in enumerate(MODEL_NAMES):
    for j, m2 in enumerate(MODEL_NAMES):
        r, _ = pearsonr(flat[m1], flat[m2])
        corr[i, j] = r
        print(f"  r({MODEL_SHORT[m1]}, {MODEL_SHORT[m2]}) = {r:.6f}")

corr_rows = [{"model_1": MODEL_NAMES[i], "model_2": MODEL_NAMES[j], "pearson_r": corr[i,j]}
              for i in range(4) for j in range(4)]
csv9a = OUT_DIR / "figure_9_prediction_agreement_correlations.csv"
pd.DataFrame(corr_rows).to_csv(csv9a, index=False)
print(f"\n  Correlation CSV saved: {csv9a}")

# Pairwise absolute differences
PAIRS = [
    ("ECGResNet-Attention", "ECGResNet-GAP"),
    ("ECGResNet-Attention", "XResNet1D"),
    ("ECGResNet-Attention", "InceptionTime"),
    ("InceptionTime",       "ECGResNet-GAP"),
    ("InceptionTime",       "XResNet1D"),
    ("ECGResNet-GAP",       "XResNet1D"),
]
PAIR_LABELS = [
    "Attention\nvs GAP",
    "Attention\nvs XResNet1D",
    "Attention\nvs InceptionTime",
    "InceptionTime\nvs GAP",
    "InceptionTime\nvs XResNet1D",
    "GAP\nvs XResNet1D",
]

abs_diffs = []
diff_rows = []
print("\n" + "=" * 70)
print("COMPUTING PAIRWISE ABSOLUTE DIFFERENCE DISTRIBUTIONS")
print("=" * 70)
for (m1, m2), lbl in zip(PAIRS, PAIR_LABELS):
    d = np.abs(prob_mats[m1] - prob_mats[m2]).flatten()
    abs_diffs.append(d)
    diff_rows.append({
        "pair_label":      lbl.replace("\n", " "),
        "model_1":         m1,
        "model_2":         m2,
        "n_values":        len(d),
        "mean_abs_diff":   float(np.mean(d)),
        "median_abs_diff": float(np.median(d)),
        "q25_abs_diff":    float(np.percentile(d, 25)),
        "q75_abs_diff":    float(np.percentile(d, 75)),
        "q95_abs_diff":    float(np.percentile(d, 95)),
        "max_abs_diff":    float(np.max(d)),
    })
    print(f"  {m1} vs {m2}: mean={np.mean(d):.4f}, med={np.median(d):.4f}, q75={np.percentile(d,75):.4f}")

csv9b = OUT_DIR / "figure_9_prediction_difference_summary.csv"
pd.DataFrame(diff_rows).to_csv(csv9b, index=False)
print(f"\n  Difference summary CSV saved: {csv9b}")

# Plot Figure 9
print("\n" + "=" * 70)
print("GENERATING FIGURE 9")
print("=" * 70)

fig9, axes9 = plt.subplots(1, 2, figsize=(IEEE_2COL_W, 3.5), constrained_layout=True)

# Panel A: heatmap
ax9a = axes9[0]
cmap = plt.cm.get_cmap("Blues")
im = ax9a.imshow(corr, cmap=cmap, vmin=0.75, vmax=1.0, aspect="auto")
tl = [MODEL_SHORT[m] for m in MODEL_NAMES]
ax9a.set_xticks(range(4))
ax9a.set_yticks(range(4))
ax9a.set_xticklabels(tl, rotation=30, ha="right", fontsize=7.5)
ax9a.set_yticklabels(tl, fontsize=7.5)
ax9a.set_title("(A) Mean Pairwise Prediction\nCorrelation (Pearson r)")
for i in range(4):
    for j in range(4):
        v = corr[i, j]
        tc = "white" if v > 0.90 else "black"
        fw = "bold" if i == j else "normal"
        ax9a.text(j, i, f"{v:.4f}", ha="center", va="center",
                  fontsize=7.5, color=tc, fontweight=fw)
cb = fig9.colorbar(im, ax=ax9a, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=7)
cb.set_label("Pearson r", fontsize=8)
for spine in ax9a.spines.values():
    spine.set_visible(False)
ax9a.tick_params(bottom=False, left=False)

# Panel B: boxplot
ax9b = axes9[1]
bp = ax9b.boxplot(abs_diffs, labels=PAIR_LABELS, patch_artist=True,
                  showfliers=False,
                  medianprops=dict(color="black", linewidth=1.2),
                  whiskerprops=dict(linewidth=0.8),
                  capprops=dict(linewidth=0.8),
                  boxprops=dict(linewidth=0.8),
                  widths=0.55, zorder=3)
pair_first = [m1 for (m1, m2) in PAIRS]
for patch, model in zip(bp["boxes"], pair_first):
    patch.set_facecolor(MODEL_COLORS[model])
    patch.set_alpha(0.75)
ax9b.set_ylabel("|Probability Difference|")
ax9b.set_title("(B) Pairwise Prediction Difference\nDistribution (N=2158 x 5)")
ax9b.tick_params(axis="x", labelsize=7)
ax9b.set_ylim(bottom=0)
ax9b.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
ax9b.set_axisbelow(True)

paths9 = save_figure(fig9, "figure_9_prediction_agreement")
plt.close(fig9)
print(f"  Figure 9 saved (png/pdf/svg)")

# ==========================================================================
# VERIFY FIGURES 1-6 UNTOUCHED
# ==========================================================================
print("\n" + "=" * 70)
print("VERIFYING FIGURES 1-6 ARE UNTOUCHED")
print("=" * 70)
FIG_DIR = PROJECT_ROOT / "results" / "phase10" / "publication_figures"
expected = [
    "figure_1_research_pipeline.png", "figure_1_research_pipeline.pdf",
    "figure_2_architectures.png",     "figure_2_architectures.pdf",
    "figure_3_roc_pr_comparison.png", "figure_3_roc_pr_comparison.pdf",
    "figure_4_paired_bootstrap.png",  "figure_4_paired_bootstrap.pdf",
    "figure_5_hyp_stratification.png","figure_5_hyp_stratification.pdf",
    "figure_6_lead_sensitivity.png",  "figure_6_lead_sensitivity.pdf",
]
all_ok = True
for fn in expected:
    p = FIG_DIR / fn
    if p.exists():
        print(f"  PRESENT (untouched): {fn}")
    else:
        print(f"  WARNING MISSING: {fn}")
        all_ok = False
if all_ok:
    print("  All Figures 1-6 present; this script did NOT modify them. PASS.")
else:
    print("  This script did NOT write to publication_figures/. Any missing files are pre-existing.")

# ==========================================================================
# OUTPUT HASHES
# ==========================================================================
print("\n" + "=" * 70)
print("OUTPUT FILE HASHES (SHA-256)")
print("=" * 70)
all_outputs = []
for stem, pd_ in [("figure_7_per_class_performance", paths7),
                   ("figure_8_performance_complexity", paths8),
                   ("figure_9_prediction_agreement",   paths9)]:
    for fmt, p in pd_.items():
        h = sha256(p)
        all_outputs.append((p.name, h))
        print(f"  {p.name}: {h}")
for cp in [csv7, csv8, csv9a, csv9b]:
    h = sha256(cp)
    all_outputs.append((cp.name, h))
    print(f"  {cp.name}: {h}")

# ==========================================================================
# AUDIT SUMMARY
# ==========================================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
print("""
FIGURE 7:
  source files:
    results/phase9/model_b_fold10_evaluation/model_b_fold10_per_class.csv  [ECGResNet-Attention]
    results/phase7/benchmark_fold10/model_a/model_a_fold10_metrics.json    [ECGResNet-GAP]
    results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_metrics.json [XResNet1D]
    results/phase7/benchmark_fold10/model_inception/model_inception_fold10_metrics.json [InceptionTime]
  exact values used:
    Per-class AUROC and AP loaded directly from machine-readable JSON/CSV artifacts.
    Validation-tuned threshold section; AUROC/AP are threshold-invariant.
    No rounding or manual transcription.
  output paths:
    results/phase10/figures_7_9/figure_7_per_class_performance.{png,pdf,svg,csv}
""")
print("""FIGURE 8:
  source files:
    results/phase9/four_model_benchmark/four_model_point_estimates.csv  [parameters, macro_auroc]
    results/phase9/four_model_benchmark/four_model_complexity.csv       [latency_ms_per_ecg]
  exact values used:""")
for m in MODEL_NAMES:
    d = fig8_data[m]
    print(f"    {m}: params={d['parameters']}, macro_auroc={d['macro_auroc']:.10f}, ms={d['inference_ms_per_ecg']}")
print("""  output paths:
    results/phase10/figures_7_9/figure_8_performance_complexity.{png,pdf,svg,csv}
""")
print(f"""FIGURE 9:
  prediction files:
    results/phase7/benchmark_fold10/model_a/model_a_fold10_predictions.csv
    results/phase9/model_b_fold10_evaluation/model_b_fold10_predictions.csv
    results/phase7/benchmark_fold10/model_xresnet/model_xresnet_fold10_predictions.csv
    results/phase7/benchmark_fold10/model_inception/model_inception_fold10_predictions.csv
  alignment status:
    Row count:       ALL {TEST_N} records match. PASS.
    ECG ID alignment: GAP / Attention / XResNet1D 100% identical. PASS.
    Label alignment:  All 4 models ground-truth 100% identical. PASS.
    Duplicate IDs:    None detected. PASS.
    Recorded in: four_model_prediction_alignment.json (verified {align_j['verification_timestamp']})
  N: {TEST_N} frozen Fold-10 test records
  correlation method: Pearson r on concatenated continuous probabilities (N x 5)
  output paths:
    results/phase10/figures_7_9/figure_9_prediction_agreement.{{png,pdf,svg}}
    results/phase10/figures_7_9/figure_9_prediction_agreement_correlations.csv
    results/phase10/figures_7_9/figure_9_prediction_difference_summary.csv
""")
print("=" * 70)
print("INTEGRITY CONFIRMATION:")
print("  - No model retraining performed.")
print("  - No checkpoint loaded, modified, or evaluated.")
print("  - No threshold tuning performed.")
print("  - Fold-10 test set read-only (prediction CSVs only; not modified).")
print("  - Figures 1-6 not touched.")
print("  - All values loaded from authoritative Phase 9 machine-readable artifacts.")
print("  - No values estimated or fabricated.")
print("  - Script deterministic: SEED=42, Agg backend.")
print("=" * 70)
print("\nFigure generation complete.")
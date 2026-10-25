#!/usr/bin/env python3
"""
STEM Track Prediction — Figure Generation Script
Journal paper: Computer Applications in Engineering Education (CAEE)

Reads results.pkl (saved by stem_analysis.py) and generates all 7 figures.
Run this whenever you need to change figure appearance — no recomputation needed.

Usage:
    python stem_analysis.py          # run once (computation + results.pkl)
    python stem_plots.py             # run as needed for figure tweaks

Requirements:
    pip install numpy pandas matplotlib seaborn shap
"""

import pickle, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works without display
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay

# ── Publication-quality plot settings ──
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})

# ── Load saved results ──
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PKL_FILE = os.path.join(REPO_ROOT, 'results.pkl')
if not os.path.isfile(PKL_FILE):
    raise FileNotFoundError(
        f"'{PKL_FILE}' not found. Run stem_analysis.py first to generate it."
    )

with open(PKL_FILE, 'rb') as f:
    R = pickle.load(f)

print("Loaded results.pkl — generating figures...\n")

# ═══════════════════════════════════════════════════════════════════════
# Figure 1: STEM vs Non-STEM distribution
# ═══════════════════════════════════════════════════════════════════════
counts = R['target_counts']
n_total = R['n_total']

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(['Non-STEM', 'STEM'], [counts[0], counts[1]],
              color=['#5B9BD5', '#ED7D31'], edgecolor='black', width=0.5)
for bar, val in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f'{val}\n({val/n_total*100:.1f}%)', ha='center', va='bottom', fontsize=12)
ax.set_ylabel('Number of Students')
# No title — caption will be in the paper
ax.set_ylim(0, max(counts)*1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_distribution.png'))
plt.close()
print("✓ Figure 1 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 4: ROC curves
# ═══════════════════════════════════════════════════════════════════════
model_order = R['model_order']
roc_data = R['roc_data']
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47', '#FFC000']

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, color in zip(model_order, colors):
    rd = roc_data[name]
    ax.plot(rd['fpr'], rd['tpr'],
            label=f"{name} (AUC = {rd['auc']:.3f})", color=color, linewidth=1.5)

rd_b = roc_data['Baseline (Dummy)']
ax.plot(rd_b['fpr'], rd_b['tpr'],
        label='Baseline (AUC = 0.500)', color='#B4B4B4', linestyle=':', linewidth=1.2)
ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
# No title — caption will be in the paper
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_roc_curves.png'))
plt.close()
print("✓ Figure 4 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Confusion matrix (best model)
# ═══════════════════════════════════════════════════════════════════════
cm = R['confusion_matrix']
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Non-STEM', 'STEM'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
# No title — caption will be in the paper
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_confusion_matrix.png'))
plt.close()
print("✓ Figure 5 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Learning curves
# ═══════════════════════════════════════════════════════════════════════
lc_data = R['learning_curve_data']
colors_lc = {'Logistic Regression': '#4472C4', 'KNN': '#ED7D31',
             'SVM': '#A5A5A5', 'Random Forest': '#70AD47', 'XGBoost': '#FFC000'}

fig, ax = plt.subplots(figsize=(7.5, 5))
for name in model_order:
    d = lc_data[name]
    ax.plot(d['train_sizes'], d['val_mean'], 'o-', label=name,
            color=colors_lc[name], linewidth=1.5, markersize=4)
    ax.fill_between(d['train_sizes'], d['val_mean'] - d['val_std'],
                    d['val_mean'] + d['val_std'],
                    alpha=0.15, color=colors_lc[name])

ax.set_xlabel('Training Set Size')
ax.set_ylabel('F1 Score (weighted)')
# No title — caption will be in the paper
ax.legend(fontsize=11)
ax.set_ylim(0.5, 0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_learning_curves.png'))
plt.close()
print("✓ Figure 6 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 7: SHAP beeswarm
# ═══════════════════════════════════════════════════════════════════════
plt.close('all')
shap.summary_plot(R['shap_values_stem'], R['X_test_trans'],
                  feature_names=R['feat_rf_en'], show=False, max_display=15,
                  plot_size=(10, 7))
fig = plt.gcf()
# No title — caption will be in the paper
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_shap_beeswarm.png'), bbox_inches='tight')
plt.close('all')
print("✓ Figure 7 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 8: SHAP bar plot (top 15)
# ═══════════════════════════════════════════════════════════════════════
top15_shap = R['top15_shap']

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.barplot(data=top15_shap, x='Mean |SHAP|', y='Feature', ax=ax, color='#5B9BD5')
ax.set_xlabel('Mean |SHAP Value|')
ax.set_ylabel('')
# No title — caption will be in the paper
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_shap_bar.png'))
plt.close()
print("✓ Figure 8 saved.")

# ═══════════════════════════════════════════════════════════════════════
# Figure 9: RF vs XGBoost feature importance
# ═══════════════════════════════════════════════════════════════════════
top15_fi = R['top15_fi']

fig, ax = plt.subplots(figsize=(7.5, 5.5))
plot_melt = top15_fi.melt(
    id_vars='Feature', value_vars=['Importance_RF', 'Importance_XGB'],
    var_name='Model', value_name='Importance',
)
plot_melt['Model'] = plot_melt['Model'].map({
    'Importance_RF': 'Random Forest', 'Importance_XGB': 'XGBoost',
})
sns.barplot(data=plot_melt, x='Importance', y='Feature', hue='Model', ax=ax,
            palette={'Random Forest': '#70AD47', 'XGBoost': '#FFC000'})
ax.set_xlabel('Feature Importance')
ax.set_ylabel('')
# No title — caption will be in the paper
ax.legend(title='')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_feature_importance.png'))
plt.close()
print("✓ Figure 9 saved.")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print("="*60)
print("  fig1_distribution.png        — STEM vs Non-STEM distribution")
print("  fig4_roc_curves.png          — ROC curves for all models")
print("  fig5_confusion_matrix.png    — Confusion matrix (best model)")
print("  fig6_learning_curves.png     — Learning curves (F1 vs training size)")
print("  fig7_shap_beeswarm.png       — SHAP beeswarm summary plot")
print("  fig8_shap_bar.png            — Mean |SHAP| bar plot (top 15)")
print("  fig9_feature_importance.png  — RF vs XGBoost feature importance")
print("\n(Figures 2 & 3 are schematic diagrams — created separately.)")

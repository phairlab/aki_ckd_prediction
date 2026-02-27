"""
Unified plot styling for the AKI-CKD prediction project.

Call setup_global_style() once at pipeline start to set matplotlib
rcParams.  Use the helper functions for consistent colors across
all figures.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def setup_global_style():
    """Apply publication-quality matplotlib settings from config."""
    c = config.PLOT_CONFIG
    plt.rcParams.update({
        "figure.facecolor": c["figure_facecolor"],
        "axes.facecolor": c["axes_facecolor"],
        "axes.edgecolor": c["axes_edgecolor"],
        "axes.linewidth": c["axes_linewidth"],
        "axes.grid": False,
        "font.family": c["font_family"],
        "font.size": c["font_size"],
        "axes.labelsize": c["axes_label_size"],
        "axes.titlesize": c["axes_title_size"],
        "axes.titleweight": "bold",
        "xtick.labelsize": c["tick_label_size"],
        "ytick.labelsize": c["tick_label_size"],
        "legend.fontsize": c["legend_font_size"],
    })


def get_outcome_colors():
    """Color mapping for CKD outcome (matches manuscript).

    Returns dict with keys 'positive' (CKD) and 'negative' (non-CKD).
    """
    return {
        "positive": "#1f77b4",   # blue  — CKD positive
        "negative": "#d62728",   # red   — non-CKD
    }


def get_model_color_palette():
    """Distinct colors for multi-model comparison plots."""
    return {
        "xgb_alberta_score": "#1f77b4",
        "xgb_alberta_raw": "#ff7f0e",
        "transformer_alberta_raw": "#2ca02c",
        "xgb_expanded": "#d62728",
        "transformer_expanded": "#9467bd",
        "transformer_egfr": "#8c564b",
    }


def get_shap_colormap():
    """Blue -> light gray -> orange colormap for SHAP plots."""
    colors_custom = ["#1f77b4", "#f0f0f0", "#ff7f0e"]
    return LinearSegmentedColormap.from_list("paper_style", colors_custom)


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Remove top/right spines and optionally set labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

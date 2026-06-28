# ============================================================
#   utils/visualize.py — All Plots & Charts
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PLOT_DPI, PLOT_STYLE, COLORS

plt.style.use(PLOT_STYLE)


def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"   📊 Saved → {path}")
    return path


def plot_feature_distributions(df):
    """Plot histogram for every column in the DataFrame."""
    print("\n📊 Plotting Feature Distributions...")
    cols = df.columns
    n    = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 12))
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
    for i, col in enumerate(cols):
        ax = axes[i // 3][i % 3]
        df[col].hist(bins=40, ax=ax, color=COLORS["primary"], edgecolor="white")
        ax.set_title(col, fontsize=10)
    # hide unused subplots
    for j in range(i + 1, (n + 2) // 3 * 3):
        axes[j // 3][j % 3].set_visible(False)
    plt.tight_layout()
    return save_fig("feature_distributions.png")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap of all features."""
    print("\n🔥 Plotting Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, square=True, linewidths=0.5,
                annot_kws={"size": 9})
    plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_fig("correlation_heatmap.png")


def plot_target_distribution(y):
    """Plot distribution of the target variable."""
    print("\n🎯 Plotting Target Distribution...")
    plt.figure(figsize=(8, 4))
    sns.histplot(y, bins=50, kde=True, color=COLORS["secondary"])
    plt.title("Target Distribution: Median House Value", fontweight="bold")
    plt.xlabel("House Value ($100K)")
    plt.ylabel("Count")
    plt.tight_layout()
    return save_fig("target_distribution.png")


def plot_model_comparison(summary_df):
    """Bar charts comparing all models across MAE, RMSE, R²."""
    print("\n📈 Plotting Model Comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    metrics = ["MAE", "RMSE", "R2"]
    colors  = [COLORS["error"], COLORS["secondary"], COLORS["success"]]
    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.barh(summary_df["Model"], summary_df[metric],
                       color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel(metric)
    plt.tight_layout()
    return save_fig("model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Scatter plot of actual vs predicted values."""
    print(f"\n🔵 Plotting Actual vs Predicted ({model_name})...")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.25, color=COLORS["primary"], s=10)
    lim = [min(y_test.min(), y_pred.min()) - 0.2,
           max(y_test.max(), y_pred.max()) + 0.2]
    plt.plot(lim, lim, "r--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Value ($100K)", fontsize=11)
    plt.ylabel("Predicted Value ($100K)", fontsize=11)
    plt.title(f"Actual vs Predicted — {model_name}", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    return save_fig("actual_vs_predicted.png")


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    """Horizontal bar chart for feature importances."""
    print(f"\n⭐ Plotting Feature Importance ({model_name})...")
    importances = pd.Series(model.feature_importances_,
                            index=feature_names).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color=COLORS["accent"])
    plt.title(f"Feature Importance — {model_name}", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    return save_fig("feature_importance.png")


def plot_residuals(y_test, y_pred, model_name):
    """Residual plot to check model errors."""
    print(f"\n📉 Plotting Residuals ({model_name})...")
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.25, color=COLORS["primary"], s=10)
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    # Distribution of Residuals
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1],
                 color=COLORS["secondary"])
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residuals")
    fig.suptitle(f"Residual Analysis — {model_name}", fontweight="bold")
    plt.tight_layout()
    return save_fig("residuals.png")


if __name__ == "__main__":
    print("Run main.py to generate all plots.")

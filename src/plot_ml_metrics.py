"""
Generate ML performance visualizations: SHAP plots, confusion matrix, calibration curve.

Run: `python src/plot_ml_metrics.py`
Outputs saved to models/*.png and models/ml_metrics.json.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import shap

logger = logging.getLogger(__name__)
plt.style.use("dark_background")
sns.set_palette("husl")


def load_artifacts():
    """Load model, features, scaler, and test data."""
    model = joblib.load("models/rf_model.pkl")
    feature_cols = joblib.load("models/model_features.pkl")
    scaler_path = "models/scaler.pkl"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    df = pd.read_csv("data/processed_data.csv")
    split_index = int(len(df) * 0.8)
    df_test = df.iloc[split_index:].copy()

    X_test = df_test[feature_cols].values
    y_test = df_test["target"].values

    if scaler is not None:
        X_test = scaler.transform(X_test)

    return model, feature_cols, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, output_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Out-of-Sample Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_feature_importance(
    model, feature_cols, output_path="models/feature_importance.png"
):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_cols))
    top_features = [feature_cols[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), top_importances[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_shap_summary(
    model, X_test, feature_cols, output_path="models/shap_summary.png"
):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_cols,
            plot_type="dot",
            show=False,
            ax=ax,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", output_path)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)


def plot_calibration_curve(y_true, y_proba, output_path="models/calibration_curve.png"):
    probs = y_proba[:, 1]
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sums = np.bincount(bin_indices, weights=probs, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracy = np.where(bin_counts > 5, bin_true / bin_counts, np.nan)
    avg_confidence = np.where(bin_counts > 5, bin_sums / bin_counts, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect Calibration")
    ax.scatter(
        avg_confidence,
        bin_accuracy,
        s=bin_counts * 5,
        alpha=0.7,
        color="#3fb950",
        label="Model Bins",
    )
    ax.set_xlabel("Predicted Probability (Confidence)")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve — Out-of-Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def generate_all():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Loading artifacts...")
    model, feature_cols, X_test, y_test = load_artifacts()

    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    logger.info("Generating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_cols)
    plot_shap_summary(model, X_test, feature_cols)
    plot_calibration_curve(y_test, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "precision_up": report["1"]["precision"],
        "recall_up": report["1"]["recall"],
        "f1_up": report["1"]["f1-score"],
        "precision_down": report["0"]["precision"],
        "recall_down": report["0"]["recall"],
        "f1_down": report["0"]["f1-score"],
        "support_total": int(report["macro avg"]["support"]),
    }

    with open("models/ml_metrics.json", "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    logger.info("✓ All diagnostic plots and metrics saved to models/")
    logger.info(
        "Accuracy: %.2f%% | Precision(UP): %.2f | Recall(UP): %.2f",
        metrics["accuracy"] * 100,
        metrics["precision_up"],
        metrics["recall_up"],
    )


if __name__ == "__main__":
    generate_all()

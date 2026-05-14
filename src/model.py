"""
Model training, prediction, evaluation, and drift detection.

Uses Random Forest with tuned hyperparameters and proper train/test split.
"""

import json
import logging
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS, add_indicators

logger = logging.getLogger(__name__)

MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/model_features.pkl"
METRICS_PATH = "models/ml_metrics.json"


def train(
    df: pd.DataFrame,
    sl_atr_mult: float = 1.0,
    tp_atr_mult: float = 2.0,
    max_hold_candles: int = 12,
    train_ratio: float = 0.8,
) -> dict:
    """
    Train Random Forest on OHLCV data.

    Target: TP hit before SL within max_hold_candles forward bars.
    Split is temporal (80/20). Scaler is fit on training data only.
    Saves model, scaler, feature list, and metrics.

    Returns metrics dict.
    """
    df = add_indicators(df, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, max_hold_candles=max_hold_candles)
    df_clean = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    logger.info("Clean rows after indicators: %d", len(df_clean))

    split_idx = int(len(df_clean) * train_ratio)
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["target"].values
    X_test = df_test[FEATURE_COLS]
    y_test = df_test["target"].values

    # Fit scaler on training data only (prevent leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info("Train: %d rows | Test: %d rows | Features: %d", len(X_train), len(X_test), len(FEATURE_COLS))

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f", acc, prec, rec, f1)

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 features: %s", feat_imp[:5])

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_COLS, FEATURES_PATH)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": len(FEATURE_COLS),
        "feature_importance": {k: round(float(v), 4) for k, v in feat_imp},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_feature_importance(feat_imp)
    _plot_confusion_matrix(y_test, y_pred)

    logger.info("Model saved to %s", MODEL_PATH)
    return metrics


def predict(X: pd.DataFrame) -> tuple[int, float]:
    """
    Load model artifacts and predict on a single feature row.

    Returns (prediction: 0|1, confidence: float).
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    # Align columns to training order
    X_aligned = pd.DataFrame([{col: float(X.get(col, 0.0)) for col in feature_cols}])
    X_scaled = scaler.transform(X_aligned)

    probas = model.predict_proba(X_scaled)[0]
    pred = int(model.predict(X_scaled)[0])
    confidence = float(probas[pred])
    return pred, confidence


def load_artifacts():
    """Load and return (model, scaler, feature_cols) tuple."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols


def check_drift(recent_confidences: list[float], threshold: float = 0.50) -> bool:
    """Return True if average recent confidence has dropped below threshold."""
    if not recent_confidences:
        return False
    avg = float(np.mean(recent_confidences))
    if avg < threshold:
        logger.warning("Drift detected: avg confidence %.3f < %.3f", avg, threshold)
        return True
    return False


def load_metrics() -> dict:
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _plot_feature_importance(feat_imp: list[tuple]):
    names = [f for f, _ in feat_imp]
    values = [v for _, v in feat_imp]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#3fb950")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Custom colormap for better contrast
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["FLAT", "LONG"], color="white", fontsize=12, weight="bold")
    ax.set_yticklabels(["FLAT", "LONG"], color="white", fontsize=12, weight="bold")

    # Add text annotations with contrasting colors
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            # Light text for dark cells, dark text for light cells
            text_color = "white" if cm[i, j] < cm.max() / 2 else "#0d1117"
            ax.text(j, i, str(val), ha="center", va="center",
                   color=text_color, fontsize=16, weight="bold")

    ax.set_xlabel("Predicted", color="white", fontsize=12, weight="bold")
    ax.set_ylabel("Actual", color="white", fontsize=12, weight="bold")
    ax.set_title("Confusion Matrix\n(TP hits before SL within 12 candles)",
                color="white", fontsize=13, weight="bold", pad=15)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", color="white")
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

"""
Model Comparison Framework — compare Random Forest vs XGBoost vs LSTM.

Run: python src/compare_models.py

Outputs:
- models/comparison_metrics.json
- models/model_comparison.png (bar chart)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# XGBoost (optional if installed)
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping. Install with: pip install xgboost")

# LSTM via PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed — skipping LSTM.")


logger = logging.getLogger(__name__)


def load_data():
    """Load and split data."""
    df = pd.read_csv("data/processed_data.csv")
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    return df_train, df_test


def prepare_X_y(df, feature_cols, scaler=None):
    """Extract features, apply optional scaling."""
    X = df[feature_cols].values
    y = df["target"].values
    if scaler is not None:
        X = scaler.fit_transform(X)
    return X, y


class SimpleLSTM(nn.Module):
    """Minimal LSTM for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(last))


def train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train LSTM model."""
    if not TORCH_AVAILABLE:
        return None

    # Reshape for LSTM: (samples, timesteps, features)
    # We'll use 5-timestep sequences
    seq_len = 5
    n_features = X_train.shape[1]

    def create_sequences(X, y):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)

    if len(X_train_seq) == 0:
        return None

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SimpleLSTM(input_size=n_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_seq = torch.FloatTensor(X_val_seq)
        preds = model(val_seq).squeeze().numpy()
        y_pred = (preds > 0.5).astype(int)
        acc = accuracy_score(y_val_seq, y_pred)

    return {"accuracy": acc, "predictions": y_pred, "model": model}


def evaluate_model(name, y_true, y_pred, y_proba=None):
    """Compute metrics dict."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    logger.info(
        "%s — Acc: %.2f%% | P: %.2f | R: %.2f | F1: %.2f",
        name,
        metrics["accuracy"] * 100,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


def compare_models():
    """Main comparison pipeline."""
    logger.info("=== Model Comparison Framework ===")

    df_train, df_test = load_data()
    feature_cols = [
        c
        for c in df_test.columns
        if c
        not in [
            "timestamp",
            "symbol",
            "target",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
    ]

    scaler = StandardScaler()
    X_train, y_train = prepare_X_y(df_train, feature_cols, scaler)
    X_test, y_test = prepare_X_y(df_test, feature_cols, scaler)

    results = {}

    # 1. Random Forest
    logger.info("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    results["RandomForest"] = evaluate_model("Random Forest", y_test, y_pred_rf)
    joblib.dump(rf, "models/rf_model_comparison.pkl")

    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("\n[2/3] Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        results["XGBoost"] = evaluate_model("XGBoost", y_test, y_pred_xgb)
        joblib.dump(xgb, "models/xgb_model_comparison.pkl")
    else:
        logger.warning("XGBoost not available — skipping")

    # 3. LSTM
    if TORCH_AVAILABLE:
        logger.info("\n[3/3] Training LSTM...")
        lstm_results = train_lstm(X_train, y_train, X_test, y_test)
        if lstm_results:
            results["LSTM"] = evaluate_model(
                "LSTM", y_test, lstm_results["predictions"]
            )
            torch.save(
                lstm_results["model"].state_dict(), "models/lstm_model_comparison.pt"
            )
    else:
        logger.warning("PyTorch not available — skipping LSTM")

    # Save metrics
    os.makedirs("models", exist_ok=True)
    with open("models/comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n✓ Comparison metrics saved to models/comparison_metrics.json")

    # Plot
    _plot_comparison(results)


def _plot_comparison(results):
    """Bar chart comparing model accuracies."""
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    precs = [results[m]["precision"] for m in models]
    recs = [results[m]["recall"] for m in models]
    f1s = [results[m]["f1"] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, metric, values in zip(
        axes, ["Accuracy", "Precision", "Recall", "F1"], [accs, precs, recs, f1s]
    ):
        bars = ax.bar(
            models, values, color=["#3fb950", "#1f6feb", "#ff9800"][: len(models)]
        )
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}",
                ha="center",
                fontsize=9,
            )

    plt.suptitle("Model Comparison — Out-of-Sample Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("✓ Comparison chart saved: models/model_comparison.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    compare_models()

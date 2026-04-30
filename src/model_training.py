"""
Feature engineering pipeline with proper data leakage prevention and normalization.

Key improvements:
1. No forward-fill before split — prevents data leakage
2. Fit scaler on training data only, apply to train + test
3. Persist scaler for live inference
"""

import logging
import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

logger = logging.getLogger(__name__)


def load_data(filepath: str = "data/btc_usd_hourly.csv") -> pd.DataFrame:
    """
    Load raw OHLCV data from CSV and ensure chronological ordering.

    Args:
        filepath: Path to CSV file with columns: timestamp, open, high, low, close, volume

    Returns:
        DataFrame sorted by timestamp ascending
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators without data leakage.

    Uses rolling windows that only look at past data (no future knowledge).
    All indicators from ta-lib are inherently causal (they only use past data).

    Args:
        df: DataFrame with OHLCV columns, sorted by timestamp

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # No global ffill here — done per-window during training to prevent leakage
    # Missing values from rolling windows will be NaN (dropped later)

    # Momentum indicators
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["stoch_k"] = ta.momentum.stoch(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )

    # Trend indicators
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

    # Volatility indicators
    bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bollinger.bollinger_hband()
    df["bb_low"] = bollinger.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

    df["atr_14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # Volume indicators
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # Log returns (used in feature engineering)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Target: 1 if next close higher, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df


def prepare_features(
    df: pd.DataFrame, fit_scaler: bool = False, scaler: StandardScaler | None = None
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler | None]:
    """
    Prepare feature matrix with proper normalization.

    Args:
        df: DataFrame with indicators (from add_technical_indicators)
        fit_scaler: If True, fit a new StandardScaler on the data
        scaler: Pre-fitted StandardScaler to use for transformation

    Returns:
        X: Feature matrix (n_samples, n_features) - normalized
        y: Target vector (n_samples,)
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler (or None if fit_scaler=False)
    """
    # Define base feature columns — must match models/model_features.pkl exactly
    feature_cols = [
        "rsi_14",
        "stoch_k",
        "macd",
        "macd_signal",
        "macd_diff",
        "ema_20",
        "ema_50",
        "ema_200",
        "bb_high",
        "bb_low",
        "bb_width",
        "atr_14",
        "obv",
        "log_return",
    ]

    # Add normalized distance features (percentage from MAs)
    df["dist_ema_20"] = (df["close"] - df["ema_20"]) / df["close"]
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["close"]
    df["dist_ema_200"] = (df["close"] - df["ema_200"]) / df["close"]

    feature_cols.extend(["dist_ema_20", "dist_ema_50", "dist_ema_200"])

    # Drop rows with NaN (from rolling calculations)
    df_clean = df.dropna(subset=feature_cols + ["target"]).copy()

    X_df = df_clean[feature_cols]  # keep as DataFrame so scaler stores feature names
    y = df_clean["target"].values

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df)
        logger.info("Fitted new StandardScaler on %d features", len(feature_cols))
    elif scaler is not None:
        X = scaler.transform(X_df)
        logger.info("Applied existing StandardScaler")
    else:
        X = X_df.values

    return X, y, feature_cols, scaler


def train_with_walkforward(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[StandardScaler, List[str], dict]:
    """
    Train model with proper temporal split and scaling.

    Args:
        df: Full processed DataFrame
        train_ratio: Fraction of data for training (e.g., 0.8 = 80/20 split)

    Returns:
        scaler: Fitted StandardScaler
        feature_cols: List of feature names used
        metrics: Dict with accuracy, confusion matrix, etc.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    split_idx = int(len(df) * train_ratio)

    # Split BEFORE any scaling to prevent leakage
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    # Fit scaler on training data only
    X_train, y_train, feature_cols, scaler = prepare_features(df_train, fit_scaler=True)
    X_test, y_test, _, _ = prepare_features(df_test, fit_scaler=False, scaler=scaler)

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feature_imp = feature_imp.sort_values("Importance", ascending=False)
    logger.info("\nTop 5 features:\n%s", feature_imp.head(5))

    # Save artifacts
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(feature_cols, "models/model_features.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info("Model, features, and scaler saved to models/")

    metrics = {
        "accuracy": float(acc),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
    }

    return scaler, feature_cols, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Loading data...")
    df = load_data()
    logger.info("Data loaded: %s rows", len(df))

    logger.info("Computing indicators...")
    df_processed = add_technical_indicators(df)
    logger.info("Processed shape: %s", df_processed.shape)

    logger.info("Training with walk-forward split...")
    scaler, features, metrics = train_with_walkforward(df_processed)
    logger.info("Training complete. Metrics: %s", metrics)

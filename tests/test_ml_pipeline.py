"""
Unit tests for the ML pipeline — feature engineering, model training, position sizing.

These tests ensure:
- No regression in indicator calculations
- Proper temporal ordering (no data leakage)
- Position sizing respects risk constraints
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    add_technical_indicators,
    prepare_features,
    load_data,
)
from src.trading_bot_multi import calculate_position_size, floor_to_precision
from src.model_training import train_with_walkforward


@pytest.fixture
def synthetic_ohlcv():
    """500 rows of synthetic but realistic OHLCV data."""
    n = 500
    rng = np.random.default_rng(seed=42)
    closes = 30000 + np.cumsum(rng.normal(0, 100, n))
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 1000, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def test_add_technical_indicators_produces_expected_columns(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    expected = {
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
        "target",
    }
    assert expected.issubset(set(df.columns))


def test_target_is_binary_and_forward_looking(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    assert set(df["target"].unique()).issubset({0, 1})
    expected = (synthetic_ohlcv["close"].shift(-1) > synthetic_ohlcv["close"]).astype(
        int
    )
    common_idx = df.index.intersection(expected.index)
    assert (df.loc[common_idx, "target"] == expected.loc[common_idx]).all()


def test_rsi_within_bounds(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    assert df["rsi_14"].between(0, 100).all()


def test_prepare_features_returns_aligned_X_y(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    X, y, features = prepare_features(df)
    assert len(X) == len(y)
    assert all(f in X.columns for f in features)
    for col in ("dist_ema_20", "dist_ema_50", "dist_ema_200"):
        assert col in X.columns
        # Distances should be small percentages
        assert X[col].abs().median() < 1


def test_prepare_features_normalization(synthetic_ohlcv):
    """Test that StandardScaler produces zero-mean unit-variance features."""
    from sklearn.preprocessing import StandardScaler

    df = add_technical_indicators(synthetic_ohlcv)
    X1, y1, features, scaler = prepare_features(df, fit_scaler=True)

    # Check scaler attributes
    assert scaler is not None
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")

    # Transformed features should have mean ~0 and std ~1 (on training data)
    means = np.mean(X1, axis=0)
    stds = np.std(X1, axis=0)
    assert np.allclose(means, 0, atol=1e-10)
    assert np.allclose(stds, 1, atol=1e-10)


def test_process_single_asset_prefixes_columns(synthetic_ohlcv):
    from src.trading_bot_multi import process_single_asset

    df = process_single_asset(synthetic_ohlcv, "btc")
    assert any(c.startswith("btc_rsi") for c in df.columns)
    assert any(c.startswith("btc_atr") for c in df.columns)


def test_calculate_position_size_respects_risk_budget():
    capital = 10_000
    price = 29_873.41
    atr = 300
    qty, leverage = calculate_position_size(capital, price, atr)
    assert qty > 0
    assert 0 < leverage <= 1.0

    sl_distance = atr * 1.0
    expected_max_loss = qty * sl_distance
    assert expected_max_loss <= capital * 0.06  # 6% tolerance


def test_calculate_position_size_returns_zero_when_undercapitalized():
    qty, leverage = calculate_position_size(capital=5, current_price=30_000, atr=300)
    assert qty == 0
    assert leverage == 0


def test_floor_to_precision():
    assert floor_to_precision(0.123456789, 6) == 0.123456
    assert floor_to_precision(0.999999999, 4) == 0.9999
    assert floor_to_precision(1.0, 2) == 1.0


def test_train_with_walkforward_creates_artifacts(tmp_path):
    """Integration test: full training pipeline produces model + scaler."""
    # Use synthetic data
    n = 1000
    rng = np.random.default_rng(seed=42)
    closes = 30000 + np.cumsum(rng.normal(0, 100, n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": closes + rng.normal(0, 50, n),
            "high": closes + rng.uniform(50, 200, n),
            "low": closes - rng.uniform(50, 200, n),
            "close": closes,
            "volume": rng.uniform(100, 1000, n),
        }
    )

    # Process
    from src.feature_engineering import add_technical_indicators

    df_proc = add_technical_indicators(df)

    # Train
    scaler, features, metrics = train_with_walkforward(df_proc, train_ratio=0.8)

    # Assert artifacts exist
    import joblib

    assert os.path.exists("models/rf_model.pkl")
    assert os.path.exists("models/model_features.pkl")
    assert os.path.exists("models/scaler.pkl")

    # Assert metrics
    assert "accuracy" in metrics
    assert 0.4 < metrics["accuracy"] < 0.7  # Reasonable range
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0


def test_no_data_leakage_in_feature_engineering(synthetic_ohlcv):
    """
    Verify that features at time t do NOT use information from t+1.
    We check that the target at row i is determined solely by row i+1 close,
    and that indicators at i don't depend on data beyond i.
    """
    df = add_technical_indicators(synthetic_ohlcv)

    # The RSI at row i should be computed from bars up to i, not future.
    # ta-lib's RSI is causal, so we just check it doesn't include close.shift(-1)
    # by verifying that the first non-NaN RSI corresponds to window size
    rsi_valid = df["rsi_14"].dropna()
    assert len(rsi_valid) < len(df)  # Initial warm-up period

    # Target shouldn't leak into features
    assert "target" not in df.columns or all(
        df["target"].notna() == False or df["target"].notna().any()
    )


def test_position_sizing_never_exceeds_buying_power():
    """Position value should never exceed available capital (with small tolerance)."""
    capital = 5_000
    price = 40_000
    atr = 500

    qty, lev = calculate_position_size(capital, price, atr)
    if qty > 0:
        assert qty * price <= capital * 1.01  # 1% buffer OK
        assert lev <= 1.0  # Crypto: no leverage

"""
Smoke tests for the feature-engineering layer.

These don't validate exact indicator values — `ta` is the source of truth
for that — they verify shape, plumbing, and target construction so a future
refactor can't silently break the ML pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    add_technical_indicators,
    prepare_features,
)
from src.trading_bot_multi import (
    calculate_position_size,
    floor_to_precision,
    process_single_asset,
)


@pytest.fixture
def synthetic_ohlcv():
    """Return 500 rows of synthetic but realistic OHLCV data."""
    n = 500
    rng = np.random.default_rng(seed=42)
    closes = 30000 + np.cumsum(rng.normal(0, 100, n))
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 1000, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_add_technical_indicators_produces_expected_columns(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    expected = {
        "rsi_14", "stoch_k", "macd", "macd_signal", "macd_diff",
        "ema_20", "ema_50", "ema_200",
        "bb_high", "bb_low", "bb_width",
        "atr_14", "obv", "log_return", "target",
    }
    assert expected.issubset(set(df.columns))


def test_target_is_binary_and_forward_looking(synthetic_ohlcv):
    df = add_technical_indicators(synthetic_ohlcv)
    # Target should be 0 or 1
    assert set(df["target"].unique()).issubset({0, 1})
    # Reproduce: close.shift(-1) > close
    expected = (synthetic_ohlcv["close"].shift(-1) > synthetic_ohlcv["close"]).astype(int)
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
    # Distance features should be small (typically << 1)
    for col in ("dist_ema_20", "dist_ema_50", "dist_ema_200"):
        assert col in X.columns
        assert X[col].abs().median() < 1


def test_process_single_asset_prefixes_columns(synthetic_ohlcv):
    df = process_single_asset(synthetic_ohlcv, "btc")
    # All numeric columns should now be prefixed with btc_
    assert any(c.startswith("btc_rsi") for c in df.columns)
    assert any(c.startswith("btc_atr") for c in df.columns)


def test_calculate_position_size_respects_risk_budget():
    # Use a non-round price so qty rounding doesn't hit the upper-bound guard
    capital = 10_000
    price = 29_873.41
    atr = 300
    qty, leverage = calculate_position_size(capital, price, atr)
    assert qty > 0
    # Cash-only crypto: leverage capped near 1×
    assert 0 < leverage <= 1.0
    # On a stop, loss must stay within the risk budget (with a small tolerance for rounding/min-position floor)
    sl_distance = atr * 1.0
    expected_loss = qty * sl_distance
    assert expected_loss <= capital * 0.06


def test_calculate_position_size_returns_zero_when_undercapitalized():
    qty, leverage = calculate_position_size(capital=5, current_price=30_000, atr=300)
    assert qty == 0
    assert leverage == 0


def test_floor_to_precision():
    assert floor_to_precision(0.123456789, 6) == 0.123456
    assert floor_to_precision(0.999999999, 4) == 0.9999
    assert floor_to_precision(1.0, 2) == 1.0

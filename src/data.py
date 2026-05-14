"""
Data ingestion and feature engineering for BTC/USD 4H bars.

Consolidates: data_ingestion + feature_engineering + model_training indicators.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import ta

logger = logging.getLogger(__name__)

# 17 features used by the model
FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema_20",
    "ema_50",
    "bb_high",
    "bb_low",
    "bb_width",
    "atr_14",
    "obv",
    "log_return",
    "dist_ema_20",
    "dist_ema_50",
    "candle_body_ratio",
    "volume_sma_ratio",
    "ema20_slope",
]


def fetch_bars(
    symbol: str,
    lookback_days: int = 90,
    timeframe_hours: int = 4,
    api_key: str = "",
    secret_key: str = "",
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca for the given symbol and timeframe.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame(timeframe_hours, TimeFrameUnit.Hour)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=tf, start=start, end=now)

    for attempt in range(max_retries):
        try:
            client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
            bars = client.get_crypto_bars(req)
            if not bars or bars.df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = bars.df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].copy()
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info("Fetched %d bars for %s (%dH)", len(df), symbol, timeframe_hours)
            return df
        except Exception as e:
            wait = (attempt + 1) * 3
            if attempt < max_retries - 1:
                logger.warning("Attempt %d/%d failed for %s: %s. Retry in %ds", attempt + 1, max_retries, symbol, e, wait)
                time.sleep(wait)
            else:
                raise


def load_csv(filepath: str, resample_to_4h: bool = True) -> pd.DataFrame:
    """
    Load OHLCV from a CSV file.
    If resample_to_4h is True and data looks hourly, resample it to 4H bars.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if resample_to_4h:
        df = df.set_index("timestamp")
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()
        logger.info("Resampled to 4H: %d bars", len(df))

    return df


def _tp_sl_target(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    max_hold: int,
) -> np.ndarray:
    """
    Label each bar 1 if TP is hit before SL within max_hold forward candles, else 0.
    Uses the high/low of each forward candle to detect which level is touched first.
    """
    n = len(close)
    target = np.zeros(n, dtype=int)
    for i in range(n - 1):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        sl = close[i] - sl_mult * atr_i
        tp = close[i] + tp_mult * atr_i
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if low[j] <= sl:
                break          # SL hit first → target stays 0
            if high[j] >= tp:
                target[i] = 1  # TP hit first
                break
    return target


def add_indicators(
    df: pd.DataFrame,
    sl_atr_mult: float = 1.0,
    tp_atr_mult: float = 2.0,
    max_hold_candles: int = 12,
) -> pd.DataFrame:
    """
    Compute all 17 technical features + target variable.

    Target: 1 if TP (tp_atr_mult × ATR above entry) is hit before SL
    (sl_atr_mult × ATR below entry) within max_hold_candles forward bars.
    Directly aligned with the bot's actual trade resolution logic.
    """
    df = df.copy()

    # Momentum
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # Trend
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)

    # Volatility
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Volume
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # Price action
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Normalized distance from moving averages
    df["dist_ema_20"] = (df["close"] - df["ema_20"]) / df["close"]
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["close"]

    # New features
    candle_range = df["high"] - df["low"]
    df["candle_body_ratio"] = (df["close"] - df["open"]) / candle_range.replace(0, np.nan)
    df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["ema20_slope"] = np.sign(df["ema_20"] - df["ema_20"].shift(3))

    # Target: 1 if TP hit before SL within max_hold_candles forward bars
    df["target"] = _tp_sl_target(
        df["close"].values, df["high"].values, df["low"].values,
        df["atr_14"].values, sl_atr_mult, tp_atr_mult, max_hold_candles,
    )

    return df


def fetch_news(symbol: str = "BTC", api_key: str = "", secret_key: str = "", max_retries: int = 3) -> list[str]:
    """Fetch latest news headlines from Alpaca News API."""
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    params = {"symbols": symbol, "limit": 5, "include_content": False}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return [n["headline"] for n in resp.json().get("news", [])]
            return []
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return []

"""
Main trading bot logic — multi-asset (BTC + ETH correlation) with dynamic risk.

This module implements the core execution loop:
1. Fetch market data (BTC + ETH)
2. Calculate technical indicators
3. Run ML prediction (Random Forest)
4. Sentiment analysis filter (Gemini LLM)
5. Position sizing (volatility-adjusted Kelly)
6. Order execution + SL/TP management
"""

import os
import sys
import time
import math
import logging
import signal
import json
from datetime import datetime, timezone
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.getcwd())

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderStatus,
    QueryOrderStatus,
    OrderType,
)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.sentiment_analysis import analyze_sentiment
from src.config import get_settings
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import joblib
import numpy as np
import pandas as pd

# Load environment
load_dotenv()
settings = get_settings()

# Configure logging
structlog_available = False
try:
    import structlog

    structlog_available = True
except ImportError:
    pass

if structlog_available:
    logger = structlog.get_logger()
else:
    logger = logging.getLogger(__name__)

# Global flags for graceful shutdown
_shutdown_requested = False


def _env_float(name: str, default: float) -> float:
    """Helper to read float from env with fallback."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


# --- Bot Configuration (from settings) ---
SYMBOL_BTC = settings.symbol_btc
SYMBOL_ETH = settings.symbol_eth
TIMEFRAME = TimeFrame.Hour if settings.timeframe == "1h" else TimeFrame.Hour

# Risk parameters
SL_ATR_MULT = settings.sl_atr_mult
TP_ATR_MULT = settings.tp_atr_mult
SL_LIMIT_SLIPPAGE = settings.sl_limit_slippage
MAX_RISK_PER_TRADE = settings.max_risk_per_trade
DAILY_LOSS_LIMIT_PCT = settings.daily_loss_limit_pct
MAX_TOTAL_EXPOSURE = settings.max_total_exposure
MAX_CONCURRENT_TRADES = settings.max_concurrent_trades

# Confidence thresholds
CONFIDENCE_THRESHOLD = settings.confidence_threshold
SENTIMENT_FLOOR = settings.sentiment_floor

# Execution parameters
FILL_POLL_ATTEMPTS = settings.fill_poll_attempts
FILL_POLL_INTERVAL_S = settings.fill_poll_interval_s
TP_WATCHDOG_INTERVAL_S = settings.tp_watchdog_interval_s

# Model paths
MODEL_PATH = settings.model_path
FEATURES_PATH = settings.features_path
SCALER_PATH = settings.scaler_path

# Drift detection
DRIFT_CHECK_WINDOW = settings.drift_check_window
DRIFT_ACCURACY_THRESHOLD = settings.drift_accuracy_threshold

# Load model, features, scaler at module level (cached)
_model = None
_feature_cols = None
_scaler = None


def _load_ml_artifacts():
    """Load model, feature list, and scaler lazily."""
    global _model, _feature_cols, _scaler
    try:
        _model = joblib.load(MODEL_PATH)
        _feature_cols = joblib.load(FEATURES_PATH)
        logger.info("Model loaded: %s, features: %s", MODEL_PATH, len(_feature_cols))
    except Exception as e:
        logger.error("Failed to load model/features: %s", e)
        raise

    try:
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded: %s", SCALER_PATH)
    except Exception as e:
        logger.warning("Scaler not found, will use unscaled features: %s", e)
        _scaler = None


def floor_to_precision(value: float, precision: int) -> float:
    """Round down to specified decimal places."""
    factor = 10**precision
    return math.floor(value * factor) / factor


def handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    logger.info("Shutdown signal received (%s), closing positions...", signum)
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


def is_shutdown_requested() -> bool:
    return _shutdown_requested


def get_latest_data(
    symbol: str, lookback_days: int = 30, max_retries: int = 5
) -> pd.DataFrame:
    """
    Fetch recent OHLCV bars from Alpaca.

    Args:
        symbol: Trading pair (e.g., "BTC/USD")
        lookback_days: How many days of history to fetch
        max_retries: Number of retry attempts on failure

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    now = datetime.now(pytz.UTC)
    start = now - timedelta(days=lookback_days)
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol], timeframe=TIMEFRAME, start=start, end=now
    )

    for attempt in range(max_retries):
        try:
            client = CryptoHistoricalDataClient(
                api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
            )
            bars = client.get_crypto_bars(req)

            if not bars or bars.df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = bars.df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].copy()

            return df
        except Exception as e:
            wait = (attempt + 1) * 3
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    symbol,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed for %s.", max_retries, symbol)
                raise


def get_latest_news(symbol: str = "BTC", max_retries: int = 3) -> List[str]:
    """Fetch latest news headlines for sentiment analysis."""
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    params = {"symbols": symbol, "limit": 5, "include_content": False}

    for attempt in range(max_retries):
        try:
            import requests

            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return [n["headline"] for n in response.json().get("news", [])]
            return []
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return []


def process_single_asset(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """
    Compute indicators for a single asset.

    Args:
        df: OHLCV DataFrame for one asset
        asset_name: Prefix for column names (e.g., 'btc')

    Returns:
        DataFrame with prefixed indicator columns
    """
    df = df.set_index("timestamp").sort_index()
    df["returns"] = df["close"].pct_change()

    # Momentum
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Trend
    ema20 = EMAIndicator(close=df["close"], window=20)
    df["ema20"] = ema20.ema_indicator()

    # Volatility
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()

    df = df.add_prefix(f"{asset_name}_")
    return df


def calculate_position_size(
    capital: float, current_price: float, atr: float, open_positions_val: float = 0.0
) -> tuple[float, float]:
    """
    Calculate position size using volatility-adjusted Kelly criterion.

    Args:
        capital: Available buying power (USD)
        current_price: Current asset price
        atr: Average True Range (volatility measure)
        open_positions_val: Total value of current positions (for exposure limit)

    Returns:
        (quantity, leverage) — quantity rounded down to 6 decimals
    """
    if capital < 11.0:  # Minimum order size guard
        return 0.0, 0.0

    # Stop loss distance
    dist_stop = atr * SL_ATR_MULT
    if dist_stop <= 0 or current_price <= 0:
        return 0.0, 0.0

    stop_pct = dist_stop / current_price

    # Base position size from Kelly
    risk_amount = capital * MAX_RISK_PER_TRADE
    ideal_position_value = risk_amount / stop_pct

    # Apply exposure limits
    max_pos = capital * MAX_TOTAL_EXPOSURE
    if open_positions_val > 0:
        max_pos = min(
            max_pos, capital * (MAX_TOTAL_EXPOSURE - (open_positions_val / capital))
        )

    min_pos = capital * 0.05  # Minimum 5% position
    final_pos_value = min(max(ideal_position_value, min_pos), max_pos)
    final_pos_value = max(final_pos_value, 11.0)  # Absolute floor
    final_pos_value = min(final_pos_value, max_pos)

    qty = floor_to_precision(final_pos_value / current_price, 6)

    # Enforce minimum order value
    if qty * current_price < 10.1:
        qty = floor_to_precision(10.5 / current_price, 6)

    if qty * current_price > max_pos:
        return 0.0, 0.0

    leverage = (qty * current_price) / capital if capital > 0 else 0.0
    return qty, leverage


def wait_for_fill(
    trading_client: TradingClient,
    order_id: str,
    attempts: int = FILL_POLL_ATTEMPTS,
    interval: float = FILL_POLL_INTERVAL_S,
):
    """Poll order until filled or terminal status."""
    for _ in range(attempts):
        try:
            order = trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            if order.status in (
                OrderStatus.CANCELED,
                OrderStatus.EXPIRED,
                OrderStatus.REJECTED,
            ):
                return order
            time.sleep(interval)
        except Exception as e:
            logger.warning("Error checking order %s: %s", order_id, e)
    return None


def cancel_open_orders_for_symbol(trading_client: TradingClient, symbol: str) -> int:
    """Cancel all open orders for a symbol. Returns count cancelled."""
    try:
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
    except Exception as e:
        logger.warning("Could not list open orders: %s", e)
        return 0

    target = symbol.replace("/", "").upper()
    cancelled = 0
    for o in open_orders:
        if o.symbol.replace("/", "").upper() != target:
            continue
        try:
            trading_client.cancel_order_by_id(o.id)
            cancelled += 1
        except Exception as e:
            logger.warning("Failed to cancel order %s: %s", o.id, e)
    return cancelled


def check_drift_and_alert(run_data: dict) -> dict:
    """
    Check for model drift by tracking recent prediction accuracy.

    Adds 'drift_warning' and 'drift_metrics' to run_data.
    This is a lightweight check that doesn't require ground truth yet.
    """
    from src.run_store import get_last_n
    from src.database import get_open_trades

    recent_runs = get_last_n(DRIFT_CHECK_WINDOW)
    if not recent_runs:
        return run_data

    # Track confidence trends
    confidences = [r.get("confidence", 0) for r in recent_runs if r.get("confidence")]
    avg_confidence = np.mean(confidences) if confidences else 0.5

    # Check if confidence has dropped significantly
    if avg_confidence < DRIFT_ACCURACY_THRESHOLD:
        logger.warning("Low average confidence detected: %.2f", avg_confidence)
        run_data["drift_warning"] = True
        run_data["drift_metrics"] = {
            "avg_confidence": float(avg_confidence),
            "recent_runs": len(confidences),
        }

    # Check error rate
    errors = sum(1 for r in recent_runs if r.get("error"))
    error_rate = errors / len(recent_runs) if recent_runs else 0
    if error_rate > 0.2:
        logger.warning("High error rate: %.1f%%", error_rate * 100)
        run_data["drift_warning"] = True
        run_data["drift_metrics"]["error_rate"] = float(error_rate)

    return run_data


def check_circuit_breakers() -> dict:
    """
    Check daily loss limits and other safety circuits.

    Returns dict with:
        - halt: bool — if trading should halt
        - reason: str — why halted (or None)
    """
    from src.database import get_todays_statistics

    stats = get_todays_statistics()
    daily_pnl = stats.get("pnl", 0.0)

    # Get starting capital (approximate from config or env)
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "10000"))
    daily_loss_limit = initial_capital * DAILY_LOSS_LIMIT_PCT

    if daily_pnl < -daily_loss_limit:
        logger.warning(
            "Daily loss limit hit: $%.2f < $-%.2f", daily_pnl, daily_loss_limit
        )
        return {"halt": True, "reason": f"Daily loss limit: ${daily_pnl:.2f}"}

    return {"halt": False, "reason": None}


def trade_logic_multi() -> dict:
    """
    Main bot execution function.

    Returns:
        dict with pipeline data for dashboard and DB storage.
    """
    global _model, _feature_cols, _scaler

    # Lazy-load ML artifacts
    if _model is None:
        try:
            _load_ml_artifacts()
        except Exception as e:
            return {
                "timestamp": datetime.now(pytz.UTC).isoformat(),
                "steps": [],
                "error": f"Failed to load model: {e}",
                "action": "ERROR",
            }

    # Initialize result
    result = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "steps": [],
        "error": None,
        "prediction": 0,
        "prediction_label": "FLAT",
        "confidence": 0.0,
        "sentiment_score": 0.0,
        "action": "WAITING",
        "position_qty": 0.0,
        "leverage": 0.0,
        "drift_warning": False,
        "drift_metrics": {},
        "circuit_breaker": None,
    }

    logger.info("--- Multi-asset bot cycle (BTC+ETH): %s ---", datetime.now())

    # --- Circuit Breaker Check ---
    t0 = time.time()
    cb = check_circuit_breakers()
    result["steps"].append(
        {
            "name": "Circuit Breaker",
            "status": "ok" if not cb["halt"] else "halted",
            "duration_ms": int((time.time() - t0) * 1000),
        }
    )
    if cb["halt"]:
        result["error"] = f"Trading halted: {cb['reason']}"
        result["action"] = "CIRCUIT_BREAKER"
        result["circuit_breaker"] = cb
        return result

    # --- Step 1: Connect to Alpaca ---
    t0 = time.time()
    try:
        trading_client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper=True
        )
        account = trading_client.get_account()
        capital = float(account.buying_power)
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        pnl = equity - last_equity
        pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0

        result["equity"] = equity
        result["buying_power"] = capital
        result["pnl_today"] = pnl
        result["pnl_today_pct"] = pnl_pct
        result["steps"].append(
            {
                "name": "Connect Alpaca",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "Connect Alpaca",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        result["error"] = f"Alpaca connection error: {e}"
        logger.error("Failed to connect to Alpaca: %s", e)
        return result

    # --- Step 2: Fetch Market Data ---
    t0 = time.time()
    logger.info("Downloading BTC and ETH bars...")
    try:
        df_btc = get_latest_data(SYMBOL_BTC)
        df_eth = get_latest_data(SYMBOL_ETH)
        last_close_btc = df_btc.iloc[-1]["close"]
        result["btc_close"] = float(last_close_btc)

        # Store last 500 candles for charting
        chart_df = df_btc.tail(500).copy()
        if "timestamp" in chart_df.columns:
            chart_df["timestamp"] = chart_df["timestamp"].astype(str)
        result["ohlcv_data"] = chart_df[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].to_dict(orient="records")

        result["steps"].append(
            {
                "name": "Fetch Market Data",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "Fetch Market Data",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        result["error"] = f"Data fetch error: {e}"
        return result

    # --- Step 3: Calculate Features ---
    t0 = time.time()
    logger.info("Computing features...")
    try:
        # Process BTC data only (single-asset version)
        df_full = process_single_asset(df_btc, "btc")
        df_full.columns = [c.replace("btc_", "") for c in df_full.columns]

        current_state = df_full.iloc[[-1]].copy()
        current_atr = float(current_state["atr"].values[0])

        # Build feature vector in same order as training
        feature_values = []
        for col in _feature_cols:
            if col in current_state.columns:
                feature_values.append(float(current_state[col].values[0]))
            else:
                logger.warning("Feature %s missing, defaulting to 0", col)
                feature_values.append(0.0)

        X_latest = np.array(feature_values).reshape(1, -1)

        # Apply scaler if available
        if _scaler is not None:
            X_latest = _scaler.transform(X_latest)

        # Extract indicator values for dashboard
        last_row = df_full.iloc[-1]
        result["indicators"] = {
            "rsi": float(last_row.get("rsi", 0)),
            "macd": float(last_row.get("macd", 0)),
            "macd_signal": float(last_row.get("macd_signal", 0)),
            "ema20": float(last_row.get("ema20", 0)),
            "bb_width": float(last_row.get("bb_width", 0)),
            "bb_high": float(last_row.get("bb_high", 0)),
            "bb_low": float(last_row.get("bb_low", 0)),
            "atr": current_atr,
        }

        # Store chart data
        chart_indicators = df_full.tail(500).copy()
        chart_indicators.index = chart_indicators.index.astype(str)
        result["chart_indicators"] = {
            "ema20": chart_indicators["ema20"].tolist(),
            "bb_high": chart_indicators["bb_high"].tolist(),
            "bb_low": chart_indicators["bb_low"].tolist(),
            "rsi": chart_indicators["rsi"].tolist(),
            "macd": chart_indicators["macd"].tolist(),
            "macd_signal": chart_indicators["macd_signal"].tolist(),
            "volume": chart_indicators["volume"].tolist()
            if "volume" in chart_indicators.columns
            else [],
            "timestamps": chart_indicators.index.tolist(),
        }

        result["steps"].append(
            {
                "name": "Calculate Features",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "Calculate Features",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        result["error"] = f"Feature calculation error: {e}"
        logger.error("Feature calculation failed: %s", e)
        return result

    # --- Step 4: Sentiment Analysis ---
    t0 = time.time()
    logger.info("Running sentiment analysis...")
    try:
        if settings.sentiment_enabled:
            headlines = get_latest_news("BTC")
            sentiment_score = analyze_sentiment(headlines) if headlines else 0.0
            result["headlines"] = headlines
        else:
            sentiment_score = 0.0
            result["headlines"] = []

        result["sentiment_score"] = float(sentiment_score)
        result["steps"].append(
            {
                "name": "Sentiment Analysis",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "Sentiment Analysis",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        sentiment_score = 0.0
        result["headlines"] = []
        result["sentiment_score"] = 0.0
        logger.warning("Sentiment failed: %s", e)

    # --- Step 5: ML Prediction ---
    t0 = time.time()
    logger.info("Running ML prediction...")
    try:
        probabilities = _model.predict_proba(X_latest)[0]
        prediction = _model.predict(X_latest)[0]
        confidence = probabilities[prediction]

        result["prediction"] = int(prediction)
        result["confidence"] = float(confidence)
        result["prediction_label"] = "LONG" if prediction == 1 else "FLAT"
        result["probabilities"] = {
            "down": float(probabilities[0]),
            "up": float(probabilities[1]),
        }
        result["steps"].append(
            {
                "name": "ML Prediction",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        logger.info(
            "Prediction: %s | confidence: %.2f%%",
            "UP" if prediction == 1 else "DOWN",
            confidence * 100,
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "ML Prediction",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        result["error"] = f"Prediction error: {e}"
        return result

    # --- Step 6: Check Existing Positions ---
    t0 = time.time()
    open_positions_val = 0.0
    btc_pos = None
    try:
        positions = trading_client.get_all_positions()
        btc_pos = next((p for p in positions if p.symbol == "BTC/USD"), None)
        for p in positions:
            open_positions_val += float(p.market_value)
        result["steps"].append(
            {
                "name": "Check Positions",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
    except Exception as e:
        result["steps"].append(
            {
                "name": "Check Positions",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )
        logger.warning("Could not check positions: %s", e)

    # --- Step 7: Execute Orders ---
    t0 = time.time()

    # Entry conditions
    signal_ok = (
        prediction == 1
        and confidence > CONFIDENCE_THRESHOLD
        and sentiment_score >= SENTIMENT_FLOOR
    )

    # Exposure limit check
    exposure_ok = (
        (open_positions_val / capital) < MAX_TOTAL_EXPOSURE if capital > 0 else False
    )
    concurrent_ok = True  # TODO: track concurrent trades from DB

    if signal_ok and exposure_ok and concurrent_ok:
        logger.info("BUY signal validated.")
        try:
            if btc_pos:
                # Already positioned — check for TP hit
                entry = float(btc_pos.avg_entry_price)
                tp_target = entry + (current_atr * TP_ATR_MULT)

                current_sl = None
                try:
                    open_orders = trading_client.get_orders(
                        GetOrdersRequest(
                            status=QueryOrderStatus.OPEN, symbols=[SYMBOL_BTC]
                        )
                    )
                    for o in open_orders:
                        if o.side == OrderSide.SELL and o.order_type in [
                            OrderType.STOP_LIMIT,
                            OrderType.STOP,
                        ]:
                            current_sl = float(o.stop_price) if o.stop_price else None
                except Exception as e:
                    logger.warning("Could not check open orders: %s", e)

                # Attach missing SL
                if not current_sl:
                    logger.warning(
                        "Active position with no SL. Attaching recovery SL..."
                    )
                    try:
                        entry = float(btc_pos.avg_entry_price)
                        rec_stop_price = round(entry - (current_atr * SL_ATR_MULT), 2)
                        rec_limit_price = round(
                            rec_stop_price * (1 - SL_LIMIT_SLIPPAGE), 2
                        )
                        rec_qty = floor_to_precision(float(btc_pos.qty) * 0.9999, 6)

                        sl_req = StopLimitOrderRequest(
                            symbol=SYMBOL_BTC,
                            qty=rec_qty,
                            side=OrderSide.SELL,
                            stop_price=rec_stop_price,
                            limit_price=rec_limit_price,
                            time_in_force=TimeInForce.GTC,
                        )
                        trading_client.submit_order(sl_req)
                        current_sl = rec_stop_price
                        logger.info("Recovery SL attached @ $%.2f", rec_stop_price)
                    except Exception as e:
                        logger.error("Recovery SL failed: %s", e)

                # Check TP
                current_price = float(btc_pos.current_price)
                if current_price >= tp_target:
                    cancel_open_orders_for_symbol(trading_client, "BTC/USD")
                    trading_client.close_position("BTC/USD")
                    result["action"] = "TAKE_PROFIT_HIT"
                    result["take_profit"] = float(tp_target)
                    result["stop_loss"] = current_sl
                    result["steps"].append(
                        {
                            "name": "Execute Order",
                            "status": "ok",
                            "duration_ms": int((time.time() - t0) * 1000),
                        }
                    )
                    return result

                result["action"] = "ALREADY_POSITIONED"
                result["stop_loss"] = current_sl
                result["take_profit"] = float(tp_target)
                result["position_qty"] = float(btc_pos.qty)
                result["leverage"] = (
                    (float(btc_pos.qty) * float(btc_pos.current_price)) / capital
                    if capital > 0
                    else 0
                )
                result["steps"].append(
                    {
                        "name": "Execute Order",
                        "status": "ok",
                        "duration_ms": int((time.time() - t0) * 1000),
                    }
                )
                return result

            # No position — clear orphan orders
            orphan = cancel_open_orders_for_symbol(trading_client, "BTC/USD")
            if orphan:
                logger.info("Cleared %d orphan order(s) before buy.", orphan)

            # Calculate position size
            qty, leverage = calculate_position_size(
                capital, last_close_btc, current_atr, open_positions_val
            )
            result["position_qty"] = float(qty)
            result["leverage"] = float(leverage)

            if qty <= 0:
                result["action"] = "INSUFFICIENT_FUNDS"
                result["error"] = "Not enough buying power for minimum order."
                result["steps"].append(
                    {
                        "name": "Execute Order",
                        "status": "error",
                        "duration_ms": int((time.time() - t0) * 1000),
                    }
                )
                return result

            logger.info("Submitting BUY: %.6f BTC @ ~$%.2f", qty, last_close_btc)

            buy_req = MarketOrderRequest(
                symbol=SYMBOL_BTC,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            buy_order = trading_client.submit_order(buy_req)
            result["order_id"] = str(buy_order.id)

            filled = wait_for_fill(trading_client, buy_order.id)
            if filled is None or filled.status != OrderStatus.FILLED:
                result["action"] = "BUY_ORDER_SENT"
                result["error"] = (
                    f"BUY not filled in time (status={getattr(filled, 'status', None)})"
                )
                result["steps"].append(
                    {
                        "name": "Execute Order",
                        "status": "error",
                        "duration_ms": int((time.time() - t0) * 1000),
                    }
                )
                return result

            entry_price = (
                float(filled.filled_avg_price)
                if filled.filled_avg_price
                else last_close_btc
            )
            filled_qty = float(filled.filled_qty) if filled.filled_qty else qty

            # Compute SL/TP
            sl_qty = floor_to_precision(filled_qty * 0.9999, 6)
            stop_price = round(entry_price - (current_atr * SL_ATR_MULT), 2)
            take_profit_price = round(entry_price + (current_atr * TP_ATR_MULT), 2)

            result["stop_loss"] = float(stop_price)
            result["take_profit"] = float(take_profit_price)
            result["entry_price"] = float(entry_price)
            result["entry_time"] = (
                filled.filled_at.isoformat()
                if filled.filled_at
                else result["timestamp"]
            )
            result["filled_qty"] = float(filled_qty)
            result["action"] = "BUY_ORDER_SENT"

            logger.info(
                "BUY filled @ $%.2f. Setting SL @ $%.2f", entry_price, stop_price
            )

            # Submit stop-loss
            sl_error = None
            try:
                sl_limit_price = round(stop_price * (1 - SL_LIMIT_SLIPPAGE), 2)
                sl_req = StopLimitOrderRequest(
                    symbol=SYMBOL_BTC,
                    qty=sl_qty,
                    side=OrderSide.SELL,
                    stop_price=stop_price,
                    limit_price=sl_limit_price,
                    time_in_force=TimeInForce.GTC,
                )
                sl_order = trading_client.submit_order(sl_req)
                result["sl_order_id"] = str(sl_order.id)
                logger.info("SL order submitted: %s", sl_order.id)
            except Exception as e:
                sl_error = str(e)
                logger.error("Failed to create SL: %s", e)

            if sl_error:
                result["error"] = f"SL: {sl_error}"
                result["steps"].append(
                    {
                        "name": "Execute Order",
                        "status": "error",
                        "duration_ms": int((time.time() - t0) * 1000),
                    }
                )
            else:
                result["steps"].append(
                    {
                        "name": "Execute Order",
                        "status": "ok",
                        "duration_ms": int((time.time() - t0) * 1000),
                    }
                )

        except Exception as e:
            result["action"] = "ORDER_ERROR"
            result["error"] = str(e)
            result["steps"].append(
                {
                    "name": "Execute Order",
                    "status": "error",
                    "duration_ms": int((time.time() - t0) * 1000),
                }
            )
            logger.error("Order failed: %s", e)

    elif prediction == 0:
        logger.info("FLAT signal — closing position if any...")
        try:
            positions = trading_client.get_all_positions()
            has_position = any(p.symbol == "BTC/USD" for p in positions)

            cancelled = cancel_open_orders_for_symbol(trading_client, "BTC/USD")
            if cancelled:
                logger.info("Cancelled %d open order(s).", cancelled)

            if has_position:
                trading_client.close_position("BTC/USD")
                result["action"] = "CLOSE_POSITION"
                logger.info("Position closed.")
            else:
                result["action"] = "NO_POSITION_TO_CLOSE"

            result["steps"].append(
                {
                    "name": "Execute Order",
                    "status": "ok",
                    "duration_ms": int((time.time() - t0) * 1000),
                }
            )
        except Exception as e:
            result["action"] = "CLOSE_ERROR"
            result["error"] = str(e)
            result["steps"].append(
                {
                    "name": "Execute Order",
                    "status": "error",
                    "duration_ms": int((time.time() - t0) * 1000),
                }
            )
            logger.error("Close failed: %s", e)
    else:
        result["action"] = "NO_SIGNAL"
        result["steps"].append(
            {
                "name": "Execute Order",
                "status": "ok",
                "duration_ms": int((time.time() - t0) * 1000),
            }
        )

    # Add drift detection
    result = check_drift_and_alert(result)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    res = trade_logic_multi()
    print(json.dumps(res, indent=2, default=str))

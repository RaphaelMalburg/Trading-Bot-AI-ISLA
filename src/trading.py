"""
Main trading pipeline — orchestrates data → features → model → sentiment → execution.
Called once per candle close by bot_loop in app.py.
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.config import get_settings
from src.data import FEATURE_COLS, add_indicators, fetch_bars, fetch_news
from src.sentiment import analyze_sentiment
from src.strategy import check_circuit_breaker, position_size, should_enter, sl_tp_prices

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_feature_cols = None


def _load_model():
    global _model, _scaler, _feature_cols
    from src.model import load_artifacts
    _model, _scaler, _feature_cols = load_artifacts()
    logger.info("Model artifacts loaded (%d features)", len(_feature_cols))


def trade_logic() -> dict:
    """
    Execute one full trading cycle.

    Returns a result dict that is stored in DB and displayed on the dashboard.
    """
    global _model, _scaler, _feature_cols

    settings = get_settings()
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_close": 0.0,
        "prediction": 0,
        "prediction_label": "FLAT",
        "confidence": 0.0,
        "sentiment_score": 0.0,
        "action": "WAITING",
        "position_qty": 0.0,
        "leverage": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "order_id": None,
        "entry_price": None,
        "entry_time": None,
        "filled_qty": None,
        "equity": 0.0,
        "buying_power": 0.0,
        "pnl_today": 0.0,
        "pnl_today_pct": 0.0,
        "drift_warning": False,
        "error": None,
        "ohlcv_data": [],
        "indicators": {},
    }

    # ── 0. Load model lazily ────────────────────────────────────────────────
    if _model is None:
        try:
            _load_model()
        except Exception as e:
            result["error"] = f"Model load failed: {e}"
            result["action"] = "ERROR"
            return result

    # ── 1. Connect to Alpaca ────────────────────────────────────────────────
    t0 = time.time()
    try:
        from src.broker import get_account_info, get_client
        client = get_client(settings.alpaca_api_key, settings.alpaca_secret_key)
        account = get_account_info(client)
        result.update(account)
        logger.info("Account: equity=%.2f buying_power=%.2f", account["equity"], account["buying_power"])
    except Exception as e:
        result["error"] = f"Alpaca connect error: {e}"
        result["action"] = "ERROR"
        return result

    # ── 2. Circuit breaker ──────────────────────────────────────────────────
    from src.database import get_todays_statistics
    today_stats = get_todays_statistics()
    daily_pnl = today_stats.get("pnl", 0.0)
    initial_capital = account["equity"] - account["pnl_today"] or account["equity"]
    if check_circuit_breaker(daily_pnl, initial_capital, settings.daily_loss_limit_pct):
        result["action"] = "CIRCUIT_BREAKER"
        result["error"] = f"Daily loss limit hit: ${daily_pnl:.2f}"
        return result

    # ── 3. Fetch market data ─────────────────────────────────────────────────
    try:
        df = fetch_bars(
            symbol=settings.symbol,
            lookback_days=90,
            timeframe_hours=settings.timeframe_hours,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        last_close = float(df.iloc[-1]["close"])
        result["btc_close"] = last_close
    except Exception as e:
        result["error"] = f"Data fetch error: {e}"
        result["action"] = "ERROR"
        return result

    # ── 4. Compute features ──────────────────────────────────────────────────
    try:
        df_feat = add_indicators(
            df.copy(),
            sl_atr_mult=settings.sl_atr_mult,
            tp_atr_mult=settings.tp_atr_mult,
            max_hold_candles=settings.max_hold_candles,
        )
        df_feat_clean = df_feat.dropna(subset=_feature_cols)
        if df_feat_clean.empty:
            raise ValueError("No valid rows after indicator computation")

        last_row = df_feat_clean.iloc[-1]
        current_atr = float(last_row.get("atr_14", 1.0))

        result["indicators"] = {
            "rsi": float(last_row.get("rsi_14", 0)),
            "macd": float(last_row.get("macd", 0)),
            "macd_signal": float(last_row.get("macd_signal", 0)),
            "ema20": float(last_row.get("ema_20", 0)),
            "bb_width": float(last_row.get("bb_width", 0)),
            "bb_high": float(last_row.get("bb_high", 0)),
            "bb_low": float(last_row.get("bb_low", 0)),
            "atr": current_atr,
        }

        # Build chart data with full indicator time series
        chart_df = df_feat.tail(200).copy()
        chart_df["timestamp"] = chart_df["timestamp"].astype(str)
        chart_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in ["rsi_14", "macd", "macd_signal", "macd_diff", "ema_20"]:
            if col in chart_df.columns:
                chart_cols.append(col)
        raw_records = chart_df[chart_cols].to_dict(orient="records")
        result["ohlcv_data"] = [
            {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
            for row in raw_records
        ]

        # Build feature vector for prediction
        X_latest = pd.DataFrame([{col: float(last_row.get(col, 0.0)) for col in _feature_cols}])
        X_scaled = _scaler.transform(X_latest)
    except Exception as e:
        result["error"] = f"Feature error: {e}"
        result["action"] = "ERROR"
        return result

    # ── 5. ML prediction ─────────────────────────────────────────────────────
    try:
        probas = _model.predict_proba(X_scaled)[0]
        prediction = int(_model.predict(X_scaled)[0])
        confidence = float(probas[prediction])
        result["prediction"] = prediction
        result["confidence"] = confidence
        result["prediction_label"] = "LONG" if prediction == 1 else "FLAT"
        logger.info("Prediction: %s | confidence: %.2f%%", result["prediction_label"], confidence * 100)
    except Exception as e:
        result["error"] = f"Prediction error: {e}"
        result["action"] = "ERROR"
        return result

    # ── 6. Drift check ────────────────────────────────────────────────────────
    from src.database import get_recent_runs
    from src.model import check_drift
    recent = get_recent_runs(settings.drift_check_window)
    recent_confidences = [r["confidence"] for r in recent if r.get("confidence")]
    result["drift_warning"] = check_drift(recent_confidences, settings.drift_confidence_floor)

    # ── 7. Sentiment ──────────────────────────────────────────────────────────
    sentiment_score = 0.0
    if settings.sentiment_enabled:
        try:
            headlines = fetch_news("BTC", api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key)
            sentiment_score = analyze_sentiment(headlines, api_key=settings.openrouter_api_key)
        except Exception as e:
            logger.warning("Sentiment failed: %s", e)
    result["sentiment_score"] = sentiment_score

    # ── 8. Execute ────────────────────────────────────────────────────────────
    from src.broker import (
        cancel_orders, close_position, get_open_orders, get_position,
        market_buy, place_disaster_stop, wait_for_fill,
    )
    from src.strategy import _floor

    btc_pos = get_position(client, settings.symbol)
    has_position = btc_pos is not None

    if should_enter(prediction, confidence, sentiment_score, has_position,
                    settings.confidence_threshold, settings.sentiment_floor):
        logger.info("BUY signal — placing order")

        qty, leverage = position_size(
            account["buying_power"], last_close, current_atr,
            settings.sl_atr_mult, settings.max_risk_per_trade, settings.max_total_exposure,
        )
        result["position_qty"] = qty
        result["leverage"] = leverage

        if qty <= 0:
            result["action"] = "INSUFFICIENT_FUNDS"
            return result

        cancel_orders(client, settings.symbol)

        buy_order = market_buy(client, settings.symbol, qty)
        if not buy_order:
            result["action"] = "ORDER_ERROR"
            result["error"] = "Failed to submit BUY order"
            return result

        result["order_id"] = str(buy_order.id)
        filled = wait_for_fill(client, buy_order.id, settings.fill_poll_attempts, settings.fill_poll_interval_s)

        from alpaca.trading.enums import OrderStatus
        if not filled or filled.status != OrderStatus.FILLED:
            result["action"] = "BUY_ORDER_SENT"
            result["error"] = "BUY not confirmed filled"
            return result

        entry_price = float(filled.filled_avg_price) if filled.filled_avg_price else last_close
        filled_qty = float(filled.filled_qty) if filled.filled_qty else qty

        _, _, take_profit_price = sl_tp_prices(
            entry_price, current_atr, settings.sl_atr_mult, settings.tp_atr_mult, settings.sl_limit_slippage
        )
        sl_price = round(entry_price - current_atr * settings.sl_atr_mult, 2)
        disaster_qty = _floor(filled_qty * 0.9999, 6)

        result.update({
            "action": "BUY_ORDER_SENT",
            "entry_price": entry_price,
            "entry_time": filled.filled_at.isoformat() if filled.filled_at else result["timestamp"],
            "filled_qty": filled_qty,
            "stop_loss": sl_price,
            "take_profit": take_profit_price,
            "disaster_order_id": None,
        })

        # Crash safety net only — app-side exit_watchdog handles real SL/TP
        disaster_order = place_disaster_stop(
            client, settings.symbol, disaster_qty, entry_price,
            current_atr, settings.sl_atr_mult,
        )
        if disaster_order:
            result["disaster_order_id"] = str(disaster_order.id)
        else:
            result["error"] = "BUY filled but disaster stop order failed"

    elif not has_position:
        result["action"] = "NO_SIGNAL"

    elif has_position and prediction == 0:
        logger.info("FLAT signal — closing position")
        cancel_orders(client, settings.symbol)
        close_position(client, settings.symbol)
        result["action"] = "CLOSE_POSITION"

    else:
        # Already positioned, signal still bullish — hold
        avg_entry = float(btc_pos.avg_entry_price)
        result["action"] = "ALREADY_POSITIONED"
        result["position_qty"] = float(btc_pos.qty)
        result["entry_price"] = avg_entry
        result["stop_loss"] = round(avg_entry - current_atr * settings.sl_atr_mult, 2)
        result["take_profit"] = round(avg_entry + current_atr * settings.tp_atr_mult, 2)

    return result

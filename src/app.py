"""
Flask web server + background trading loop.
Two pages: / (dashboard) and /backtest.
"""

import copy
import csv
import io
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.database import (
    get_closed_trades,
    get_daily_pnl,
    get_equity_history,
    get_open_trades,
    get_recent_runs,
    get_statistics,
    get_todays_statistics,
    init_db,
    mark_all_open_as_exited,
    store_run,
    store_trade,
    sync_closed_trades_only,
)
from src.model import load_metrics

logger = logging.getLogger(__name__)

settings = get_settings()

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"),
)

# In-memory store for the latest bot run (avoids DB round-trip on every page load)
_latest_run: dict = {}
_run_history: list[dict] = []
_run_lock = threading.Lock()

# Active trade state — entry/SL/TP managed by app-side exit_watchdog
_active_trade: dict = {}
_trade_lock = threading.Lock()
ACTIVE_TRADE_PATH = "data/active_trade.json"

# Trading client (shared)
_trading_client = None
try:
    from src.broker import get_client
    _trading_client = get_client(settings.alpaca_api_key, settings.alpaca_secret_key)
except Exception as e:
    logger.warning("Trading client not available: %s", e)


def _seed_history_from_db():
    """Populate in-memory run history from DB on startup so sparklines work after redeploy."""
    global _latest_run, _run_history
    try:
        rows = get_recent_runs(50)
        if rows:
            with _run_lock:
                _run_history = rows
                _latest_run = rows[0]
            logger.info("Seeded run history from DB: %d rows", len(rows))
    except Exception as e:
        logger.warning("Could not seed history: %s", e)


def _add_run(run: dict):
    global _latest_run, _run_history
    with _run_lock:
        _latest_run = run
        _run_history.insert(0, run)
        if len(_run_history) > 50:
            _run_history.pop()


def _get_latest() -> dict:
    with _run_lock:
        return copy.deepcopy(_latest_run)


def _get_history(n: int = 10) -> list[dict]:
    with _run_lock:
        return copy.deepcopy(_run_history[:n])


def _set_active_trade(trade: dict):
    global _active_trade
    with _trade_lock:
        _active_trade = trade.copy()
    try:
        os.makedirs("data", exist_ok=True)
        with open(ACTIVE_TRADE_PATH, "w") as f:
            json.dump(trade, f)
    except Exception as e:
        logger.warning("Could not save active trade: %s", e)


def _clear_active_trade():
    global _active_trade
    with _trade_lock:
        _active_trade = {}
    try:
        if os.path.exists(ACTIVE_TRADE_PATH):
            os.remove(ACTIVE_TRADE_PATH)
    except Exception:
        pass


def _get_active_trade() -> dict:
    with _trade_lock:
        return copy.deepcopy(_active_trade)


def _restore_active_trade():
    """On startup, reload active trade from disk if a position is still open."""
    if not os.path.exists(ACTIVE_TRADE_PATH):
        return
    try:
        with open(ACTIVE_TRADE_PATH) as f:
            trade = json.load(f)
        if not _trading_client:
            return
        positions = _trading_client.get_all_positions()
        sym = trade.get("symbol", settings.symbol).replace("/", "").upper()
        pos = next((p for p in positions if p.symbol.replace("/", "").upper() == sym), None)
        if pos:
            global _active_trade
            with _trade_lock:
                _active_trade = trade
            logger.info(
                "Active trade restored: entry=%.2f SL=%.2f TP=%.2f",
                trade.get("entry_price", 0), trade.get("sl_price", 0), trade.get("tp_price", 0),
            )
        else:
            os.remove(ACTIVE_TRADE_PATH)
            logger.info("Stale active trade file removed (no open position)")
    except Exception as e:
        logger.warning("Could not restore active trade: %s", e)


# ── Signal handlers ──────────────────────────────────────────────────────────

def _handle_shutdown(signum, frame):
    logger.warning("Shutdown signal %s — closing positions...", signum)
    if _trading_client:
        try:
            _trading_client.close_all_positions(cancel_orders=True)
        except Exception as e:
            logger.error("Shutdown cleanup: %s", e)
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich_with_account(run: dict) -> dict:
    if not _trading_client:
        return run
    try:
        from src.broker import get_account_info
        info = get_account_info(_trading_client)
        run.update(info)
    except Exception as e:
        logger.warning("Could not fetch account: %s", e)
    return run


def _active_positions() -> list:
    if not _trading_client:
        return []
    try:
        positions = _trading_client.get_all_positions()
        trade = _get_active_trade()
        out = []
        for p in positions:
            pos_sym = p.symbol.replace("/", "").upper()
            trade_sym = trade.get("symbol", "").replace("/", "").upper()
            if trade and trade_sym == pos_sym:
                sl_price = trade.get("sl_price")
                tp_price = trade.get("tp_price")
                entry_time = trade.get("entry_time")
            else:
                sl_price = tp_price = entry_time = None
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "entry_time": entry_time,
            })
        return out
    except Exception as e:
        logger.warning("Could not fetch positions: %s", e)
        return []


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def dashboard():
    run = _enrich_with_account(_get_latest())
    history = _get_history(10)
    active_positions = _active_positions()

    if _trading_client:
        try:
            sync_closed_trades_only(_trading_client)
        except Exception:
            pass

    closed_trades = get_closed_trades(50)
    stats = get_statistics()
    stats_today = get_todays_statistics()
    ml_metrics = load_metrics()

    bot_config = {
        "timeframe_hours":       settings.timeframe_hours,
        "sl_atr_mult":           settings.sl_atr_mult,
        "tp_atr_mult":           settings.tp_atr_mult,
        "confidence_threshold":  settings.confidence_threshold,
        "sentiment_floor":       settings.sentiment_floor,
        "max_risk_per_trade":    settings.max_risk_per_trade,
        "daily_loss_limit_pct":  settings.daily_loss_limit_pct,
        "max_hold_candles":      settings.max_hold_candles,
    }
    return render_template(
        "dashboard.html",
        run=run,
        history=history,
        active_positions=active_positions,
        closed_trades=closed_trades,
        stats=stats,
        stats_today=stats_today,
        ml_metrics=ml_metrics,
        bot_config=bot_config,
        timeframe_hours=settings.timeframe_hours,
    )


@app.route("/backtest")
def backtest_page():
    results = None
    if os.path.exists("models/backtest_results.json"):
        try:
            with open("models/backtest_results.json") as f:
                results = json.load(f)
        except Exception:
            pass
    ml_metrics = load_metrics()
    return render_template("backtest.html", results=results, ml_metrics=ml_metrics, timeframe_hours=settings.timeframe_hours)


@app.route("/static/models/<path:filename>")
def serve_model_image(filename):
    path = os.path.join("models", filename)
    if not os.path.isfile(path):
        abort(404)
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/png")


# ── JSON APIs ─────────────────────────────────────────────────────────────────

@app.route("/api/latest")
def api_latest():
    run = _enrich_with_account(_get_latest())
    if not run:
        return jsonify({"status": "waiting"})
    safe = {k: v for k, v in run.items() if k not in ("ohlcv_data", "chart_indicators")}
    return jsonify(safe)


def _next_candle_seconds() -> int:
    """Seconds until next candle close (aligned to timeframe_hours)."""
    import math
    from datetime import timedelta
    tf = settings.timeframe_hours
    now = datetime.now(timezone.utc)
    hours_f = now.hour + now.minute / 60 + now.second / 3600
    next_nth = math.ceil(hours_f / tf) * tf
    if next_nth >= 24:
        next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        next_close = now.replace(hour=int(next_nth), minute=0, second=0, microsecond=0)
    return max(int((next_close - now).total_seconds()), 0)


@app.route("/api/dashboard_data")
def api_dashboard_data():
    run = _enrich_with_account(_get_latest())
    if not run:
        return jsonify({"status": "waiting", "run": None})
    active_positions = _active_positions()
    closed_trades = get_closed_trades(50)
    history = _get_history(24)
    run_history_slim = [
        {"ts": r.get("timestamp"), "confidence": r.get("confidence", 0),
         "sentiment": r.get("sentiment_score", 0), "action": r.get("action", ""),
         "btc_close": r.get("btc_close", 0)}
        for r in history
    ]
    # Compute volatility regime from recent OHLCV if available
    vol_regime = "Normal"
    atr_pct = None
    if run.get("indicators") and run["indicators"].get("atr"):
        cur_atr = run["indicators"]["atr"]
        btc = run.get("btc_close", 1)
        atr_pct = round(cur_atr / btc * 100, 2) if btc else None
        if atr_pct:
            vol_regime = "High" if atr_pct > 2.0 else ("Low" if atr_pct < 0.8 else "Normal")
    # Strip bulky fields not needed by the dashboard poll
    run_slim = {k: v for k, v in run.items() if k not in ("ohlcv_data", "indicators", "chart_indicators")}
    return jsonify({
        "run": run_slim,
        "active_positions": active_positions,
        "closed_trades": closed_trades,
        "closed_trades_all": get_closed_trades(200),
        "ml_metrics": load_metrics(),
        "run_history": run_history_slim,
        "next_candle_seconds": _next_candle_seconds(),
        "vol_regime": vol_regime,
        "atr_pct": atr_pct,
        "stats": get_statistics(),
        "stats_today": get_todays_statistics(),
    })


@app.route("/api/chart_data")
def api_chart_data():
    """Return OHLCV + indicator time series for the candlestick chart.
    Serialized explicitly to avoid numpy/pandas dtype issues."""
    run = _get_latest()
    ohlcv = run.get("ohlcv_data", [])
    safe = []
    for row in ohlcv:
        safe_row = {}
        for k, v in row.items():
            if v is None:
                safe_row[k] = None
            elif isinstance(v, float) and (v != v):  # NaN
                safe_row[k] = None
            else:
                try:
                    safe_row[k] = float(v) if not isinstance(v, str) else v
                except (TypeError, ValueError):
                    safe_row[k] = str(v)
        safe.append(safe_row)
    return jsonify({"ohlcv": safe, "has_data": len(safe) > 0})


@app.route("/api/live_stats")
def api_live_stats():
    if not _trading_client:
        return jsonify({"error": "No trading client"}), 500
    try:
        from src.broker import get_account_info
        info = get_account_info(_trading_client)
        positions = _active_positions()
        info["active_positions"] = positions
        info["server_time"] = datetime.now(timezone.utc).isoformat()
        # Current BTC price: prefer live position price, fall back to last bot run
        if positions:
            info["btc_price"] = positions[0].get("current_price")
        else:
            info["btc_price"] = _get_latest().get("btc_close")
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades")
def api_trades():
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))
    return jsonify(get_closed_trades(limit, offset))


@app.route("/api/trades.csv")
def api_trades_csv():
    trades = get_closed_trades(10000)
    headers = ["entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
               "pnl_dollars", "pnl_percent", "duration_hours", "qty"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows([[t.get(h, "") for h in headers] for t in trades])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=trades.csv"})


@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    if not _trading_client:
        return jsonify({"error": "No trading client"}), 500
    try:
        latest = _get_latest()
        last_close = float(latest.get("btc_close", 0)) if latest else 0.0
        _trading_client.cancel_orders()
        _trading_client.close_all_positions(cancel_orders=True)
        n = mark_all_open_as_exited(exit_price=last_close, exit_reason="KILL_SWITCH")
        logger.warning("Kill switch: closed %d trade(s)", n)
        return jsonify({"status": "success", "trades_closed": n})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
def api_logs():
    lines = int(request.args.get("lines", 40))
    log_path = "logs/trading_bot.log"
    try:
        if not os.path.exists(log_path):
            return jsonify({"lines": []})
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        recent = [l.rstrip() for l in all_lines[-lines:]]
        return jsonify({"lines": recent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/daily_pnl")
def api_daily_pnl():
    days = int(request.args.get("days", 14))
    return jsonify(get_daily_pnl(days))


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update bot parameters in .env and reload settings in-memory."""
    allowed = {
        "CONFIDENCE_THRESHOLD", "SENTIMENT_FLOOR",
        "SL_ATR_MULT", "TP_ATR_MULT",
        "MAX_RISK_PER_TRADE", "DAILY_LOSS_LIMIT_PCT", "MAX_HOLD_CANDLES",
    }
    data = request.get_json(force=True) or {}
    updates = {k.upper(): str(v) for k, v in data.items() if k.upper() in allowed}
    if not updates:
        return jsonify({"error": "No valid fields"}), 400
    # Read existing .env
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
    # Update or append
    for key, val in updates.items():
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(key + "="):
                lines[i] = f"{key}={val}\n"
                found = True
                break
        if not found:
            lines.append(f"{key}={val}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    # Reload env vars for running process
    for key, val in updates.items():
        os.environ[key] = val
    # Reload settings
    global settings
    from importlib import reload
    import src.config as cfg_module
    reload(cfg_module)
    settings = cfg_module.get_settings()
    logger.info("Settings updated: %s", updates)
    return jsonify({"status": "ok", "updated": updates})


@app.route("/api/run_backtest", methods=["POST"])
def api_run_backtest():
    def _runner():
        try:
            from src.data import load_csv
            from src.backtest import run as run_backtest
            df = load_csv("data/btc_usd_hourly.csv", resample_to_4h=True)
            run_backtest(df, use_walkforward=True)
        except Exception as e:
            logger.error("Backtest failed: %s", e)

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/showcase")
def api_showcase():
    path = "models/showcase_results.json"
    if not os.path.exists(path):
        return jsonify({"status": "not_generated"})
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_showcase", methods=["POST"])
def api_run_showcase():
    def _runner():
        try:
            from run_showcase_backtest import run as run_showcase
            run_showcase(settings)
            logger.info("Showcase backtest complete")
        except Exception as e:
            logger.error("Showcase failed: %s", e)

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "started"})


# ── Bot loop ──────────────────────────────────────────────────────────────────

def _record_trade(run_id, run):
    if run.get("action") != "BUY_ORDER_SENT" or not run.get("order_id"):
        return
    try:
        store_trade(run_id, {
            "order_id": run["order_id"],
            "entry_price": float(run["entry_price"]),
            "entry_time": run.get("entry_time") or run.get("timestamp"),
            "qty": float(run.get("filled_qty") or run.get("position_qty") or 0),
        })
    except Exception as e:
        logger.warning("Could not record trade: %s", e)


def _maybe_set_active_trade(run: dict):
    """Persist active trade state on BUY or ALREADY_POSITIONED so SL/TP stay fresh."""
    action = run.get("action")
    if action not in ("BUY_ORDER_SENT", "ALREADY_POSITIONED"):
        return
    if not run.get("entry_price") or not run.get("stop_loss") or not run.get("take_profit"):
        return
    existing = copy.deepcopy(_active_trade)
    _set_active_trade({
        "symbol": settings.symbol,
        "entry_price": float(run["entry_price"]),
        "sl_price": float(run["stop_loss"]),
        "tp_price": float(run["take_profit"]),
        "entry_time": run.get("entry_time") or existing.get("entry_time") or run.get("timestamp"),
        "disaster_order_id": run.get("disaster_order_id") or existing.get("disaster_order_id"),
        "qty": float(run.get("filled_qty") or run.get("position_qty") or existing.get("qty") or 0),
    })
    if action == "BUY_ORDER_SENT":
        logger.info(
            "Active trade set: entry=%.2f SL=%.2f TP=%.2f",
            run["entry_price"], run["stop_loss"], run["take_profit"],
        )


def bot_loop():
    """Background trading loop (runs at each candle close per configured timeframe)."""
    from src.trading import trade_logic

    logger.info("Bot loop started (%dH timeframe)", settings.timeframe_hours)

    # Run immediately on startup
    try:
        result = trade_logic()
        if result:
            _add_run(result)
            run_id = store_run(result)
            _record_trade(run_id, result)
            _maybe_set_active_trade(result)
    except Exception as e:
        logger.error("Initial run failed: %s\n%s", e, traceback.format_exc())

    while True:
        try:
            import math
            from datetime import timedelta
            tf = settings.timeframe_hours
            now = datetime.now(timezone.utc)
            hours_f = now.hour + now.minute / 60 + now.second / 3600
            next_nth = math.ceil(hours_f / tf) * tf
            if next_nth >= 24:
                next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_close = now.replace(hour=int(next_nth), minute=0, second=0, microsecond=0)
            sleep_s = max((next_close - now).total_seconds() + 60, 60)

            logger.info("Sleeping %.0f min until next %dH candle close...", sleep_s / 60, tf)
            time.sleep(sleep_s)

            result = trade_logic()
            if result:
                _add_run(result)
                run_id = store_run(result)
                _record_trade(run_id, result)
                _maybe_set_active_trade(result)
                logger.info("Run complete: %s", result.get("action", "N/A"))
        except Exception as e:
            logger.error("Bot loop error: %s\n%s", e, traceback.format_exc())
            time.sleep(60)


def exit_watchdog():
    """
    Poll every 10s and close the position when SL or TP is hit (app-side),
    or when the max hold candle limit is exceeded.
    Alpaca only holds a wide disaster stop as a crash safety net.
    """
    if not _trading_client:
        return
    logger.info("Exit watchdog started (interval=10s)")
    while True:
        try:
            time.sleep(10)
            trade = _get_active_trade()
            if not trade:
                continue

            sl_price = trade.get("sl_price")
            tp_price = trade.get("tp_price")
            entry_time_str = trade.get("entry_time")
            symbol = trade.get("symbol", settings.symbol)

            if not sl_price or not tp_price:
                continue

            positions = _trading_client.get_all_positions()
            sym_norm = symbol.replace("/", "").upper()
            pos = next((p for p in positions if p.symbol.replace("/", "").upper() == sym_norm), None)

            if not pos:
                # Position gone externally (disaster stop fired, manual close, etc.)
                logger.info("Exit watchdog: position gone — clearing active trade")
                _clear_active_trade()
                continue

            current_price = float(pos.current_price)
            exit_reason = None

            if current_price <= sl_price:
                exit_reason = "SL_HIT"
                logger.warning("Exit watchdog SL hit: %.2f <= %.2f", current_price, sl_price)
            elif current_price >= tp_price:
                exit_reason = "TP_HIT"
                logger.warning("Exit watchdog TP hit: %.2f >= %.2f", current_price, tp_price)
            elif entry_time_str:
                try:
                    from datetime import timedelta
                    et = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    held_hours = (datetime.now(timezone.utc) - et).total_seconds() / 3600
                    max_hold_hours = settings.max_hold_candles * settings.timeframe_hours
                    if held_hours >= max_hold_hours:
                        exit_reason = "MAX_HOLD"
                        logger.warning(
                            "Exit watchdog MAX_HOLD: %.1fh >= %dh", held_hours, max_hold_hours
                        )
                except Exception:
                    pass

            if exit_reason:
                from src.broker import cancel_orders, close_position
                cancel_orders(_trading_client, symbol)
                close_position(_trading_client, symbol)
                n = mark_all_open_as_exited(exit_price=current_price, exit_reason=exit_reason)
                logger.info("Exit watchdog closed %d trade(s): %s @ %.2f", n, exit_reason, current_price)
                _clear_active_trade()

        except Exception as e:
            logger.error("Exit watchdog error: %s", e)


if __name__ == "__main__":
    init_db()
    _restore_active_trade()
    threading.Thread(target=bot_loop, daemon=True).start()
    threading.Thread(target=exit_watchdog, daemon=True).start()
    port = int(os.environ.get("PORT", settings.flask_port))
    logger.info("Dashboard starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

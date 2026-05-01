"""
Flask web server + trading bot loop.
Entry point for Railway deployment.

Supports graceful shutdown via SIGTERM/SIGINT.
"""

import copy
import csv
import io
import json
import os
import sys
import time
import signal
import logging
import threading
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request, Response, abort
from src.config import get_settings
from src.trading_bot_multi import (
    trade_logic_multi,
    SYMBOL_BTC,
    CONFIDENCE_THRESHOLD,
    SENTIMENT_FLOOR,
    SL_ATR_MULT,
    TP_ATR_MULT,
    MAX_RISK_PER_TRADE,
    SL_LIMIT_SLIPPAGE,
    cancel_open_orders_for_symbol,
    is_shutdown_requested,
)
from src.run_store import add_run, get_latest, get_last_n
from src.charts import build_candlestick_chart, build_equity_chart
from src.database import (
    init_db,
    store_run,
    store_trade,
    get_recent_runs,
    get_closed_trades,
    get_open_trades,
    get_statistics,
    get_todays_statistics,
    get_equity_history,
    sync_trades_from_alpaca,
    sync_closed_trades_only,
    mark_all_open_as_exited,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Load settings
settings = get_settings()

# --- Configuration ---
API_KEY = settings.alpaca_api_key
SECRET_KEY = settings.alpaca_secret_key
TP_WATCHDOG_INTERVAL_S = settings.tp_watchdog_interval_s
DB_PATH = settings.db_path

trading_client = (
    TradingClient(API_KEY, SECRET_KEY, paper=True) if API_KEY and SECRET_KEY else None
)

# Logging configuration — configure root logger so all modules (including
# trading_bot_multi) emit to stdout in Railway
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
    ),
)

# Global shutdown flag
_shutdown_requested = False


def handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    logger.warning("Shutdown signal %s received — initiating cleanup...", signum)
    _shutdown_requested = True
    if trading_client:
        try:
            trading_client.close_all_positions(cancel_orders=True)
            logger.info("All positions closed via API.")
        except Exception as e:
            logger.error("Shutdown cleanup error: %s", e)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


# ==============================================================
# Helpers
# ==============================================================


def fetch_active_positions_with_sl_tp():
    """Return list of open positions enriched with SL/TP prices from resting orders."""
    if not trading_client:
        return []
    positions = trading_client.get_all_positions()
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN)
    )

    out = []
    for p in positions:
        side_str = str(p.side).split(".")[-1].lower() if p.side else "unknown"
        sl_price = None
        tp_price = None
        # Position symbols come without slash (e.g. BTCUSD); order symbols with slash.
        pos_sym = p.symbol.replace("/", "").upper()
        for o in open_orders:
            if o.symbol.replace("/", "").upper() != pos_sym:
                continue

            otype = str(o.order_type).lower()
            oside = str(o.side).lower()

            if "sell" in oside:
                if "stop" in otype and o.stop_price:
                    sl_price = float(o.stop_price)
                elif "limit" in otype and o.limit_price:
                    tp_price = float(o.limit_price)

        # Fallback for soft Take Profit from the latest bot run
        if tp_price is None:
            latest = get_latest()
            if latest and latest.get("take_profit") and pos_sym == "BTCUSD":
                tp_price = float(latest["take_profit"])

        out.append(
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": side_str,
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "sl_price": sl_price,
                "tp_price": tp_price,
            }
        )
    return out


def maybe_record_trade(run_id: int, run: dict) -> None:
    """If the run resulted in a filled BUY, persist a trade row keyed by run_id."""
    if not run_id or not run:
        return
    if run.get("action") != "BUY_ORDER_SENT":
        return
    if not run.get("order_id") or not run.get("entry_price"):
        return
    try:
        store_trade(
            run_id,
            {
                "order_id": run["order_id"],
                "entry_price": float(run["entry_price"]),
                "entry_time": run.get("entry_time") or run.get("timestamp"),
                "qty": float(run.get("filled_qty") or run.get("position_qty") or 0),
            },
        )
        logger.info(
            "Trade recorded in ledger: order_id=%s run_id=%s", run["order_id"], run_id
        )
    except Exception as e:
        logger.warning("Could not record trade in ledger: %s", e)


# ==============================================================
# Pages
# ==============================================================


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def dashboard():
    # Deep-copy so live-data overrides never mutate the cached run in run_store.
    run = copy.deepcopy(get_latest())
    runs = get_last_n(10)

    active_positions = []
    if trading_client and run:
        try:
            account = trading_client.get_account()
            run["equity"] = float(account.equity)
            run["buying_power"] = float(account.buying_power)
            last_equity = float(account.last_equity)
            pnl = run["equity"] - last_equity
            run["pnl_today"] = pnl
            run["pnl_today_pct"] = (pnl / last_equity) * 100 if last_equity > 0 else 0

            active_positions = fetch_active_positions_with_sl_tp()
        except Exception as e:
            logger.error("Failed to fetch live account data: %s", e)

    # Lightweight: only close any newly-filled SELLs since last sync.
    if trading_client:
        try:
            sync_closed_trades_only(trading_client)
        except Exception as e:
            logger.warning("Incremental trade sync skipped: %s", e)

    closed_trades = get_closed_trades(20)
    stats_overall = get_statistics()
    stats_today = get_todays_statistics()

    chart_json = (
        build_candlestick_chart(run, active_positions, closed_trades) if run else "{}"
    )
    equity_history = get_equity_history()
    equity_chart_json = build_equity_chart(equity_history)

    # Load ML metrics if available
    ml_metrics = {}
    try:
        with open("models/ml_metrics.json", "r") as f:
            ml_metrics = json.load(f)
    except Exception:
        pass

    return render_template(
        "dashboard.html",
        run=run,
        runs=runs,
        chart_json=chart_json,
        equity_chart_json=equity_chart_json,
        closed_trades=closed_trades,
        stats_overall=stats_overall,
        stats_today=stats_today,
        active_positions=active_positions,
        ml_metrics=ml_metrics,
        CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD,
        SENTIMENT_FLOOR=SENTIMENT_FLOOR,
        SL_ATR_MULT=SL_ATR_MULT,
        TP_ATR_MULT=TP_ATR_MULT,
        MAX_RISK_PER_TRADE=MAX_RISK_PER_TRADE,
    )


@app.route("/methodology")
def methodology():
    """Static methodology page describing the bot's pipeline."""
    return render_template(
        "methodology.html",
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sentiment_floor=SENTIMENT_FLOOR,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
        max_risk=MAX_RISK_PER_TRADE,
    )


@app.route("/config")
def config_page():
    """Read-only display of the bot's runtime configuration."""
    model_path = os.getenv("MODEL_PATH", "models/rf_model.pkl")
    features_path = os.getenv("FEATURES_PATH", "models/model_features.pkl")
    model_info = {"exists": False}
    try:
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            model_info = {
                "exists": True,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
            }
    except Exception:
        pass

    features = []
    try:
        import joblib

        if os.path.exists(features_path):
            features = joblib.load(features_path)
    except Exception as e:
        logger.warning("Could not load feature list: %s", e)

    return render_template(
        "config.html",
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sentiment_floor=SENTIMENT_FLOOR,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
        sl_slippage=SL_LIMIT_SLIPPAGE,
        max_risk=MAX_RISK_PER_TRADE,
        tp_watchdog_interval=TP_WATCHDOG_INTERVAL_S,
        symbol=SYMBOL_BTC,
        model_info=model_info,
        features=features,
    )


@app.route("/backtest")
def backtest_page():
    """Backtest results — image + stats from backtest_results.json."""
    results = None
    json_path = "models/backtest_results.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                results = json.load(f)
        except Exception as e:
            logger.warning("Could not read backtest_results.json: %s", e)

    # Most recent backtest image (any file matching dream_dynamic_*.png)
    image_url = None
    try:
        models_dir = "models"
        if os.path.isdir(models_dir):
            pngs = sorted(
                [
                    f
                    for f in os.listdir(models_dir)
                    if f.startswith("dream_dynamic") and f.endswith(".png")
                ],
                key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
                reverse=True,
            )
            if pngs:
                image_url = f"/static/backtest/{pngs[0]}"
    except Exception:
        pass

    return render_template("backtest.html", results=results, image_url=image_url)


# Serve the backtest image without copying it to static/
@app.route("/static/backtest/<path:filename>")
def static_backtest(filename):
    full = os.path.join("models", filename)
    if not os.path.isfile(full):
        abort(404)
    with open(full, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/png")


# ==============================================================
# JSON APIs
# ==============================================================


@app.route("/api/latest")
def api_latest():
    run = copy.deepcopy(get_latest())
    if run:
        if trading_client:
            try:
                account = trading_client.get_account()
                run["equity"] = float(account.equity)
                run["buying_power"] = float(account.buying_power)
                last_equity = float(account.last_equity)
                pnl = run["equity"] - last_equity
                run["pnl_today"] = pnl
                run["pnl_today_pct"] = (
                    (pnl / last_equity) * 100 if last_equity > 0 else 0
                )
            except Exception as e:
                logger.error("Failed to fetch live account data for API: %s", e)

        # Strip large chart data from API response
        safe = {
            k: v for k, v in run.items() if k not in ("ohlcv_data", "chart_indicators")
        }
        return jsonify(safe)
    return jsonify({"status": "waiting_for_first_run"})


@app.route("/api/runs")
def api_runs():
    runs = get_last_n(10)
    safe_runs = []
    for r in runs:
        safe_runs.append(
            {k: v for k, v in r.items() if k not in ("ohlcv_data", "chart_indicators")}
        )
    return jsonify(safe_runs)


@app.route("/api/trades")
def api_trades():
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))
    trades = get_closed_trades(limit, offset)
    return jsonify(trades)


@app.route("/api/live_stats")
def api_live_stats():
    if not trading_client:
        return jsonify({"error": "No trading client configured"}), 500

    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        last_equity = float(account.last_equity)
        pnl = equity - last_equity
        pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0

        return jsonify(
            {
                "equity": equity,
                "buying_power": buying_power,
                "pnl_today": pnl,
                "pnl_today_pct": pnl_pct,
                "active_positions": fetch_active_positions_with_sl_tp(),
                "server_time": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logger.error("Live stats API error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/charts")
def api_charts():
    run = copy.deepcopy(get_latest())
    active_positions = fetch_active_positions_with_sl_tp() if trading_client else []
    closed_trades = get_closed_trades(100)

    chart_json = (
        build_candlestick_chart(run, active_positions, closed_trades) if run else "{}"
    )
    equity_history = get_equity_history()
    equity_chart_json = build_equity_chart(equity_history)

    # Load ML metrics if available
    ml_metrics = {}
    try:
        import json as _json

        with open("models/ml_metrics.json", "r") as _f:
            ml_metrics = _json.load(_f)
    except Exception:
        pass

    return jsonify(
        {
            "candlestick": chart_json,
            "equity": equity_chart_json,
            "ml_metrics": ml_metrics,
        }
    )


@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    if not trading_client:
        return jsonify({"error": "No trading client configured"}), 500

    try:
        logger.warning("KILL SWITCH ACTIVATED")

        # Capture last known close before closing, so the DB rows record a sensible exit price.
        latest = get_latest()
        last_close = (
            float(latest.get("btc_close"))
            if latest and latest.get("btc_close")
            else 0.0
        )

        trading_client.cancel_orders()
        logger.warning("All open orders cancelled.")

        trading_client.close_all_positions(cancel_orders=True)
        logger.warning("All positions closed.")

        # Reflect the closure in the local ledger immediately.
        n = mark_all_open_as_exited(exit_price=last_close, exit_reason="KILL_SWITCH")
        logger.warning("Marked %d open trade(s) as KILL_SWITCH-exited in DB.", n)

        return jsonify(
            {
                "status": "success",
                "message": f"All orders cancelled, positions closed, {n} trade row(s) marked exited.",
            }
        )
    except Exception as e:
        logger.error("Kill switch error: %s", e)
        return jsonify({"error": str(e)}), 500


# ==============================================================
# CSV exports
# ==============================================================


def _csv_response(headers: list, rows: list, filename: str) -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/trades.csv")
def api_trades_csv():
    trades = get_closed_trades(10000)
    headers = [
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "pnl_dollars",
        "pnl_percent",
        "duration_hours",
        "qty",
    ]
    rows = [[t.get(h, "") for h in headers] for t in trades]
    return _csv_response(headers, rows, "trades.csv")


@app.route("/api/runs.csv")
def api_runs_csv():
    runs = get_recent_runs(10000)
    headers = [
        "timestamp",
        "btc_close",
        "prediction_label",
        "confidence",
        "sentiment_score",
        "action",
        "stop_loss",
        "take_profit",
        "position_qty",
        "leverage",
        "error",
    ]
    rows = [[r.get(h, "") for h in headers] for r in runs]
    return _csv_response(headers, rows, "runs.csv")


# ==============================================================
# Admin
# ==============================================================


@app.route("/admin/resync", methods=["POST"])
def admin_resync():
    """Heavy full reconciliation from Alpaca. Rarely needed."""
    if not trading_client:
        return jsonify({"error": "No trading client configured"}), 500
    try:
        sync_trades_from_alpaca(trading_client)
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error("Resync error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_backtest", methods=["POST"])
def api_run_backtest():
    """Kick off a backtest in a background thread. Non-blocking."""

    def _runner():
        try:
            from src.backtest_dynamic import run_backtest  # type: ignore

            run_backtest()
        except Exception as e:
            logger.error("Backtest run failed: %s", e)

    threading.Thread(target=_runner, daemon=True).start()
    return jsonify({"status": "started"})


# ==============================================================
# Bot loop + TP watchdog
# ==============================================================


def bot_loop():
    """Background trading bot loop — runs hourly."""
    logger.info("Bot loop started. Running initial analysis...")

    try:
        result = trade_logic_multi()
        if result:
            add_run(result)
            run_id = store_run(result)
            maybe_record_trade(run_id, result)
            logger.info("Initial run complete. Action: %s", result.get("action", "N/A"))
    except Exception as e:
        logger.error("Initial run failed: %s", e)
        logger.error(traceback.format_exc())

    while True:
        try:
            now = datetime.now()
            next_hour = (
                now.replace(minute=0, second=0, microsecond=0).timestamp() + 3600
            )
            sleep_seconds = next_hour - now.timestamp() + 60  # 60s buffer

            logger.info("Sleeping %d min until next candle...", int(sleep_seconds / 60))
            time.sleep(sleep_seconds)

            logger.info("Running market analysis...")
            result = trade_logic_multi()
            if result:
                add_run(result)
                run_id = store_run(result)
                maybe_record_trade(run_id, result)
                logger.info("Run complete. Action: %s", result.get("action", "N/A"))

        except Exception as e:
            logger.error("Bot loop error: %s", e)
            logger.error(traceback.format_exc())
            time.sleep(60)


def tp_watchdog_loop():
    """
    Soft Take-Profit watchdog. Alpaca crypto only allows one resting SELL order
    against position quantity (no OCO/bracket), so we hold the SL on Alpaca and
    enforce TP in code. Polls every TP_WATCHDOG_INTERVAL_S seconds.
    """
    if not trading_client:
        logger.warning("TP watchdog disabled: no trading client configured.")
        return

    logger.info("TP watchdog started (interval=%ds).", TP_WATCHDOG_INTERVAL_S)
    while True:
        try:
            time.sleep(TP_WATCHDOG_INTERVAL_S)

            latest = get_latest()
            if not latest:
                continue
            tp_target = latest.get("take_profit")
            if not tp_target or tp_target <= 0:
                continue

            positions = trading_client.get_all_positions()
            btc_pos = next(
                (p for p in positions if p.symbol.replace("/", "").upper() == "BTCUSD"),
                None,
            )
            if not btc_pos:
                continue

            current_price = float(btc_pos.current_price)
            if current_price < float(tp_target):
                continue

            logger.warning(
                "TP watchdog: %.2f >= %.2f. Closing position.", current_price, tp_target
            )
            cancel_open_orders_for_symbol(trading_client, "BTC/USD")
            trading_client.close_position("BTC/USD")

            # Reflect TP exit in local ledger right away (don't wait for next sync).
            try:
                exit_time = datetime.now(timezone.utc).isoformat()
                from src.database import _db_lock  # internal; small leak but OK
                import sqlite3

                with _db_lock:
                    with sqlite3.connect("data/trading_bot.db") as conn:
                        c = conn.cursor()
                        c.execute(
                            """
                            UPDATE trades SET exit_price = ?, exit_time = ?,
                                exit_reason = 'Take Profit'
                            WHERE exit_price IS NULL
                        """,
                            (current_price, exit_time),
                        )
                        conn.commit()
            except Exception as e:
                logger.warning("TP watchdog: could not update DB row: %s", e)
        except Exception as e:
            logger.error("TP watchdog error: %s", e)


if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()

    # Start bot in background thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    # Start TP watchdog
    tp_thread = threading.Thread(target=tp_watchdog_loop, daemon=True)
    tp_thread.start()

    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting dashboard on port %s...", port)
    app.run(host="0.0.0.0", port=port, debug=False)

"""
Flask web server + trading bot loop.
Entry point for Railway deployment.
"""
import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from src.trading_bot_multi import trade_logic_multi
from src.run_store import add_run, get_latest, get_last_n
from src.charts import build_candlestick_chart
from src.database import init_db, store_run, get_recent_runs, get_closed_trades, get_statistics, get_todays_statistics
from alpaca.trading.client import TradingClient

# Initialize Alpaca Client
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True) if API_KEY and SECRET_KEY else None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, template_folder=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'
))


@app.route("/")
def dashboard():
    run = get_latest()
    runs = get_last_n(10)
    chart_json = build_candlestick_chart(run) if run else "{}"

    # Get live account data and active positions if possible
    active_positions = []
    if trading_client and run:
        try:
            account = trading_client.get_account()
            # Override run metrics with live metrics
            run["equity"] = float(account.equity)
            run["buying_power"] = float(account.buying_power)
            last_equity = float(account.last_equity)
            pnl = run["equity"] - last_equity
            run["pnl_today"] = pnl
            run["pnl_today_pct"] = (pnl / last_equity) * 100 if last_equity > 0 else 0

            # Get open positions
            positions = trading_client.get_all_positions()
            for p in positions:
                side_str = str(p.side).split('.')[-1].lower() if p.side else 'unknown'
                active_positions.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": side_str,
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc) * 100
                })
        except Exception as e:
            logger.error(f"Failed to fetch live account data: {e}")

    # Get trade data
    if trading_client:
        from src.database import sync_trades_from_alpaca
        sync_trades_from_alpaca(trading_client)
    
    closed_trades = get_closed_trades(20)
    stats_overall = get_statistics()
    stats_today = get_todays_statistics()

    return render_template("dashboard.html",
                         run=run, runs=runs, chart_json=chart_json,
                         closed_trades=closed_trades,
                         stats_overall=stats_overall,
                         stats_today=stats_today,
                         active_positions=active_positions)


@app.route("/api/latest")
def api_latest():
    run = get_latest()
    if run:
        if trading_client:
            try:
                account = trading_client.get_account()
                run["equity"] = float(account.equity)
                run["buying_power"] = float(account.buying_power)
                last_equity = float(account.last_equity)
                pnl = run["equity"] - last_equity
                run["pnl_today"] = pnl
                run["pnl_today_pct"] = (pnl / last_equity) * 100 if last_equity > 0 else 0
            except Exception as e:
                logger.error(f"Failed to fetch live account data for API: {e}")

        # Strip large chart data from API response
        safe = {k: v for k, v in run.items() if k not in ("ohlcv_data", "chart_indicators")}
        return jsonify(safe)
    return jsonify({"status": "waiting_for_first_run"})


@app.route("/api/runs")
def api_runs():
    runs = get_last_n(10)
    safe_runs = []
    for r in runs:
        safe_runs.append({k: v for k, v in r.items() if k not in ("ohlcv_data", "chart_indicators")})
    return jsonify(safe_runs)


def bot_loop():
    """Background trading bot loop — runs hourly, same as main.py."""
    logger.info("Bot loop started. Running initial analysis...")

    # Run immediately on startup so the dashboard has data
    try:
        result = trade_logic_multi()
        if result:
            add_run(result)
            store_run(result)  # Persist to database
            logger.info(f"Initial run complete. Action: {result.get('action', 'N/A')}")
    except Exception as e:
        logger.error(f"Initial run failed: {e}")
        logger.error(traceback.format_exc())

    # Then run on hourly schedule
    while True:
        try:
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0).timestamp() + 3600
            sleep_seconds = next_hour - now.timestamp() + 60  # 60s buffer

            logger.info(f"Sleeping {int(sleep_seconds / 60)} min until next candle...")
            time.sleep(sleep_seconds)

            logger.info("Running market analysis...")
            result = trade_logic_multi()
            if result:
                add_run(result)
                store_run(result)  # Persist to database
                logger.info(f"Run complete. Action: {result.get('action', 'N/A')}")

        except Exception as e:
            logger.error(f"Bot loop error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    # Initialize database
    logger.info("Initializing database...")
    init_db()

    # Start bot in background thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting dashboard on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

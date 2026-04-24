"""
Flask web server + trading bot loop.
Entry point for Railway deployment.
"""
import copy
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

from flask import Flask, render_template, jsonify, request
from src.trading_bot_multi import trade_logic_multi
from src.run_store import add_run, get_latest, get_last_n
from src.charts import build_candlestick_chart, build_equity_chart
from src.database import init_db, store_run, get_recent_runs, get_closed_trades, get_statistics, get_todays_statistics, get_equity_history
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

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


def fetch_active_positions_with_sl_tp():
    """Return list of open positions enriched with SL/TP prices from resting orders."""
    if not trading_client:
        return []
    positions = trading_client.get_all_positions()
    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))

    out = []
    for p in positions:
        side_str = str(p.side).split('.')[-1].lower() if p.side else 'unknown'
        sl_price = None
        tp_price = None
        # Position symbols come without slash (e.g. BTCUSD); order symbols with slash.
        pos_sym = p.symbol.replace("/", "").upper()
        for o in open_orders:
            if o.symbol.replace("/", "").upper() != pos_sym:
                continue
            
            # Alpaca SDK uses enums; check both .name and .value for robustness
            otype = str(o.order_type).lower()
            oside = str(o.side).lower()
            
            # SL orders are SELL orders for long positions
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

        out.append({
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
        })
    return out


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
            logger.error(f"Failed to fetch live account data: {e}")

    # Get trade data
    if trading_client:
        from src.database import sync_trades_from_alpaca
        sync_trades_from_alpaca(trading_client)
    
    closed_trades = get_closed_trades(20)
    stats_overall = get_statistics()
    stats_today = get_todays_statistics()

    # Build chart after getting active_positions and closed_trades
    chart_json = build_candlestick_chart(run, active_positions, closed_trades) if run else "{}"
    
    # Build Equity Curve Chart
    equity_history = get_equity_history()
    equity_chart_json = build_equity_chart(equity_history)

    return render_template("dashboard.html",
                         run=run, runs=runs, chart_json=chart_json,
                         equity_chart_json=equity_chart_json,
                         closed_trades=closed_trades,
                         stats_overall=stats_overall,
                         stats_today=stats_today,
                         active_positions=active_positions)


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

        return jsonify({
            "equity": equity,
            "buying_power": buying_power,
            "pnl_today": pnl,
            "pnl_today_pct": pnl_pct,
            "active_positions": fetch_active_positions_with_sl_tp(),
        })
    except Exception as e:
        logger.error(f"Live stats API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/charts")
def api_charts():
    run = copy.deepcopy(get_latest())
    active_positions = fetch_active_positions_with_sl_tp() if trading_client else []
    
    # Get historical trades for annotations
    closed_trades = get_closed_trades(100) # Get more trades for historical plotting
    
    chart_json = build_candlestick_chart(run, active_positions, closed_trades) if run else "{}"
    
    equity_history = get_equity_history()
    equity_chart_json = build_equity_chart(equity_history)
    
    return jsonify({
        "candlestick": chart_json,
        "equity": equity_chart_json
    })

@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    if not trading_client:
        return jsonify({"error": "No trading client configured"}), 500
    
    try:
        logger.warning("🚨 KILL SWITCH ACTIVATED 🚨")
        # 1. Cancel all open orders
        trading_client.cancel_orders()
        logger.warning("All open orders cancelled.")
        
        # 2. Close all positions at market price
        trading_client.close_all_positions(cancel_orders=True)
        logger.warning("All positions closed.")
        
        return jsonify({"status": "success", "message": "All orders cancelled and positions closed."})
    except Exception as e:
        logger.error(f"Kill switch error: {e}")
        return jsonify({"error": str(e)}), 500

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

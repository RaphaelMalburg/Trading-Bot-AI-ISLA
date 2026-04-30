import json
import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe (Railway has no GUI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_backtest_dynamic(initial_capital=100, base_leverage=3.0, sl_atr=1.0, tp_atr=3.0, confidence_threshold=0.55):
    """
    Dynamic backtest of the Random Forest strategy with compound interest and
    risk-budget position sizing.

    Long-only on Bitcoin: position size adapts to volatility (ATR) and is capped
    so a stop-loss hit costs at most `max_risk_per_trade_pct` of current capital.

    Args:
        initial_capital (float): Starting capital in USD.
        base_leverage (float): Reserved (risk budget already governs sizing).
        sl_atr (float): Stop-loss distance multiplier on ATR.
        tp_atr (float): Take-profit distance multiplier on ATR.
        confidence_threshold (float): Minimum probability to enter a trade.

    Returns a dict of summary metrics (also written to models/backtest_results.json).
    """
    logger.info("Starting dynamic backtest (compound + risk-adjusted sizing)...")

    try:
        df = pd.read_csv("data/processed_data.csv")
        model = joblib.load("models/rf_model.pkl")
        features = joblib.load("models/model_features.pkl")
    except Exception as e:
        logger.error("Could not load files for backtest: %s", e)
        return None

    # Use only the last 20% — unseen during training
    split_index = int(len(df) * 0.8)
    df_test = df.iloc[split_index:].copy().reset_index(drop=True)

    # Probability of class 1 (price going up)
    X_test = df_test[features]
    probs = model.predict_proba(X_test)[:, 1]

    # Same threshold as the live bot
    df_test['prediction'] = (probs > confidence_threshold).astype(int)

    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    position_size_locked = 0
    trades = []
    equity_curve = [initial_capital]
    equity_dates = [pd.to_datetime(df_test.loc[0, 'timestamp'])]
    btc_prices = [df_test.loc[0, 'close']]
    position_sizes = []
    trade_dates = []

    fee_pct = 0.0005           # simulated 0.05% trading fee per side
    max_risk_per_trade_pct = 0.05

    for i in range(len(df_test) - 1):
        if capital <= 10:
            break

        current_price = df_test.loc[i, 'close']
        current_date = pd.to_datetime(df_test.loc[i, 'timestamp'])
        atr = df_test.loc[i, 'atr_14']
        pred = df_test.loc[i, 'prediction']

        dist_stop = atr * sl_atr
        if dist_stop <= 0 or current_price <= 0:
            equity_curve.append(capital)
            equity_dates.append(current_date)
            btc_prices.append(current_price)
            continue

        stop_pct = dist_stop / current_price
        ideal_position_size = (capital * max_risk_per_trade_pct) / stop_pct
        position_size = min(ideal_position_size, capital * 5.0)
        position_size = max(position_size, capital * 1.0)
        current_leverage = position_size / capital if capital > 0 else 0

        # Exit logic
        if position == 1:
            if current_price <= stop_loss:
                pct_loss = (stop_loss - entry_price) / entry_price
                pnl = position_size_locked * pct_loss
                cost = position_size_locked * fee_pct
                capital += (pnl - cost)
                trades.append(pnl - cost)
                position = 0
            elif current_price >= take_profit:
                pct_gain = (take_profit - entry_price) / entry_price
                pnl = position_size_locked * pct_gain
                cost = position_size_locked * fee_pct
                capital += (pnl - cost)
                trades.append(pnl - cost)
                position = 0

        # Entry logic
        if position == 0 and pred == 1:
            position = 1
            entry_price = current_price
            stop_loss = current_price - dist_stop
            take_profit = current_price + (atr * tp_atr)
            position_size_locked = position_size
            position_sizes.append(current_leverage)
            trade_dates.append(current_date)
            capital -= position_size_locked * fee_pct

        equity_curve.append(capital)
        equity_dates.append(current_date)
        btc_prices.append(current_price)

    # Summary
    total_return = (capital - initial_capital) / initial_capital * 100
    avg_leverage = float(np.mean(position_sizes)) if position_sizes else 0
    btc_start_price = btc_prices[0]
    btc_end_price = btc_prices[-1]
    buy_and_hold_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100 if btc_start_price else 0

    # Metrics
    total_trades = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    best_trade = float(max(trades)) if trades else 0
    worst_trade = float(min(trades)) if trades else 0

    # Max drawdown
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks * 100
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0

    # Sharpe (hourly returns annualized to daily for crypto 24/7)
    eq_returns = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = float((np.mean(eq_returns) / np.std(eq_returns)) * np.sqrt(24 * 365)) if eq_returns.size and np.std(eq_returns) > 0 else 0

    logger.info("=" * 40)
    logger.info("Confidence threshold: %.2f", confidence_threshold)
    logger.info("Period: %s to %s", equity_dates[0].strftime('%Y-%m-%d'), equity_dates[-1].strftime('%Y-%m-%d'))
    logger.info("Final capital: $%.2f", capital)
    logger.info("Strategy return: %+.2f%%", total_return)
    logger.info("Buy & Hold return: %+.2f%%", buy_and_hold_return)
    logger.info("Avg leverage used: %.2fx", avg_leverage)
    logger.info("Worst capital point: $%.2f", min(equity_curve))
    logger.info("Total trades: %d | Win rate: %.1f%%", total_trades, win_rate)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(equity_dates, equity_curve, color='#1f6feb')
    plt.title(f'Equity Curve — Strategy ({total_return:+.0f}%) vs Buy & Hold ({buy_and_hold_return:+.0f}%)')
    plt.ylabel('Capital ($)')
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(equity_dates, btc_prices, color='gray')
    plt.title('BTC/USD over the same out-of-sample period')
    plt.ylabel('Price ($)')

    plt.subplot(3, 1, 3)
    plt.plot(trade_dates, position_sizes, color='orange', alpha=0.7, marker='.')
    plt.title('Leverage used per trade')
    plt.ylabel('Leverage (x)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    image_path = f'models/dream_dynamic_{confidence_threshold}.png'
    plt.savefig(image_path)
    plt.close()
    logger.info("Chart saved: %s", image_path)

    # Persist machine-readable summary for /backtest page
    summary = {
        "confidence_threshold": confidence_threshold,
        "period_start": equity_dates[0].strftime('%Y-%m-%d'),
        "period_end": equity_dates[-1].strftime('%Y-%m-%d'),
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return": round(total_return, 2),
        "buy_hold_return": round(buy_and_hold_return, 2),
        "avg_leverage": round(avg_leverage, 2),
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "image": image_path,
    }
    try:
        with open("models/backtest_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary written to models/backtest_results.json")
    except Exception as e:
        logger.warning("Could not write backtest_results.json: %s", e)

    return summary


def run_backtest():
    """Public entry point used by /api/run_backtest."""
    return run_backtest_dynamic()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    run_backtest_dynamic(confidence_threshold=threshold)

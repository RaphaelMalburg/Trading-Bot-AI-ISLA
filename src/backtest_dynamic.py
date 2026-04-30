"""
Enhanced backtest with walk-forward validation support.
Key improvement: rolling window retraining simulates production ML lifecycle.
"""

import json
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class BacktestResult(TypedDict, total=False):
    total_return: float
    buy_hold_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_leverage: float
    period_start: str
    period_end: str
    final_capital: float
    image: str
    walkforward: bool
    n_splits: int


def run_backtest_dynamic(
    initial_capital: float = 100.0,
    sl_atr: float = 1.0,
    tp_atr: float = 2.0,
    confidence_threshold: float = 0.55,
    use_walkforward: bool = False,
    n_splits: int = 5,
    retrain_every: int = 100,
) -> Optional[BacktestResult]:
    """
    Run dynamic backtest with compounding and risk-adjusted position sizing.

    Args:
        use_walkforward: If True, re-train model on rolling windows (more robust).
                        If False, use single train/test split (faster).
    """
    logger.info("Loading data and model...")
    try:
        df = pd.read_csv("data/processed_data.csv")
        model = joblib.load("models/rf_model.pkl")
        features = joblib.load("models/model_features.pkl")
    except Exception as e:
        logger.error("Load failed: %s", e)
        return None

    # Split
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    if use_walkforward:
        logger.info("Walk-forward mode: %d splits", n_splits)
        result = _run_walkforward(
            df_test, initial_capital, sl_atr, tp_atr, confidence_threshold, n_splits
        )
    else:
        logger.info("Single-split mode (baseline)")
        result = _run_single_split(
            df_test,
            model,
            features,
            initial_capital,
            sl_atr,
            tp_atr,
            confidence_threshold,
        )

    if result:
        _save_summary(result)
        _plot_equity_curve(result)
        logger.info("Backtest complete. Return: %.1f%%", result["total_return"])

    return result


def _run_single_split(
    df_test: pd.DataFrame,
    model,
    features: list,
    initial_capital: float,
    sl_atr: float,
    tp_atr: float,
    threshold: float,
) -> BacktestResult:
    """Standard single-split backtest (original logic)."""
    X_test = df_test[features]
    probs = model.predict_proba(X_test)[:, 1]
    df_test["prediction"] = (probs > threshold).astype(int)

    capital = initial_capital
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size_locked = 0.0
    trades = []
    equity_curve = [initial_capital]
    equity_dates = [pd.to_datetime(df_test.loc[0, "timestamp"])]
    btc_prices = [df_test.loc[0, "close"]]
    position_sizes = []
    trade_dates = []

    fee_pct = 0.0005
    max_risk_pct = 0.05

    for i in range(len(df_test) - 1):
        if capital <= 10:
            break

        current_price = float(df_test.loc[i, "close"])
        current_date = pd.to_datetime(df_test.loc[i, "timestamp"])
        atr = float(df_test.loc[i, "atr_14"])
        pred = int(df_test.loc[i, "prediction"])

        dist_stop = atr * sl_atr
        if dist_stop <= 0 or current_price <= 0:
            equity_curve.append(capital)
            equity_dates.append(current_date)
            btc_prices.append(current_price)
            continue

        stop_pct = dist_stop / current_price
        ideal_pos = (capital * max_risk_pct) / stop_pct
        pos_size = min(ideal_pos, capital * 5.0)
        pos_size = max(pos_size, capital * 1.0)
        leverage = pos_size / capital if capital > 0 else 0.0

        # Exit
        if position == 1:
            if current_price <= stop_loss:
                pnl = position_size_locked * ((stop_loss - entry_price) / entry_price)
                capital += pnl - position_size_locked * fee_pct
                trades.append(pnl - position_size_locked * fee_pct)
                position = 0
            elif current_price >= take_profit:
                pnl = position_size_locked * ((take_profit - entry_price) / entry_price)
                capital += pnl - position_size_locked * fee_pct
                trades.append(pnl - position_size_locked * fee_pct)
                position = 0

        # Entry
        if position == 0 and pred == 1:
            position = 1
            entry_price = current_price
            stop_loss = current_price - dist_stop
            take_profit = current_price + (atr * tp_atr)
            position_size_locked = pos_size
            position_sizes.append(leverage)
            trade_dates.append(current_date)
            capital -= position_size_locked * fee_pct

        equity_curve.append(capital)
        equity_dates.append(current_date)
        btc_prices.append(current_price)

    return _compute_metrics(
        equity_curve,
        equity_dates,
        btc_prices,
        position_sizes,
        trade_dates,
        trades,
        initial_capital,
        sl_atr,
        tp_atr,
        threshold,
        walkforward=False,
    )


def _run_walkforward(
    df: pd.DataFrame,
    initial_capital: float,
    sl_atr: float,
    tp_atr: float,
    threshold: float,
    n_splits: int,
) -> BacktestResult:
    """Walk-forward validation: train on expanding window, test on next chunk."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_equity = []
    all_btc = []
    all_dates = []
    all_trades = []
    all_pos_sizes = []
    all_trade_dates = []

    capital = initial_capital

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        logger.info(
            "Fold %d: train=%d rows, test=%d rows",
            fold + 1,
            len(train_df),
            len(test_df),
        )

        # Train model on this fold's training data
        from src.feature_engineering import prepare_features
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train, features, scaler = prepare_features(train_df, fit_scaler=True)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Get predictions on test set
        X_test = test_df[features]
        if scaler:
            X_test = scaler.transform(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        test_df["prediction"] = (probs > threshold).astype(int)

        # Simulate trading on this test fold (same logic as single split)
        pos = 0
        entry_p = 0.0
        sl = 0.0
        tp = 0.0
        pos_locked = 0.0
        fee = 0.0005

        for i in range(len(test_df) - 1):
            if capital <= 10:
                break

            cp = float(test_df.loc[i, "close"])
            dt = pd.to_datetime(test_df.loc[i, "timestamp"])
            atr = float(test_df.loc[i, "atr_14"])
            pred = int(test_df.loc[i, "prediction"])

            dist_sl = atr * sl_atr
            if dist_sl <= 0 or cp <= 0:
                all_equity.append(capital)
                all_dates.append(dt)
                all_btc.append(cp)
                continue

            stop_pct = dist_sl / cp
            ideal = (capital * 0.05) / stop_pct
            pos_size = min(ideal, capital * 5.0)
            pos_size = max(pos_size, capital * 1.0)
            lev = pos_size / capital if capital > 0 else 0.0

            if pos == 1:
                if cp <= sl:
                    pnl = pos_locked * ((sl - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0
                elif cp >= tp:
                    pnl = pos_locked * ((tp - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0

            if pos == 0 and pred == 1:
                pos = 1
                entry_p = cp
                sl = cp - dist_sl
                tp = cp + (atr * tp_atr)
                pos_locked = pos_size
                all_pos_sizes.append(lev)
                all_trade_dates.append(dt)
                capital -= pos_locked * fee

            all_equity.append(capital)
            all_dates.append(dt)
            all_btc.append(cp)

    return _compute_metrics(
        all_equity,
        all_dates,
        all_btc,
        all_pos_sizes,
        all_trade_dates,
        all_trades,
        initial_capital,
        sl_atr,
        tp_atr,
        threshold,
        walkforward=True,
        n_splits=n_splits,
    )


def _compute_metrics(
    equity_curve: list,
    equity_dates: list,
    btc_prices: list,
    position_sizes: list,
    trade_dates: list,
    trades: list,
    initial_capital: float,
    sl_atr: float,
    tp_atr: float,
    threshold: float,
    walkforward: bool,
    n_splits: Optional[int] = None,
) -> BacktestResult:
    """Compute summary statistics from equity curve."""
    final_cap = float(equity_curve[-1]) if equity_curve else initial_capital
    total_return = (final_cap - initial_capital) / initial_capital * 100
    avg_lev = float(np.mean(position_sizes)) if position_sizes else 0.0
    btc_start = btc_prices[0] if btc_prices else 0
    btc_end = btc_prices[-1] if btc_prices else 0
    bh_return = ((btc_end - btc_start) / btc_start * 100) if btc_start else 0

    total_trades = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades else 0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    best = float(max(trades)) if trades else 0.0
    worst = float(min(trades)) if trades else 0.0

    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks * 100
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    eq_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (
        float((np.mean(eq_ret) / np.std(eq_ret)) * np.sqrt(24 * 365))
        if eq_ret.size and np.std(eq_ret) > 0
        else 0.0
    )

    logger.info("=" * 40)
    logger.info("Threshold: %.2f | Walkforward: %s", threshold, walkforward)
    logger.info(
        "Period: %s to %s",
        equity_dates[0].strftime("%Y-%m-%d") if equity_dates else "N/A",
        equity_dates[-1].strftime("%Y-%m-%d") if equity_dates else "N/A",
    )
    logger.info("Final: $%.2f | Return: %+.1f%%", final_cap, total_return)
    logger.info("BH Return: %+.1f%% | Sharpe: %.2f", bh_return, sharpe)
    logger.info(
        "Trades: %d | Win Rate: %.1f%% | Max DD: %.1f%%", total_trades, win_rate, max_dd
    )

    result: BacktestResult = {
        "total_return": round(total_return, 2),
        "buy_hold_return": round(bh_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "avg_leverage": round(avg_lev, 2),
        "period_start": equity_dates[0].strftime("%Y-%m-%d") if equity_dates else "",
        "period_end": equity_dates[-1].strftime("%Y-%m-%d") if equity_dates else "",
        "final_capital": round(final_cap, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "best_trade": round(best, 2),
        "worst_trade": round(worst, 2),
        "walkforward": walkforward,
        "n_splits": n_splits if walkforward else 1,
    }

    return result


def _plot_equity_curve(result: BacktestResult):
    """Simple equity curve placeholder — full plotting is in plot_ml_metrics."""
    # Minimal plot for backtest endpoint
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0, 1], [100, result["final_capital"]], color="#3fb950")
    ax.set_title(
        f"Strategy: {result['total_return']:+.0f}% | Buy & Hold: {result['buy_hold_return']:+.0f}%"
    )
    ax.set_ylabel("Capital ($)")
    os.makedirs("models", exist_ok=True)
    path = f"models/dream_dynamic_{'wf' if result.get('walkforward') else 'single'}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    result["image"] = path


def _save_summary(result: BacktestResult):
    try:
        os.makedirs("models", exist_ok=True)
        path = "models/backtest_results.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", path)
    except Exception as e:
        logger.warning("Couldn't save summary: %s", e)


def run_backtest():
    """Used by Flask /api/run_backtest."""
    return run_backtest_dynamic(use_walkforward=False)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    wf = "--walkforward" in sys.argv
    run_backtest_dynamic(confidence_threshold=threshold, use_walkforward=wf)

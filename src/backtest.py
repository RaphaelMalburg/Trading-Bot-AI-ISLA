"""
Walk-forward backtesting with compounding equity simulation.

Two modes:
  - single: one 80/20 split (fast baseline)
  - walkforward: TimeSeriesSplit with N folds, retrain per fold
"""

import json
import logging
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS, add_indicators

logger = logging.getLogger(__name__)

RESULTS_PATH = "models/backtest_results.json"
CHART_PATH = "models/backtest_equity.png"


def run(
    df: pd.DataFrame,
    initial_capital: float = 100.0,
    sl_atr: float = 1.0,
    tp_atr: float = 2.0,
    confidence_threshold: float = 0.55,
    max_hold_candles: int = 12,
    use_walkforward: bool = True,
    n_splits: int = 5,
) -> dict:
    """
    Run backtest and save results + equity curve PNG.

    Returns a results dict with metrics.
    """
    df = add_indicators(df, sl_atr_mult=sl_atr, tp_atr_mult=tp_atr, max_hold_candles=max_hold_candles)
    df_clean = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.8)
    df_test = df_clean.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info("Backtest mode: %s | test rows: %d", "walkforward" if use_walkforward else "single", len(df_test))

    if use_walkforward:
        result = _walkforward(df_clean, initial_capital, sl_atr, tp_atr, confidence_threshold, n_splits)
    else:
        model_pkl = "models/rf_model.pkl"
        import joblib
        try:
            model = joblib.load(model_pkl)
            scaler = joblib.load("models/scaler.pkl")
        except Exception as e:
            logger.error("Could not load model for backtest: %s", e)
            return {}
        result = _simulate(df_test, model, scaler, initial_capital, sl_atr, tp_atr, confidence_threshold)

    os.makedirs("models", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "_equity_curve"}, f, indent=2)

    _plot(result, use_walkforward)
    logger.info("Backtest done: return=%.1f%%, sharpe=%.2f, win_rate=%.1f%%",
                result["total_return"], result["sharpe"], result["win_rate"])
    return result


def _simulate(df: pd.DataFrame, model, scaler, initial_capital, sl_atr, tp_atr, threshold) -> dict:
    X = df[FEATURE_COLS]
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]
    df = df.copy()
    df["pred"] = (probs > threshold).astype(int)

    capital = initial_capital
    pos = 0
    entry_p = stop = tp = pos_locked = 0.0
    trades, equity, dates, btc = [], [capital], [], [df.loc[0, "close"]]
    fee = 0.0005

    for i in range(len(df) - 1):
        if capital <= 10:
            break
        cp = float(df.loc[i, "close"])
        dt = pd.to_datetime(df.loc[i, "timestamp"]) if "timestamp" in df.columns else pd.Timestamp(i)
        atr = float(df.loc[i, "atr_14"])
        pred = int(df.loc[i, "pred"])

        dist_sl = atr * sl_atr
        if dist_sl <= 0 or cp <= 0:
            equity.append(capital)
            dates.append(dt)
            btc.append(cp)
            continue

        stop_pct = dist_sl / cp
        pos_size = min((capital * 0.05) / stop_pct, capital * 0.98)
        pos_size = max(pos_size, capital * 0.05)

        if pos == 1:
            if cp <= stop:
                pnl = pos_locked * ((stop - entry_p) / entry_p)
                capital += pnl - pos_locked * fee
                trades.append(pnl - pos_locked * fee)
                pos = 0
            elif cp >= tp:
                pnl = pos_locked * ((tp - entry_p) / entry_p)
                capital += pnl - pos_locked * fee
                trades.append(pnl - pos_locked * fee)
                pos = 0

        if pos == 0 and pred == 1:
            pos = 1
            entry_p = cp
            stop = cp - dist_sl
            tp = cp + atr * tp_atr
            pos_locked = pos_size
            capital -= pos_locked * fee

        equity.append(capital)
        dates.append(dt)
        btc.append(cp)

    return _metrics(equity, dates, btc, trades, initial_capital, sl_atr, tp_atr, threshold, walkforward=False)


def _walkforward(df: pd.DataFrame, initial_capital, sl_atr, tp_atr, threshold, n_splits) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    capital = initial_capital
    all_equity, all_dates, all_btc, all_trades = [capital], [], [], []
    fee = 0.0005

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info("Fold %d: train=%d, test=%d", fold + 1, len(train_df), len(test_df))

        X_tr = train_df[FEATURE_COLS]
        y_tr = train_df["target"].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
        model.fit(X_tr_s, y_tr)

        X_te = test_df[FEATURE_COLS]
        X_te_s = scaler.transform(X_te)
        probs = model.predict_proba(X_te_s)[:, 1]
        test_df["pred"] = (probs > threshold).astype(int)

        pos = 0
        entry_p = stop = tp_p = pos_locked = 0.0

        for i in range(len(test_df) - 1):
            if capital <= 10:
                break
            cp = float(test_df.loc[i, "close"])
            dt = pd.to_datetime(test_df.loc[i, "timestamp"]) if "timestamp" in test_df.columns else pd.Timestamp(i)
            atr = float(test_df.loc[i, "atr_14"])
            pred = int(test_df.loc[i, "pred"])

            dist_sl = atr * sl_atr
            if dist_sl <= 0 or cp <= 0:
                all_equity.append(capital)
                all_dates.append(dt)
                all_btc.append(cp)
                continue

            stop_pct = dist_sl / cp
            pos_size = min((capital * 0.05) / stop_pct, capital * 0.98)
            pos_size = max(pos_size, capital * 0.05)

            if pos == 1:
                if cp <= stop:
                    pnl = pos_locked * ((stop - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0
                elif cp >= tp_p:
                    pnl = pos_locked * ((tp_p - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0

            if pos == 0 and pred == 1:
                pos = 1
                entry_p = cp
                stop = cp - dist_sl
                tp_p = cp + atr * tp_atr
                pos_locked = pos_size
                capital -= pos_locked * fee

            all_equity.append(capital)
            all_dates.append(dt)
            all_btc.append(cp)

    return _metrics(all_equity, all_dates, all_btc, all_trades, initial_capital, sl_atr, tp_atr, threshold,
                    walkforward=True, n_splits=n_splits)


def _metrics(equity, dates, btc, trades, initial_capital, sl_atr, tp_atr, threshold,
             walkforward: bool, n_splits: int = 1) -> dict:
    final = float(equity[-1]) if equity else initial_capital
    total_return = (final - initial_capital) / initial_capital * 100
    bh_start = btc[0] if btc else 0
    bh_end = btc[-1] if btc else 0
    bh_return = ((bh_end - bh_start) / bh_start * 100) if bh_start else 0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    win_rate = len(wins) / n * 100 if n > 0 else 0

    eq_arr = np.array(equity)
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks * 100)) if len(eq_arr) else 0
    eq_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (
        float(np.mean(eq_ret) / np.std(eq_ret) * np.sqrt(6 * 365))
        if eq_ret.size and np.std(eq_ret) > 0 else 0.0
    )

    return {
        "total_return": round(total_return, 2),
        "buy_hold_return": round(bh_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": n,
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
        "final_capital": round(final, 2),
        "period_start": str(dates[0])[:10] if dates else "",
        "period_end": str(dates[-1])[:10] if dates else "",
        "walkforward": walkforward,
        "n_splits": n_splits,
        "_equity_curve": equity,
        "_btc_prices": btc,
        "_dates": [str(d)[:10] for d in dates],
    }


def _plot(result: dict, walkforward: bool):
    equity = result.get("_equity_curve", [])
    if not equity:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, color="#3fb950", linewidth=1.5, label=f"Strategy {result['total_return']:+.1f}%")
    ax.axhline(equity[0], color="#8b949e", linewidth=0.8, linestyle="--", label=f"BH {result['buy_hold_return']:+.1f}%")
    ax.set_title(f"Equity Curve {'(Walk-Forward)' if walkforward else '(Single Split)'}", color="white")
    ax.set_ylabel("Capital ($)", color="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    result["image"] = CHART_PATH

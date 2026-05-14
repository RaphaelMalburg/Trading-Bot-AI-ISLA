"""
6-month simulated performance showcase.
Runs a walk-forward backtest using the live bot's exact parameters and saves:
  - equity curve with dates
  - individual trade records
  - summary metrics
to models/showcase_results.json
"""
import json
import logging
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import get_settings
from src.data import FEATURE_COLS, add_indicators, fetch_bars

logger = logging.getLogger(__name__)

OUT_PATH     = "models/showcase_results.json"
LOOKBACK     = 180   # days
N_SPLITS     = 3     # folds — keeps training window realistic for 6 months
CAPITAL0     = 100.0
FEE          = 0.0005


def _profit_factor(trades):
    wins   = sum(t for t in trades if t > 0)
    losses = abs(sum(t for t in trades if t < 0))
    return round(wins / losses, 2) if losses > 0 else float("inf")


def run(settings):
    sl_atr   = settings.sl_atr_mult
    tp_atr   = settings.tp_atr_mult
    conf_thr = settings.confidence_threshold

    logger.info("Fetching %d days of %dH BTC/USD data...", LOOKBACK, settings.timeframe_hours)
    df = fetch_bars(
        symbol="BTC/USD",
        lookback_days=LOOKBACK,
        timeframe_hours=settings.timeframe_hours,
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    df = add_indicators(df, sl_atr_mult=sl_atr, tp_atr_mult=tp_atr,
                        max_hold_candles=settings.max_hold_candles)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    logger.info("Bars after indicators: %d", len(df))

    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    capital = CAPITAL0

    eq_curve   = []   # [{date, equity, btc_price}]
    trade_log  = []   # individual trade records
    pnl_series = []   # dollar pnl per trade (for metrics)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info("Fold %d: train=%d  test=%d", fold + 1, len(train_df), len(test_df))

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(train_df[FEATURE_COLS])
        X_te   = scaler.transform(test_df[FEATURE_COLS])

        model  = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, train_df["target"].values)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs > conf_thr).astype(int)

        pos      = 0
        entry_p  = stop = tp = locked = 0.0
        entry_dt = None

        for i in range(len(test_df) - 1):
            if capital <= 10:
                break
            row = test_df.iloc[i]
            cp  = float(row["close"])
            dt  = str(row["timestamp"])[:16]      # "YYYY-MM-DD HH:MM"
            atr = float(row["atr_14"])
            pred = int(preds[i])

            dist_sl = atr * sl_atr
            if dist_sl <= 0 or cp <= 0:
                eq_curve.append({"date": dt, "equity": round(capital, 4), "btc_price": cp})
                continue

            stop_pct = dist_sl / cp
            pos_size = min(
                max((capital * settings.max_risk_per_trade) / stop_pct, capital * 0.05),
                capital * 0.98,
            )

            # ── exits ──
            if pos == 1:
                hit_sl = cp <= stop
                hit_tp = cp >= tp
                if hit_sl or hit_tp:
                    exit_price  = stop if hit_sl else tp
                    reason      = "SL" if hit_sl else "TP"
                    pnl_pct     = (exit_price - entry_p) / entry_p * 100
                    pnl_dollar  = locked * pnl_pct / 100 - locked * FEE
                    capital    += pnl_dollar
                    pnl_series.append(pnl_dollar)
                    trade_log.append({
                        "entry_time":   entry_dt,
                        "exit_time":    dt,
                        "entry_price":  round(entry_p, 2),
                        "exit_price":   round(exit_price, 2),
                        "pnl_pct":      round(pnl_pct, 3),
                        "pnl_dollar":   round(pnl_dollar, 4),
                        "exit_reason":  reason,
                        "capital_after": round(capital, 4),
                    })
                    pos = 0

            # ── entry ──
            if pos == 0 and pred == 1:
                pos      = 1
                entry_p  = cp
                entry_dt = dt
                stop     = cp - dist_sl
                tp       = cp + atr * tp_atr
                locked   = pos_size
                capital -= locked * FEE

            eq_curve.append({"date": dt, "equity": round(capital, 4), "btc_price": cp})

    # ── metrics ──
    eq_arr   = np.array([p["equity"] for p in eq_curve])
    btc_arr  = np.array([p["btc_price"] for p in eq_curve])
    final    = float(eq_arr[-1])
    tr       = (final - CAPITAL0) / CAPITAL0 * 100
    bh_start = btc_arr[0]
    bh_ret   = (btc_arr[-1] - bh_start) / bh_start * 100

    peaks  = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks * 100))
    rets   = np.diff(eq_arr) / eq_arr[:-1]
    ann_factor = np.sqrt(settings.timeframe_hours * 365 * (24 / settings.timeframe_hours))
    sharpe = float(np.mean(rets) / np.std(rets) * ann_factor) if rets.size and np.std(rets) > 0 else 0.0

    wins   = [t for t in pnl_series if t > 0]
    losses = [t for t in pnl_series if t <= 0]
    n      = len(pnl_series)

    # Normalize equity curve so B&H starts at 100 for fair comparison
    bh_norm = [{"date": p["date"], "bh_equity": round(CAPITAL0 * p["btc_price"] / bh_start, 4)}
               for p in eq_curve]
    for i, p in enumerate(eq_curve):
        p["bh_equity"] = bh_norm[i]["bh_equity"]

    result = {
        "generated_at":    pd.Timestamp.now().isoformat()[:19],
        "lookback_days":   LOOKBACK,
        "timeframe_hours": settings.timeframe_hours,
        "initial_capital": CAPITAL0,
        "final_capital":   round(final, 2),
        "total_return":    round(tr, 2),
        "buy_hold_return": round(bh_ret, 2),
        "outperformance":  round(tr - bh_ret, 2),
        "sharpe":          round(sharpe, 2),
        "max_drawdown":    round(max_dd, 2),
        "win_rate":        round(len(wins) / n * 100, 1) if n else 0,
        "total_trades":    n,
        "avg_win":         round(float(np.mean(wins)),   2) if wins   else 0,
        "avg_loss":        round(float(np.mean(losses)), 2) if losses else 0,
        "profit_factor":   _profit_factor(pnl_series),
        "equity_curve":    eq_curve,
        "trades":          trade_log,
    }

    os.makedirs("models", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)
    logger.info("Showcase saved → %s  (return=%.2f%%  trades=%d)", OUT_PATH, tr, n)
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    settings = get_settings()
    run(settings)


if __name__ == "__main__":
    main()

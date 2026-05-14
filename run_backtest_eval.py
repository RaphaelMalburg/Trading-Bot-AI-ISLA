"""
Run backtest with model evaluation (confusion matrix, accuracy, etc.)
and equity curve visualization.
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
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLS, add_indicators, fetch_bars
from src.config import get_settings

logger = logging.getLogger(__name__)

RESULTS_PATH = "models/backtest_results_eval.json"
EQUITY_CHART_PATH = "models/backtest_equity_eval.png"
CONFUSION_MATRIX_PATH = "models/confusion_matrix_eval.png"


def fetch_data(lookback_days=365):
    """Fetch BTC/USD 4H bars from Alpaca."""
    settings = get_settings()
    logger.info(f"Fetching {lookback_days} days of 4H BTC/USD data...")
    df = fetch_bars(
        symbol="BTC/USD",
        lookback_days=lookback_days,
        timeframe_hours=4,
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    return df


def run_walkforward_with_eval(df, n_splits=5):
    """
    Run walkforward backtest and collect predictions for evaluation.
    Returns dict with equity curve metrics and lists of all predictions and actuals.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    capital = 100.0  # initial capital
    fee = 0.0005
    all_equity = [capital]
    all_dates = []
    all_btc = []
    all_trades = []
    all_preds = []
    all_actuals = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info(
            "Fold %d: train=%d, test=%d", fold + 1, len(train_df), len(test_df)
        )

        # Features and target
        X_tr = train_df[FEATURE_COLS]
        y_tr = train_df["target"].values
        X_te = test_df[FEATURE_COLS]
        y_te = test_df["target"].values

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train model
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)

        # Predict
        probs = model.predict_proba(X_te_s)[:, 1]
        preds = (probs > 0.55).astype(int)  # using threshold from config
        all_preds.extend(preds)
        all_actuals.extend(y_te)

        # Store test close prices for buy & hold calculation
        test_close = test_df["close"].values

        # Simulate trading on this fold
        pos = 0
        entry_p = stop = tp = pos_locked = 0.0

        for i in range(len(test_df) - 1):
            if capital <= 10:
                break
            cp = float(test_df.iloc[i, :]["close"])
            dt = pd.to_datetime(test_df.iloc[i, :]["timestamp"])
            atr = float(test_df.iloc[i, :]["atr_14"])
            pred = int(preds[i])

            dist_sl = atr * 1.0  # sl_atr from config
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
                elif cp >= tp:
                    pnl = pos_locked * ((tp - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0

            if pos == 0 and pred == 1:
                pos = 1
                entry_p = cp
                stop = cp - dist_sl
                tp = cp + atr * 2.0  # tp_atr from config
                pos_locked = pos_size
                capital -= pos_locked * fee

            all_equity.append(capital)
            all_dates.append(dt)
            all_btc.append(cp)

    # Final metrics from equity curve
    final = float(all_equity[-1]) if all_equity else 100.0
    total_return = (final - 100.0) / 100.0 * 100
    bh_start = all_btc[0] if all_btc else 0
    bh_end = all_btc[-1] if all_btc else 0
    bh_return = ((bh_end - bh_start) / bh_start * 100) if bh_start else 0

    wins = [t for t in all_trades if t > 0]
    losses = [t for t in all_trades if t <= 0]
    n = len(all_trades)
    win_rate = len(wins) / n * 100 if n > 0 else 0

    eq_arr = np.array(all_equity)
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks * 100)) if len(eq_arr) else 0
    eq_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (
        float(np.mean(eq_ret) / np.std(eq_ret) * np.sqrt(6 * 365))
        if eq_ret.size and np.std(eq_ret) > 0 else 0.0
    )

    # Evaluation metrics
    accuracy = accuracy_score(all_actuals, all_preds)
    cm = confusion_matrix(all_actuals, all_preds)
    report = classification_report(all_actuals, all_preds, output_dict=True)

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
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "_equity_curve": all_equity,
        "_dates": [str(d)[:10] for d in all_dates],
        "_btc_prices": [float(p) for p in all_btc],
    }


def plot_equity(result):
    """Plot equity curve and buy & hold."""
    equity = result.get("_equity_curve", [])
    if not equity:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, color="#3fb950", linewidth=1.5, label=f"Strategy {result['total_return']:+.1f}%")
    ax.axhline(equity[0], color="#8b949e", linewidth=0.8, linestyle="--", label=f"BH {result['buy_hold_return']:+.1f}%")
    ax.set_title(f"Equity Curve (Walk-Forward)", color="white")
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
    plt.savefig(EQUITY_CHART_PATH, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def plot_confusion_matrix(cm):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/backtest_eval.log", encoding="utf-8"),
        ],
    )
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Fetch data
    df = fetch_data(lookback_days=365)
    df = add_indicators(df, sl_atr_mult=1.0, tp_atr_mult=2.0, max_hold_candles=12)
    df_clean = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    logger.info(f"Total bars after indicators: {len(df_clean)}")

    # Run walkforward backtest with evaluation
    result = run_walkforward_with_eval(df_clean, n_splits=5)

    # Save results (without internal arrays)
    results_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_to_save, f, indent=2)

    # Plot equity curve
    plot_equity(result)

    # Plot confusion matrix
    cm = np.array(result["confusion_matrix"])
    plot_confusion_matrix(cm)

    # Log summary
    logger.info("Backtest evaluation completed.")
    logger.info(f"Total Return: {result['total_return']}%")
    logger.info(f"Buy & Hold Return: {result['buy_hold_return']}%")
    logger.info(f"Sharpe Ratio: {result['sharpe']}")
    logger.info(f"Max Drawdown: {result['max_drawdown']}%")
    logger.info(f"Win Rate: {result['win_rate']}%")
    logger.info(f"Total Trades: {result['total_trades']}")
    logger.info(f"Accuracy: {result['accuracy']}")
    logger.info(f"Results saved to {RESULTS_PATH}")
    logger.info(f"Equity curve saved to {EQUITY_CHART_PATH}")
    logger.info(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
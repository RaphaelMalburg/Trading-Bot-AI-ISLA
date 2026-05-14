"""
Kelly Criterion vs Fixed-Fraction position sizing comparison.

Runs the same walk-forward backtest 4 ways:
  fixed_5pct   — always risk 5% of capital (current bot behaviour)
  full_kelly   — f* = (p*b - q) / b  using model probability per trade
  half_kelly   — 0.5 × full_kelly
  quarter_kelly— 0.25 × full_kelly

Prints a comparison table and saves an equity-curve plot.
"""
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

from src.config import get_settings
from src.data import FEATURE_COLS, add_indicators, fetch_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SL_ATR   = 1.0
TP_ATR   = 2.0
RR       = TP_ATR / SL_ATR          # reward:risk = 2.0
CONF_THR = 0.55
FEE      = 0.0005
CAPITAL0 = 100.0
N_SPLITS = 5
MAX_FRAC = 0.50   # hard cap on any Kelly fraction (avoid overbetting)


def _metrics(equity, trades, btc_prices):
    eq = np.array(equity)
    final = float(eq[-1])
    total_ret = (final - CAPITAL0) / CAPITAL0 * 100
    bh_ret = ((btc_prices[-1] - btc_prices[0]) / btc_prices[0] * 100) if btc_prices else 0
    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n      = len(trades)
    win_rate = len(wins) / n * 100 if n else 0
    peaks  = np.maximum.accumulate(eq)
    max_dd = float(np.max((peaks - eq) / peaks * 100)) if len(eq) else 0
    rets   = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(6 * 365)) if rets.size and np.std(rets) > 0 else 0
    return {
        "total_return":   round(total_ret, 2),
        "buy_hold_return": round(bh_ret, 2),
        "sharpe":          round(sharpe, 2),
        "max_drawdown":    round(max_dd, 2),
        "win_rate":        round(win_rate, 2),
        "total_trades":    n,
        "avg_win":         round(float(np.mean(wins)),   2) if wins   else 0,
        "avg_loss":        round(float(np.mean(losses)), 2) if losses else 0,
        "final_capital":   round(final, 2),
    }


def _kelly_fraction(prob, rr=RR):
    """Full Kelly: f* = (p*b - q) / b  where b = reward:risk ratio."""
    return max(0.0, (prob * rr - (1 - prob)) / rr)


def run_comparison(df):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    strategies = {
        "fixed_5pct":    {"equity": [CAPITAL0], "trades": [], "btc": [], "cap": CAPITAL0},
        "full_kelly":    {"equity": [CAPITAL0], "trades": [], "btc": [], "cap": CAPITAL0},
        "half_kelly":    {"equity": [CAPITAL0], "trades": [], "btc": [], "cap": CAPITAL0},
        "quarter_kelly": {"equity": [CAPITAL0], "trades": [], "btc": [], "cap": CAPITAL0},
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info("Fold %d: train=%d  test=%d", fold + 1, len(train_df), len(test_df))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[FEATURE_COLS])
        X_te = scaler.transform(test_df[FEATURE_COLS])

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, train_df["target"].values)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs > CONF_THR).astype(int)

        # Independent position state per strategy
        state = {
            name: {"pos": 0, "entry_p": 0.0, "stop": 0.0, "tp": 0.0, "locked": 0.0}
            for name in strategies
        }

        for i in range(len(test_df) - 1):
            row  = test_df.iloc[i]
            cp   = float(row["close"])
            atr  = float(row["atr_14"])
            prob = float(probs[i])
            pred = int(preds[i])

            dist_sl = atr * SL_ATR
            if dist_sl <= 0 or cp <= 0:
                for s in strategies.values():
                    s["equity"].append(s["cap"])
                    s["btc"].append(cp)
                continue

            stop_pct = dist_sl / cp

            # --- position size per strategy ---
            kelly_f  = _kelly_fraction(prob)
            sizes = {
                "fixed_5pct":    min(max((strategies["fixed_5pct"]["cap"]    * 0.05) / stop_pct,
                                         strategies["fixed_5pct"]["cap"] * 0.05),
                                     strategies["fixed_5pct"]["cap"]    * 0.98),
                "full_kelly":    strategies["full_kelly"]["cap"]    * min(kelly_f,             MAX_FRAC),
                "half_kelly":    strategies["half_kelly"]["cap"]    * min(kelly_f * 0.5,       MAX_FRAC),
                "quarter_kelly": strategies["quarter_kelly"]["cap"] * min(kelly_f * 0.25,      MAX_FRAC),
            }

            for name, s_data in strategies.items():
                cap  = s_data["cap"]
                st   = state[name]
                psize = sizes[name]

                if cap <= 10:
                    s_data["equity"].append(cap)
                    s_data["btc"].append(cp)
                    continue

                # Check exits
                if st["pos"] == 1:
                    if cp <= st["stop"]:
                        pnl = st["locked"] * ((st["stop"] - st["entry_p"]) / st["entry_p"])
                        cap += pnl - st["locked"] * FEE
                        s_data["trades"].append(pnl - st["locked"] * FEE)
                        st["pos"] = 0
                    elif cp >= st["tp"]:
                        pnl = st["locked"] * ((st["tp"] - st["entry_p"]) / st["entry_p"])
                        cap += pnl - st["locked"] * FEE
                        s_data["trades"].append(pnl - st["locked"] * FEE)
                        st["pos"] = 0

                # Check entry (Kelly 0 → skip; fixed always enters if pred==1)
                if st["pos"] == 0 and pred == 1 and psize > 0:
                    st["pos"]     = 1
                    st["entry_p"] = cp
                    st["stop"]    = cp - dist_sl
                    st["tp"]      = cp + atr * TP_ATR
                    st["locked"]  = psize
                    cap -= psize * FEE

                s_data["cap"] = cap
                s_data["equity"].append(cap)
                s_data["btc"].append(cp)

    return {
        name: _metrics(s["equity"], s["trades"], s["btc"])
        | {"_equity": s["equity"]}
        for name, s in strategies.items()
    }


def print_table(results):
    cols = ["total_return", "buy_hold_return", "sharpe", "max_drawdown",
            "win_rate", "total_trades", "avg_win", "avg_loss", "final_capital"]
    header = f"{'Strategy':<16}" + "".join(f"{c:>16}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        row = f"{name:<16}" + "".join(f"{str(r[c]):>16}" for c in cols)
        print(row)
    print("=" * len(header) + "\n")


def plot_curves(results):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {
        "fixed_5pct":    "#58a6ff",
        "full_kelly":    "#3fb950",
        "half_kelly":    "#f0883e",
        "quarter_kelly": "#d2a520",
    }
    labels = {
        "fixed_5pct":    "Fixed 5% (current)",
        "full_kelly":    "Full Kelly",
        "half_kelly":    "Half Kelly",
        "quarter_kelly": "Quarter Kelly",
    }
    for name, r in results.items():
        eq = r["_equity"]
        ax.plot(eq, color=colors[name], linewidth=1.8,
                label=f"{labels[name]}  {r['total_return']:+.1f}%  Sharpe {r['sharpe']}")

    ax.axhline(CAPITAL0, color="#8b949e", linewidth=0.8, linestyle="--", label="Starting capital")
    ax.set_title("Kelly vs Fixed Fraction — Walk-Forward Backtest", color="white", fontsize=13)
    ax.set_ylabel("Capital ($)", color="white")
    ax.set_xlabel("Bars", color="white")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=10)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = "models/kelly_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info("Chart saved → %s", out)


def main():
    os.makedirs("models", exist_ok=True)
    settings = get_settings()

    logger.info("Fetching 365 days of 1H BTC/USD data...")
    df = fetch_bars(
        symbol="BTC/USD",
        lookback_days=365,
        timeframe_hours=settings.timeframe_hours,
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    df = add_indicators(df, sl_atr_mult=SL_ATR, tp_atr_mult=TP_ATR,
                        max_hold_candles=settings.max_hold_candles)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    logger.info("Bars after indicators: %d", len(df))

    results = run_comparison(df)
    print_table(results)
    plot_curves(results)

    # Extra: show Kelly fractions at sample confidence levels
    print("Kelly fraction at various model confidence levels (RR=2.0):")
    print(f"  {'Confidence':>12}  {'Full Kelly':>12}  {'Half Kelly':>12}  {'Quarter':>10}  {'Enters?':>8}")
    for p in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        f = _kelly_fraction(p)
        enters = "YES" if f > 0 else "NO"
        print(f"  {p:>12.0%}  {f:>12.1%}  {f*0.5:>12.1%}  {f*0.25:>10.1%}  {enters:>8}")
    print()
    break_even_p = 1 / (1 + RR)
    print(f"Break-even precision for RR={RR}: {break_even_p:.1%}  "
          f"(Kelly is positive only above this)")


if __name__ == "__main__":
    main()

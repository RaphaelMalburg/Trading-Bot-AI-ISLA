"""
Low timeframe backtest with feature importance analysis.
This script runs a backtest on 1H timeframe data and analyzes which features
are most important for the model's predictions.

Features analyzed:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price action features
- Volume features
- Volatility measures

The script outputs:
1. Equity curve comparison (strategy vs buy & hold)
2. Confusion matrix for classification performance
3. Feature importance plot showing top predictors
4. Detailed metrics JSON file
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

# Paths for saving results
RESULTS_PATH = "models/backtest_results_lowtf.json"
EQUITY_CHART_PATH = "models/backtest_equity_lowtf.png"
CONFUSION_MATRIX_PATH = "models/confusion_matrix_lowtf.png"
FEATURE_IMPORTANCE_PATH = "models/feature_importance_lowtf.png"


def fetch_low_tf_data(lookback_days=90):
    """
    Fetch BTC/USD data with lower timeframe (1H instead of 4H).
    Using 90 days of 1H data gives us roughly the same number of bars
    as 365 days of 4H data, but with higher frequency signals.
    """
    settings = get_settings()
    logger.info(f"Fetching {lookback_days} days of 1H BTC/USD data...")
    df = fetch_bars(
        symbol="BTC/USD",
        lookback_days=lookback_days,
        timeframe_hours=1,  # Changed from 4 to 1 hour
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    return df


def run_low_tf_backtest_with_features(df, n_splits=5):
    """
    Run walkforward backtest on low timeframe data and collect:
    - Equity curve metrics
    - All predictions and actuals for evaluation
    - Feature importances from each fold (averaged)
    
    Returns dict with all results including feature importance.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    capital = 100.0  # initial capital
    fee = 0.0005
    
    # Track everything across folds
    all_equity = [capital]
    all_dates = []
    all_btc = []
    all_trades = []
    all_preds = []
    all_actuals = []
    
    # For feature importance - collect from each fold
    fold_importances = []
    
    logger.info("Starting walkforward backtest with %d splits...", n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info(
            "Fold %d: train=%d bars, test=%d bars", 
            fold + 1, len(train_df), len(test_df)
        )

        # Prepare features and target
        X_tr = train_df[FEATURE_COLS]
        y_tr = train_df["target"].values
        X_te = test_df[FEATURE_COLS]
        y_te = test_df["target"].values

        # Scale features (fit on train, transform on both)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42 + fold,  # Different seed per fold for variety
            n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)

        # Store feature importances from this fold
        fold_importances.append(model.feature_importances_)

        # Predict on test set
        probs = model.predict_proba(X_te_s)[:, 1]
        preds = (probs > 0.55).astype(int)  # threshold from config
        all_preds.extend(preds)
        all_actuals.extend(y_te)

        # Simulate trading on this fold
        pos = 0
        entry_p = stop = tp = pos_locked = 0.0

        for i in range(len(test_df) - 1):
            if capital <= 10:  # Stop if capital too low
                break
                
            # Current price and time
            cp = float(test_df.iloc[i, :]["close"])
            dt = pd.to_datetime(test_df.iloc[i, :]["timestamp"])
            atr = float(test_df.iloc[i, :]["atr_14"])
            pred = int(preds[i])

            # Calculate stop loss distance
            dist_sl = atr * 1.0  # sl_atr from config
            if dist_sl <= 0 or cp <= 0:
                # No valid trade, just track equity
                all_equity.append(capital)
                all_dates.append(dt)
                all_btc.append(cp)
                continue

            # Position sizing based on risk management
            stop_pct = dist_sl / cp
            pos_size = min((capital * 0.05) / stop_pct, capital * 0.98)  # Max 5% risk per trade
            pos_size = max(pos_size, capital * 0.05)  # Minimum 5% of capital

            # Manage existing position
            if pos == 1:
                if cp <= stop:  # Stop loss hit
                    pnl = pos_locked * ((stop - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0
                elif cp >= tp:  # Take profit hit
                    pnl = pos_locked * ((tp - entry_p) / entry_p)
                    capital += pnl - pos_locked * fee
                    all_trades.append(pnl - pos_locked * fee)
                    pos = 0

            # Look for new entry signal
            if pos == 0 and pred == 1:  # Buy signal
                pos = 1
                entry_p = cp
                stop = cp - dist_sl
                tp = cp + atr * 2.0  # tp_atr from config
                pos_locked = pos_size
                capital -= pos_locked * fee  # Pay entry fee

            # Track portfolio value
            all_equity.append(capital)
            all_dates.append(dt)
            all_btc.append(cp)

    # Calculate final performance metrics from equity curve
    final = float(all_equity[-1]) if all_equity else 100.0
    total_return = (final - 100.0) / 100.0 * 100
    bh_start = all_btc[0] if all_btc else 0
    bh_end = all_btc[-1] if all_btc else 0
    bh_return = ((bh_end - bh_start) / bh_start * 100) if bh_start else 0

    # Trading statistics
    wins = [t for t in all_trades if t > 0]
    losses = [t for t in all_trades if t <= 0]
    n = len(all_trades)
    win_rate = len(wins) / n * 100 if n > 0 else 0

    # Risk metrics
    eq_arr = np.array(all_equity)
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks * 100)) if len(eq_arr) else 0
    eq_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (
        float(np.mean(eq_ret) / np.std(eq_ret) * np.sqrt(6 * 365))
        if eq_ret.size and np.std(eq_ret) > 0 else 0.0
    )

    # Average feature importance across all folds
    avg_importances = np.mean(fold_importances, axis=0) if fold_importances else np.zeros(len(FEATURE_COLS))
    
    # Create feature importance dictionary
    feature_importance_dict = dict(zip(FEATURE_COLS, avg_importances))
    # Sort by importance descending
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Evaluation metrics
    accuracy = accuracy_score(all_actuals, all_preds)
    cm = confusion_matrix(all_actuals, all_preds)
    report = classification_report(all_actuals, all_preds, output_dict=True)

    return {
        # Performance metrics
        "total_return": round(total_return, 2),
        "buy_hold_return": round(bh_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": n,
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
        "final_capital": round(final, 2),
        
        # Model evaluation
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        
        # Feature importance
        "feature_importance": feature_importance_dict,
        "sorted_feature_importance": sorted_features,
        
        # Internal data for plotting (not saved to JSON)
        "_equity_curve": all_equity,
        "_dates": [str(d)[:10] for d in all_dates],
        "_btc_prices": [float(p) for p in all_btc],
    }


def plot_equity(result):
    """Plot equity curve showing strategy vs buy & hold performance."""
    equity = result.get("_equity_curve", [])
    if not equity:
        logger.warning("No equity curve data to plot")
        return
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot strategy equity curve
    ax.plot(equity, color="#3fb950", linewidth=2, label=f"Strategy ({result['total_return']:+.1f}%)")
    
    # Plot buy & hold reference line
    ax.axhline(equity[0], color="#8b949e", linewidth=1.5, linestyle="--", 
               label=f"Buy & Hold ({result['buy_hold_return']:+.1f}%)")
    
    # Styling
    ax.set_title("Equity Curve - 1H Timeframe Backtest", fontsize=14, fontweight='bold', color="white")
    ax.set_ylabel("Capital ($)", fontsize=12, color="white")
    ax.set_xlabel("Time Period", fontsize=12, color="white")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=10)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color="#30363d")
    
    plt.tight_layout()
    plt.savefig(EQUITY_CHART_PATH, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info("Equity curve saved to %s", EQUITY_CHART_PATH)


def plot_confusion_matrix(cm):
    """Plot confusion matrix as a formatted heatmap with trading performance explanations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={'label': 'Count'},
        ax=ax,
        xticklabels=["Predicted: No Trade", "Predicted: Trade"],
        yticklabels=["Actual: No Trade", "Actual: Trade"],
    )
    
    # Add explanatory text about what each quadrant means for trading
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    accuracy = (tn + tp) / total if total > 0 else 0
    precision_class1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_class0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Add explanation box
    explanation = (
        f"CONFUSION MATRIX EXPLANATION:\n\n"
        f"Top Left (TN={tn}): Correctly predicted NO TRADE\n"
        f"  → Avoided {tn} losing/flat trades\n\n"
        f"Top Right (FP={fp}): FALSE BUY SIGNALS\n"
        f"  → Would have entered {fp} losing trades\n\n"
        f"Bottom Left (FN={fn}): MISSED BUY SIGNALS\n"
        f"  → Missed {fn} profitable opportunities\n\n"
        f"Bottom Right (TP={tp}): CORRECT BUY SIGNALS\n"
        f"  → Correctly caught {tp} winning trades\n\n"
        f"PERFORMANCE METRICS:\n"
        f"• Accuracy: {accuracy:.1%}\n"
        f"• Trade Precision: {precision_class1:.1%} (when model says BUY)\n"
        f"• Trade Recall: {recall_class1:.1%} (of actual winning trades caught)\n"
        f"• Avoidance Precision: {precision_class0:.1%} (when model says NO TRADE)\n"
        f"• Avoidance Recall: {recall_class0:.1%} (of actual no-trade periods avoided)"
    )
    
    # Add text box to the right of the confusion matrix
    props = dict(boxstyle='round', facecolor='#0d1117', alpha=0.8, edgecolor='#30363d')
    ax.text(1.05, 0.5, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=props, color='white', fontfamily='monospace')
    
    ax.set_title("Confusion Matrix - Model Predictions\n(With Trading Performance Explanation)", 
                 fontsize=14, fontweight='bold', color="white", pad=20)
    ax.set_ylabel("Actual Label", fontsize=12, color="white")
    ax.set_xlabel("Predicted Label", fontsize=12, color="white")
    ax.tick_params(colors="#8b949e")
    
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", CONFUSION_MATRIX_PATH)


def plot_feature_importance(feature_dict, top_n=15):
    """
    Plot feature importance as a horizontal bar chart.
    Shows the top N most important features for the model.
    """
    # Sort features by importance
    sorted_items = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_items[:top_n]
    
    # Prepare data for plotting
    features = [item[0] for item in top_features]
    importance = [item[1] for item in top_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color="#3fb950", alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()  # Most important on top
    ax.set_xlabel("Feature Importance Score", fontsize=12, color="white")
    ax.set_title(f"Top {top_n} Most Important Features - Random Forest Model", 
                 fontsize=14, fontweight='bold', color="white")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color="#30363d", axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(importance)*0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9, color="white")
    
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", FEATURE_IMPORTANCE_PATH)


def main():
    """Main function to run the low timeframe backtest with feature analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/backtest_lowtf.log", encoding="utf-8"),
        ],
    )
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("LOW TIMEFRAME BACKTEST WITH FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)
    
    # Fetch lower timeframe data (1H instead of 4H)
    df = fetch_low_tf_data(lookback_days=90)  # 90 days of 1H data
    logger.info(f"Fetched {len(df)} raw bars")
    
    # Add technical indicators and target variable
    df = add_indicators(df, sl_atr_mult=1.0, tp_atr_mult=2.0, max_hold_candles=12)
    df_clean = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    logger.info(f"After indicators and cleaning: {len(df_clean)} bars")
    
    if len(df_clean) < 100:
        logger.error("Insufficient data after cleaning. Need at least 100 bars.")
        return
    
    # Run the backtest with feature importance analysis
    result = run_low_tf_backtest_with_features(df_clean, n_splits=5)
    
    # Save results (excluding internal plotting data)
    results_to_save = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)
    
    # Generate all plots
    logger.info("Generating visualizations...")
    plot_equity(result)
    plot_confusion_matrix(np.array(result["confusion_matrix"]))
    plot_feature_importance(result["feature_importance"], top_n=15)
    
    # Print summary to console
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Timeframe: 1H candles")
    logger.info(f"Period: {result.get('_dates', [''])[0]} to {result.get('_dates', [''])[-1] if result.get('_dates') else ''}")
    logger.info(f"Total Return: {result['total_return']}%")
    logger.info(f"Buy & Hold Return: {result['buy_hold_return']}%")
    logger.info(f"Outperformance: {result['total_return'] - result['buy_hold_return']:+.1f}%")
    logger.info(f"Sharpe Ratio: {result['sharpe']}")
    logger.info(f"Max Drawdown: {result['max_drawdown']}%")
    logger.info(f"Win Rate: {result['win_rate']}%")
    logger.info(f"Total Trades: {result['total_trades']}")
    logger.info(f"Model Accuracy: {result['accuracy']}")
    logger.info("")
    logger.info("TOP 5 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(result["sorted_feature_importance"][:5], 1):
        logger.info(f"  {i}. {feature}: {importance:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
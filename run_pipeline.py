"""
CLI pipeline runner: fetch data → train model → run backtest → (optional) start bot.

Usage:
    python run_pipeline.py            # train + backtest only
    python run_pipeline.py --start    # train + backtest + start bot
    python run_pipeline.py --csv data/btc_usd_hourly.csv  # use local CSV instead of API
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ISLA Bot Pipeline")
    parser.add_argument("--csv", default="", help="Path to CSV for training (optional)")
    parser.add_argument("--timeframe", type=int, default=None, help="Candle timeframe in hours (1 or 4). Overrides TIMEFRAME_HOURS env var.")
    parser.add_argument("--start", action="store_true", help="Start the bot after pipeline")
    parser.add_argument("--backtest-only", action="store_true", help="Skip training, run backtest with existing model")
    parser.add_argument("--walkforward", action="store_true", default=True, help="Use walk-forward backtest (default: on)")
    args = parser.parse_args()

    from src.config import get_settings
    settings_obj = get_settings()
    tf_hours = args.timeframe if args.timeframe else settings_obj.timeframe_hours
    logger.info("Pipeline using %dH timeframe", tf_hours)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    from src.data import fetch_bars, load_csv

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    if args.csv and os.path.exists(args.csv):
        logger.info("Loading data from CSV: %s", args.csv)
        resample = tf_hours == 4
        df = load_csv(args.csv, resample_to_4h=resample)
    else:
        logger.info("Fetching %dH BTC/USD data from Alpaca API...", tf_hours)
        lookback = 1200 if tf_hours == 4 else 365
        df = fetch_bars(
            symbol=settings_obj.symbol,
            lookback_days=lookback,
            timeframe_hours=tf_hours,
            api_key=settings_obj.alpaca_api_key,
            secret_key=settings_obj.alpaca_secret_key,
        )
        out_path = f"data/btc_{tf_hours}h.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved to %s", out_path)

    logger.info("Data: %d rows from %s to %s", len(df),
                str(df.iloc[0]["timestamp"])[:10], str(df.iloc[-1]["timestamp"])[:10])

    # ── Step 2: Train (unless backtest-only) ─────────────────────────────────
    if not args.backtest_only:
        logger.info("Training Random Forest on %dH data...", tf_hours)
        from src.model import train
        metrics = train(
            df,
            sl_atr_mult=settings_obj.sl_atr_mult,
            tp_atr_mult=settings_obj.tp_atr_mult,
            max_hold_candles=settings_obj.max_hold_candles,
        )
        logger.info("Training complete:")
        logger.info("  Accuracy:  %.4f", metrics["accuracy"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  F1:        %.4f", metrics["f1"])
        logger.info("  Train/Test: %d / %d samples", metrics["train_samples"], metrics["test_samples"])

    # ── Step 3: Backtest ──────────────────────────────────────────────────────
    logger.info("Running backtest (walkforward=%s)...", args.walkforward)
    from src.backtest import run as run_backtest
    results = run_backtest(
        df,
        sl_atr=settings_obj.sl_atr_mult,
        tp_atr=settings_obj.tp_atr_mult,
        confidence_threshold=settings_obj.confidence_threshold,
        max_hold_candles=settings_obj.max_hold_candles,
        use_walkforward=args.walkforward,
    )
    logger.info("Backtest results:")
    logger.info("  Total return:   %+.2f%%", results["total_return"])
    logger.info("  Buy & Hold:     %+.2f%%", results["buy_hold_return"])
    logger.info("  Sharpe ratio:   %.2f", results["sharpe"])
    logger.info("  Max drawdown:   %.2f%%", results["max_drawdown"])
    logger.info("  Win rate:       %.1f%%", results["win_rate"])
    logger.info("  Total trades:   %d", results["total_trades"])

    # ── Step 4: Start bot (optional) ─────────────────────────────────────────
    if args.start:
        logger.info("Starting trading bot...")
        import subprocess
        subprocess.run([sys.executable, "main.py"])


if __name__ == "__main__":
    main()

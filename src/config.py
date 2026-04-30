"""
Simplified config — reads from environment with defaults.
Works with any Python version without Pydantic if needed.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # API Keys
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    openrouter_api_key: Optional[str] = None

    # Model paths
    model_path: str = "models/rf_model.pkl"
    features_path: str = "models/model_features.pkl"
    scaler_path: str = "models/scaler.pkl"

    # Trading parameters
    confidence_threshold: float = 0.55
    sentiment_floor: float = -0.5
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 2.0
    sl_limit_slippage: float = 0.005
    max_risk_per_trade: float = 0.05
    daily_loss_limit_pct: float = 0.10

    # Execution
    fill_poll_attempts: int = 10
    fill_poll_interval_s: float = 1.0
    tp_watchdog_interval_s: int = 60

    # Risk limits
    max_total_exposure: float = 1.0  # Max 100% of capital in total positions
    max_concurrent_trades: int = 1

    # Drift detection
    drift_check_window: int = 50
    drift_accuracy_threshold: float = 0.50

    # Symbols
    symbol_btc: str = "BTC/USD"
    symbol_eth: str = "ETH/USD"
    timeframe: str = "1h"

    # LLM
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "google/gemini-2.5-flash-lite"

    # Database
    db_path: str = "data/trading_bot.db"

    # Flask
    flask_port: int = 5000
    flask_debug: bool = False

    @property
    def sentiment_enabled(self) -> bool:
        return self.openrouter_api_key is not None


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Build Settings from environment, with validation."""
    global _settings
    if _settings is None:
        # Required
        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        # Optional with defaults
        _settings = Settings(
            alpaca_api_key=api_key,
            alpaca_secret_key=secret,
            alpaca_base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.55")),
            sentiment_floor=float(os.getenv("SENTIMENT_FLOOR", "-0.5")),
            sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.0")),
            tp_atr_mult=float(os.getenv("TP_ATR_MULT", "2.0")),
            sl_limit_slippage=float(os.getenv("SL_LIMIT_SLIPPAGE", "0.005")),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.05")),
            daily_loss_limit_pct=float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.10")),
            fill_poll_attempts=int(os.getenv("FILL_POLL_ATTEMPTS", "10")),
            fill_poll_interval_s=float(os.getenv("FILL_POLL_INTERVAL_S", "1.0")),
            tp_watchdog_interval_s=int(os.getenv("TP_WATCHDOG_INTERVAL_S", "60")),
            max_total_exposure=float(os.getenv("MAX_TOTAL_EXPOSURE", "1.0")),
            max_concurrent_trades=int(os.getenv("MAX_CONCURRENT_TRADES", "1")),
            drift_check_window=int(os.getenv("DRIFT_CHECK_WINDOW", "50")),
            drift_accuracy_threshold=float(
                os.getenv("DRIFT_ACCURACY_THRESHOLD", "0.50")
            ),
        )
    return _settings

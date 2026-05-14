import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # API
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    openrouter_api_key: Optional[str] = None

    # Model paths
    model_path: str = "models/rf_model.pkl"
    features_path: str = "models/model_features.pkl"
    scaler_path: str = "models/scaler.pkl"

    # Trading
    symbol: str = "BTC/USD"
    timeframe_hours: int = 1
    confidence_threshold: float = 0.55
    sentiment_floor: float = -0.5
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 2.0
    sl_limit_slippage: float = 0.005
    max_risk_per_trade: float = 0.05
    daily_loss_limit_pct: float = 0.10
    max_total_exposure: float = 1.0
    max_concurrent_trades: int = 1
    max_hold_candles: int = 12  # forward bars to check for TP/SL hit when building labels

    # Execution
    fill_poll_attempts: int = 10
    fill_poll_interval_s: float = 1.0
    tp_watchdog_interval_s: int = 60

    # Drift detection
    drift_check_window: int = 30
    drift_confidence_floor: float = 0.50

    # LLM
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "google/gemini-2.5-flash-lite"

    # Database
    db_path: str = "data/trading_bot.db"

    # Flask
    flask_port: int = 5000

    @property
    def sentiment_enabled(self) -> bool:
        return self.openrouter_api_key is not None


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        _settings = Settings(
            alpaca_api_key=api_key,
            alpaca_secret_key=secret,
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            timeframe_hours=int(os.getenv("TIMEFRAME_HOURS", "1")),
            max_hold_candles=int(os.getenv("MAX_HOLD_CANDLES", "12")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.55")),
            sentiment_floor=float(os.getenv("SENTIMENT_FLOOR", "-0.5")),
            sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.0")),
            tp_atr_mult=float(os.getenv("TP_ATR_MULT", "2.0")),
            sl_limit_slippage=float(os.getenv("SL_LIMIT_SLIPPAGE", "0.005")),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.05")),
            daily_loss_limit_pct=float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.10")),
            max_total_exposure=float(os.getenv("MAX_TOTAL_EXPOSURE", "1.0")),
        )
    return _settings

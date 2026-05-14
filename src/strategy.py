"""
Trading signal logic, position sizing, and SL/TP calculation.
"""

import math
import logging

logger = logging.getLogger(__name__)


def should_enter(
    prediction: int,
    confidence: float,
    sentiment: float,
    has_open_position: bool,
    confidence_threshold: float = 0.55,
    sentiment_floor: float = -0.5,
) -> bool:
    """Return True if all entry conditions are met."""
    if has_open_position:
        return False
    if prediction != 1:
        return False
    if confidence < confidence_threshold:
        logger.info("Signal rejected: confidence %.3f < %.3f", confidence, confidence_threshold)
        return False
    if sentiment < sentiment_floor:
        logger.info("Signal rejected: sentiment %.3f < %.3f", sentiment, sentiment_floor)
        return False
    return True


def position_size(
    buying_power: float,
    current_price: float,
    atr: float,
    sl_atr_mult: float = 1.0,
    max_risk_per_trade: float = 0.05,
    max_total_exposure: float = 1.0,
) -> tuple[float, float]:
    """
    Calculate position size using volatility-adjusted Kelly criterion.

    Returns (quantity, leverage). Quantity is floored to 6 decimal places.
    """
    if buying_power < 11.0 or current_price <= 0:
        return 0.0, 0.0

    sl_distance = atr * sl_atr_mult
    if sl_distance <= 0:
        return 0.0, 0.0

    stop_pct = sl_distance / current_price
    risk_dollars = buying_power * max_risk_per_trade
    ideal_value = risk_dollars / stop_pct

    max_value = buying_power * max_total_exposure * 0.98
    min_value = buying_power * 0.05
    pos_value = min(max(ideal_value, min_value), max_value)
    pos_value = max(pos_value, 11.0)

    qty = _floor(pos_value / current_price, 6)

    # Guard against minimum order size
    if qty * current_price < 10.1:
        qty = _floor(10.5 / current_price, 6)

    if qty * current_price > max_value:
        return 0.0, 0.0

    leverage = (qty * current_price) / buying_power if buying_power > 0 else 0.0
    return qty, leverage


def sl_tp_prices(
    entry_price: float,
    atr: float,
    sl_atr_mult: float = 1.0,
    tp_atr_mult: float = 2.0,
    sl_limit_slippage: float = 0.005,
) -> tuple[float, float, float]:
    """
    Compute stop-loss stop price, stop-loss limit price, and take-profit price.

    Returns (stop_price, limit_price, take_profit_price).
    """
    stop_price = round(entry_price - atr * sl_atr_mult, 2)
    limit_price = round(stop_price * (1 - sl_limit_slippage), 2)
    take_profit = round(entry_price + atr * tp_atr_mult, 2)
    return stop_price, limit_price, take_profit


def check_circuit_breaker(daily_pnl: float, initial_capital: float, daily_loss_limit_pct: float) -> bool:
    """Return True if daily loss limit has been hit (trading should halt)."""
    limit = initial_capital * daily_loss_limit_pct
    if daily_pnl < -limit:
        logger.warning("Circuit breaker: daily P&L $%.2f < -$%.2f", daily_pnl, limit)
        return True
    return False


def _floor(value: float, precision: int) -> float:
    factor = 10 ** precision
    return math.floor(value * factor) / factor

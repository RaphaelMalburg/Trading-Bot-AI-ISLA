"""
Alpaca API wrapper — all order execution and account queries live here.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def get_client(api_key: str, secret_key: str):
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key, secret_key, paper=True)


def get_account_info(client) -> dict:
    """Return equity, buying_power, and today's P&L."""
    account = client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    last_equity = float(account.last_equity)
    pnl = equity - last_equity
    pnl_pct = (pnl / last_equity * 100) if last_equity > 0 else 0.0
    return {
        "equity": equity,
        "buying_power": buying_power,
        "pnl_today": pnl,
        "pnl_today_pct": pnl_pct,
    }


def get_position(client, symbol: str) -> Optional[object]:
    """Return the open position for symbol, or None."""
    try:
        positions = client.get_all_positions()
        norm = symbol.replace("/", "").upper()
        return next((p for p in positions if p.symbol.replace("/", "").upper() == norm), None)
    except Exception as e:
        logger.warning("Could not fetch positions: %s", e)
        return None


def get_all_positions(client) -> list:
    """Return list of all open positions."""
    try:
        return client.get_all_positions()
    except Exception as e:
        logger.warning("Could not fetch all positions: %s", e)
        return []


def market_buy(client, symbol: str, qty: float) -> Optional[object]:
    """Submit a market buy order. Returns the order object."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
    try:
        order = client.submit_order(req)
        logger.info("BUY order submitted: %s qty=%.6f", order.id, qty)
        return order
    except Exception as e:
        logger.error("BUY failed: %s", e)
        return None


def stop_limit_sell(client, symbol: str, qty: float, stop_price: float, limit_price: float) -> Optional[object]:
    """Submit a stop-limit sell (stop-loss) order. Returns the order object."""
    from alpaca.trading.requests import StopLimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    req = StopLimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        stop_price=stop_price,
        limit_price=limit_price,
        time_in_force=TimeInForce.GTC,
    )
    try:
        order = client.submit_order(req)
        logger.info("SL order submitted: %s stop=%.2f", order.id, stop_price)
        return order
    except Exception as e:
        logger.error("SL failed: %s", e)
        return None


def close_position(client, symbol: str) -> bool:
    """Close position for symbol. Returns True on success."""
    try:
        client.close_position(symbol)
        logger.info("Position closed: %s", symbol)
        return True
    except Exception as e:
        logger.warning("Close position error (%s): %s", symbol, e)
        return False


def cancel_orders(client, symbol: str) -> int:
    """Cancel all open orders for symbol. Returns count cancelled."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    try:
        open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        logger.warning("Could not list orders: %s", e)
        return 0

    norm = symbol.replace("/", "").upper()
    cancelled = 0
    for o in open_orders:
        if o.symbol.replace("/", "").upper() != norm:
            continue
        try:
            client.cancel_order_by_id(o.id)
            cancelled += 1
        except Exception as e:
            logger.warning("Failed to cancel order %s: %s", o.id, e)
    return cancelled


def wait_for_fill(client, order_id: str, attempts: int = 10, interval: float = 1.0) -> Optional[object]:
    """Poll order until filled or terminal state. Returns final order object."""
    from alpaca.trading.enums import OrderStatus

    for _ in range(attempts):
        try:
            order = client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            if order.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                return order
            time.sleep(interval)
        except Exception as e:
            logger.warning("Error polling order %s: %s", order_id, e)
    return None


def get_open_orders(client, symbol: str) -> list:
    """Return open orders for symbol."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        return orders
    except Exception as e:
        logger.warning("Could not fetch open orders: %s", e)
        return []


def place_disaster_stop(
    client,
    symbol: str,
    qty: float,
    entry_price: float,
    atr: float,
    sl_atr_mult: float = 1.0,
    slippage: float = 0.01,
) -> Optional[object]:
    """
    Place a wide stop-limit sell at 3× the normal SL distance.
    This is a crash safety net — app-side exit logic handles the real SL/TP.
    """
    disaster_distance = atr * sl_atr_mult * 3
    stop_price = round(entry_price - disaster_distance, 2)
    limit_price = round(stop_price * (1 - slippage), 2)
    logger.info("Disaster stop: stop=%.2f limit=%.2f (3×ATR below entry)", stop_price, limit_price)
    return stop_limit_sell(client, symbol, qty, stop_price, limit_price)

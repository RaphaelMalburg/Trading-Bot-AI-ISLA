#!/usr/bin/env python3
"""
One-shot recovery: attach a Stop Loss to any open BTC/USD position that is
currently missing one. Crypto on Alpaca only allows ONE resting SELL order
per position, so we attach the SL only — TP is enforced by the bot's hourly
loop comparing price against entry + (ATR * TP_ATR_MULT).
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopLimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from src.trading_bot_multi import (
    SYMBOL_BTC, SL_ATR_MULT, TP_ATR_MULT, SL_LIMIT_SLIPPAGE,
    get_latest_data, process_single_asset,
)

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)


def current_atr():
    df = get_latest_data(SYMBOL_BTC)
    proc = process_single_asset(df, "btc")
    return float(proc["btc_atr"].iloc[-1]), float(proc["btc_close"].iloc[-1])


def main():
    positions = [p for p in trading_client.get_all_positions() if p.symbol.replace("/", "") == "BTCUSD"]
    if not positions:
        print("No BTC/USD position. Nothing to do.")
        return

    p = positions[0]
    qty = float(p.qty)
    entry = float(p.avg_entry_price)

    open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    sl_exists = any(
        o.symbol == p.symbol and o.order_type.name in ("STOP", "STOP_LIMIT")
        for o in open_orders
    )

    atr, last_close = current_atr()
    stop_price = round(entry - (atr * SL_ATR_MULT), 2)
    tp_target = round(entry + (atr * TP_ATR_MULT), 2)

    print(f"Position: {p.symbol} qty={qty} entry=${entry:.2f} | last=${last_close:.2f} ATR=${atr:.2f}")
    print(f"SL trigger=${stop_price:.2f} ({((stop_price-entry)/entry)*100:.2f}%)  "
          f"TP target (soft) =${tp_target:.2f} ({((tp_target-entry)/entry)*100:+.2f}%)")

    if sl_exists:
        print("  SL already present, nothing to do.")
        return

    sl_limit_price = round(stop_price * (1 - SL_LIMIT_SLIPPAGE), 2)
    sl = trading_client.submit_order(StopLimitOrderRequest(
        symbol=SYMBOL_BTC, qty=qty, side=OrderSide.SELL,
        stop_price=stop_price, limit_price=sl_limit_price,
        time_in_force=TimeInForce.GTC,
    ))
    print(f"  SL submitted: trigger=${stop_price:.2f} limit=${sl_limit_price:.2f} id={sl.id}")


if __name__ == "__main__":
    main()

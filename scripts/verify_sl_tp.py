#!/usr/bin/env python3
"""
Verify that the bot has placed both an SL (STOP) and a TP (LIMIT) sell order
for each open long position. Alpaca crypto does not support bracket/OCO orders,
so the bot submits two independent orders after a BUY fills.
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

positions = trading_client.get_all_positions()
if not positions:
    print("No open positions.")
    raise SystemExit(0)

open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))

# Position symbols come without slash (e.g. BTCUSD), order symbols with slash (BTC/USD).
def normalize(s: str) -> str:
    return s.replace("/", "").upper()

for p in positions:
    matching = [o for o in open_orders if normalize(o.symbol) == normalize(p.symbol)]
    sl = next((o for o in matching if o.order_type.name in ("STOP", "STOP_LIMIT")), None)
    tp = next((o for o in matching if o.order_type.name == "LIMIT"), None)
    print(f"{p.symbol} qty={p.qty} entry=${float(p.avg_entry_price):.2f}")
    print(f"  SL: {'OK trigger=$' + str(float(sl.stop_price)) if sl else 'MISSING'}")
    print(f"  TP: {'OK @ $' + str(float(tp.limit_price)) if tp else 'MISSING (soft TP via hourly check)'}")

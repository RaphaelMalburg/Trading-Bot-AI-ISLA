"""Quick debug: list BTC orders and balance state."""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()
tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

print("=== ACCOUNT ===")
acct = tc.get_account()
print(f"buying_power={acct.buying_power} cash={acct.cash} equity={acct.equity}")

print("\n=== POSITIONS ===")
for p in tc.get_all_positions():
    print(f"  {p.symbol} qty={p.qty} avg_entry=${float(p.avg_entry_price):.2f} side={p.side}")

for status in [QueryOrderStatus.OPEN, QueryOrderStatus.ALL]:
    orders = tc.get_orders(GetOrdersRequest(status=status))
    btc = [o for o in orders if "BTC" in o.symbol]
    print(f"\n=== {status.name} ({len(btc)} BTC orders) ===")
    for o in btc[:20]:
        print(f"  {o.symbol} {o.side.name} {o.order_type.name} qty={o.qty} "
              f"status={o.status.name} stop={o.stop_price} lim={o.limit_price} created={o.created_at}")

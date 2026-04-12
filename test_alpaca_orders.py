import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

try:
    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=5, nested=True)
    orders = trading_client.get_orders(req)
    for o in orders:
        print(o.id, o.side, o.qty, o.filled_avg_price, o.filled_at)
except Exception as e:
    print(e)

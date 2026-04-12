import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAccountActivitiesRequest

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

try:
    req = GetAccountActivitiesRequest(activity_types=["FILL"])
    activities = trading_client.get_account_activities(req)
    for act in activities:
        print(act)
except Exception as e:
    print(e)

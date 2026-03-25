import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Configurações do Modelo
MODEL_PATH = "models/model.pkl"
Data_PATH = "data/market_data.csv"

# Parâmetros de Trading
SYMBOL = "EURUSD"
TIMEFRAME = "1Hour"

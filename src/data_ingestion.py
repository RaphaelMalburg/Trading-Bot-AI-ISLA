import os
import logging
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv
import pytz

load_dotenv()

# ==========================================
# General ingestion configuration
# ==========================================
SYMBOL = "BTC/USD"  # Target asset
TIMEFRAME = TimeFrame.Hour  # Bar size (H1 = 1 hour)
START_DATE = datetime(2020, 1, 1, tzinfo=pytz.UTC)
END_DATE = datetime(2023, 12, 31, tzinfo=pytz.UTC)
DATA_PATH = "data/btc_usd_hourly.csv"

logger = logging.getLogger(__name__)


def fetch_historical_data():
    """
    Pull historical OHLCV bars from Alpaca for ML training.
    """
    try:
        client = CryptoHistoricalDataClient()

        logger.info("Starting data collection for %s...", SYMBOL)
        logger.info("Period: %s to %s", START_DATE.date(), END_DATE.date())

        req = CryptoBarsRequest(
            symbol_or_symbols=[SYMBOL],
            timeframe=TIMEFRAME,
            start=START_DATE,
            end=END_DATE
        )

        bars = client.get_crypto_bars(req)

        if bars.df.empty:
            logger.warning("No data returned.")
            return None

        df = bars.df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        logger.info("Data saved to: %s", DATA_PATH)
        logger.info("Total records collected: %d", len(df))

        return df

    except Exception as e:
        logger.error("Data collection failed: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    fetch_historical_data()

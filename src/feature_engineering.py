import logging
import pandas as pd
import numpy as np
import ta
import joblib

logger = logging.getLogger(__name__)


def load_data(filepath="data/btc_usd_hourly.csv"):
    """
    Load raw OHLCV data from CSV and ensure chronological ordering.
    Order is critical so that rolling stats and lags never leak future data.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def add_technical_indicators(df):
    """
    Feature-engineering core.
    Takes raw OHLCV bars and computes technical indicators that help the ML
    model spot market patterns.
    """
    df = df.copy()

    # Forward-fill missing values
    df = df.ffill()

    # ==========================================
    # 1. Momentum indicators (trend strength)
    # ==========================================

    # RSI (Relative Strength Index): overbought / oversold gauge
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    # Stochastic oscillator: another momentum gauge based on price vs. recent range
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)

    # ==========================================
    # 2. Trend indicators (direction)
    # ==========================================

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # EMAs (exponential moving averages): weight recent prices more heavily
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

    # ==========================================
    # 3. Volatility indicators (risk / range)
    # ==========================================

    # Bollinger Bands: price dispersion around the moving average
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']  # Normalized

    # ATR (Average True Range): typical candle size; vital for stop-loss sizing and Kelly criterion
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # ==========================================
    # 4. Volume indicators
    # ==========================================
    # OBV (On-Balance Volume): relates volume flow to price changes
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # ==========================================
    # 5. Lags + target construction
    # ==========================================

    # Log returns (normalized percentage changes)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # TARGET:
    # 1 (long) if next hour's close is higher than now.
    # 0 (flat) if next hour's close is lower or equal.
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop rows left empty by rolling/lagged calculations
    df = df.dropna()

    return df


def prepare_features(df):
    """
    Filter to ML-useful columns and create normalized features.
    The model handles relative distances (e.g. % from EMA) better than absolute prices.
    """
    feature_cols = [
        'rsi_14', 'stoch_k',
        'macd', 'macd_signal', 'macd_diff',
        'bb_width', 'atr_14', 'obv',
        'log_return'
    ]

    # Normalized distances from moving averages (price relative to EMA, in %)
    df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['close']
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close']
    df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['close']

    feature_cols.extend(['dist_ema_20', 'dist_ema_50', 'dist_ema_200'])

    X = df[feature_cols]
    y = df['target']

    return X, y, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting feature engineering...")

    df = load_data()
    logger.info("Data loaded: %s", df.shape)

    df_processed = add_technical_indicators(df)
    logger.info("Indicators computed. Shape after dropna: %s", df_processed.shape)

    X, y, features = prepare_features(df_processed)

    df_processed.to_csv("data/processed_data.csv", index=False)
    logger.info("Processed data saved to data/processed_data.csv")

    joblib.dump(features, "models/feature_names.pkl")

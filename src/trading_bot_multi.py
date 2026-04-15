import os
import sys
import time
import pandas as pd
import joblib
import ta
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

sys.path.append(os.getcwd())

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.sentiment_analysis import analyze_sentiment
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# --- Bot Configuration ---
SYMBOL_BTC = "BTC/USD"
SYMBOL_ETH = "ETH/USD"
TIMEFRAME = TimeFrame.Hour
SL_ATR_MULT = 1.0
MAX_RISK_PER_TRADE = 0.05
MODEL_PATH = 'models/rf_model.pkl'
FEATURES_PATH = 'models/model_features.pkl'


def get_latest_data(symbol, lookback_days=30):
    client = CryptoHistoricalDataClient()
    now = datetime.now(pytz.UTC)
    start = now - timedelta(days=lookback_days)

    req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TIMEFRAME, start=start, end=now)
    bars = client.get_crypto_bars(req)
    df = bars.df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    return df


def get_latest_news(symbol="BTC"):
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
    params = {"symbols": symbol, "limit": 5, "include_content": False}
    try:
        import requests
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return [n['headline'] for n in response.json().get('news', [])]
        return []
    except:
        return []


def process_single_asset(df, asset_name):
    df = df.set_index('timestamp').sort_index()
    df['returns'] = df['close'].pct_change()

    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    ema20 = EMAIndicator(close=df['close'], window=20)
    df['ema20'] = ema20.ema_indicator()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    df = df.add_prefix(f'{asset_name}_')
    return df


def prepare_features(df_btc, df_eth):
    df_btc_proc = process_single_asset(df_btc, 'btc')
    df_eth_proc = process_single_asset(df_eth, 'eth')

    df = pd.merge(df_btc_proc, df_eth_proc, left_index=True, right_index=True, how='inner')

    df['spread_returns'] = df['btc_returns'] - df['eth_returns']
    df['price_ratio'] = df['btc_close'] / df['eth_close']
    df['rolling_corr_24h'] = df['btc_returns'].rolling(window=24).corr(df['eth_returns'])

    for lag in [1, 2, 3]:
        df[f'btc_ret_lag_{lag}'] = df['btc_returns'].shift(lag)
        df[f'eth_ret_lag_{lag}'] = df['eth_returns'].shift(lag)
        df[f'btc_rsi_lag_{lag}'] = df['btc_rsi'].shift(lag)
        df[f'eth_rsi_lag_{lag}'] = df['eth_rsi'].shift(lag)

    df = df.dropna()
    return df


def calculate_position_size(capital, current_price, atr):
    # If we don't even have enough buying power for Alpaca's crypto minimum ($10 + buffer), abort.
    if capital < 11.0:
        return 0, 0

    dist_stop = atr * SL_ATR_MULT
    stop_pct = dist_stop / current_price

    if stop_pct == 0:
        return 0, 0

    risk_amount = capital * MAX_RISK_PER_TRADE
    position_value = risk_amount / stop_pct

    # Adaptive leverage based on account size
    # Small accounts (<$1k): 1x only (no leverage, safe for underfunded accounts)
    # Medium accounts ($1k-$10k): scale from 1x to 5x
    # Large accounts (>$10k): full 5x leverage
    if capital < 1000:
        max_leverage = 1.0
    elif capital < 10000:
        # Linear interpolation: 1x at $1k, 5x at $10k
        max_leverage = 1.0 + (capital - 1000) / 9000 * 4.0
    else:
        max_leverage = 5.0

    # Reserve 5% buffer for Alpaca bracket order overhead (SL/TP legs)
    max_pos = capital * max_leverage * 0.90
    min_pos = capital * 0.05
    final_pos_value = min(max(position_value, min_pos), max_pos)
    
    # Alpaca minimum order for crypto is usually $10 or $1 depending on the asset. We use $11 to be safe.
    final_pos_value = max(final_pos_value, 11.0)

    qty = final_pos_value / current_price
    qty = round(qty, 4)
    
    # Ensure minimum qty after rounding still meets Alpaca's notional requirements
    if qty * current_price < 10.5:
        qty = round(11.0 / current_price, 4)

    leverage = (qty * current_price) / capital if capital > 0 else 0

    return qty, leverage


def trade_logic_multi():
    """
    Main bot execution function.
    Returns a dict with all pipeline data for the web dashboard.
    """
    result = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "steps": [],
        "error": None,
    }

    print(f"\n--- Bot Multi-Ativo (BTC+ETH): {datetime.now()} ---")

    # --- Step 1: Connect to Alpaca ---
    t0 = time.time()
    try:
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        capital = float(account.buying_power)
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        pnl = equity - last_equity
        pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0

        result["equity"] = equity
        result["buying_power"] = capital
        result["pnl_today"] = pnl
        result["pnl_today_pct"] = pnl_pct
        result["steps"].append({"name": "Connect Alpaca", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print(f"Equity Total: ${equity:.2f}")
    except Exception as e:
        result["steps"].append({"name": "Connect Alpaca", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
        result["error"] = f"Alpaca connection error: {e}"
        print(f"Erro ao conectar na Alpaca: {e}")
        return result

    # --- Step 2: Fetch Market Data ---
    t0 = time.time()
    print("Baixando dados BTC e ETH...")
    try:
        df_btc = get_latest_data(SYMBOL_BTC)
        df_eth = get_latest_data(SYMBOL_ETH)
        last_close_btc = df_btc.iloc[-1]['close']
        result["btc_close"] = float(last_close_btc)

        # Store last 100 candles for charting
        chart_df = df_btc.tail(100).copy()
        if 'timestamp' in chart_df.columns:
            chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        result["ohlcv_data"] = chart_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')

        result["steps"].append({"name": "Fetch Market Data", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print(f"BTC Close: {last_close_btc:.2f}")
    except Exception as e:
        result["steps"].append({"name": "Fetch Market Data", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
        result["error"] = f"Data fetch error: {e}"
        print(f"Erro ao baixar dados: {e}")
        return result

    # --- Step 3: Calculate Features ---
    t0 = time.time()
    print("Calculando features...")
    try:
        df_full = process_single_asset(df_btc, 'btc')
        df_full.columns = [c.replace('btc_', '') for c in df_full.columns]

        current_state = df_full.iloc[[-1]].copy()
        current_atr = current_state['atr'].values[0]

        feature_cols = joblib.load(FEATURES_PATH)

        for col in feature_cols:
            if col not in current_state.columns:
                current_state[col] = 0.0

        X_latest = current_state[feature_cols]

        # Extract indicator values for dashboard display
        last_row = df_full.iloc[-1]
        result["indicators"] = {
            "rsi": float(last_row.get('rsi', 0)),
            "macd": float(last_row.get('macd', 0)),
            "macd_signal": float(last_row.get('macd_signal', 0)),
            "ema20": float(last_row.get('ema20', 0)),
            "bb_width": float(last_row.get('bb_width', 0)),
            "bb_high": float(last_row.get('bb_high', 0)),
            "bb_low": float(last_row.get('bb_low', 0)),
            "atr": float(current_atr),
        }

        # Store indicator series for chart overlays (last 100 bars)
        chart_indicators = df_full.tail(100).copy()
        chart_indicators.index = chart_indicators.index.astype(str)
        result["chart_indicators"] = {
            "ema20": chart_indicators['ema20'].tolist(),
            "bb_high": chart_indicators['bb_high'].tolist(),
            "bb_low": chart_indicators['bb_low'].tolist(),
            "rsi": chart_indicators['rsi'].tolist(),
            "macd": chart_indicators['macd'].tolist(),
            "macd_signal": chart_indicators['macd_signal'].tolist(),
            "volume": chart_indicators['volume'].tolist() if 'volume' in chart_indicators.columns else [],
            "timestamps": chart_indicators.index.tolist(),
        }

        result["steps"].append({"name": "Calculate Features", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print(f"ATR Atual (BTC): {current_atr:.2f}")
    except Exception as e:
        result["steps"].append({"name": "Calculate Features", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
        result["error"] = f"Feature calculation error: {e}"
        print(f"Erro no processamento de features: {e}")
        return result

    # --- Step 4: Sentiment Analysis ---
    t0 = time.time()
    print("Analisando Sentimento (Gemini)...")
    try:
        headlines = get_latest_news("BTC")
        sentiment_score = analyze_sentiment(headlines) if headlines else 0.0
        result["headlines"] = headlines
        result["sentiment_score"] = float(sentiment_score)
        result["steps"].append({"name": "Sentiment Analysis", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print(f"Score de Sentimento: {sentiment_score:.2f}")
    except Exception as e:
        result["steps"].append({"name": "Sentiment Analysis", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
        sentiment_score = 0.0
        result["headlines"] = []
        result["sentiment_score"] = 0.0
        print(f"Erro no sentimento (usando neutro): {e}")

    # --- Step 5: ML Prediction ---
    t0 = time.time()
    print("Executando Modelo Preditivo...")
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        confianca = probability[prediction]

        result["prediction"] = int(prediction)
        result["confidence"] = float(confianca)
        result["prediction_label"] = "BUY" if prediction == 1 else "SELL"
        result["probabilities"] = {"down": float(probability[0]), "up": float(probability[1])}
        result["steps"].append({"name": "ML Prediction", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print(f"Previsao: {'SUBIR (1)' if prediction == 1 else 'CAIR (0)'}")
        print(f"Confianca: {confianca:.2%}")
    except Exception as e:
        result["steps"].append({"name": "ML Prediction", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
        result["error"] = f"Prediction error: {e}"
        print(f"Erro na predicao: {e}")
        return result

    # --- Step 6: Execute Orders ---
    t0 = time.time()
    if prediction == 1 and confianca > 0.53 and sentiment_score >= -0.5:
        print("SINAL DE COMPRA VALIDADO!")
        try:
            positions = trading_client.get_all_positions()
            has_btc_position = any(p.symbol == "BTC/USD" for p in positions)

            if has_btc_position:
                result["action"] = "ALREADY_POSITIONED"
                result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
                print("Ja existe uma posicao em BTC/USD. Mantendo.")
                return result

            qty, leverage = calculate_position_size(capital, last_close_btc, current_atr)
            result["position_qty"] = float(qty)
            result["leverage"] = float(leverage)

            if qty <= 0:
                result["action"] = "INSUFFICIENT_FUNDS"
                result["error"] = "Not enough buying power to meet minimum order size ($10)."
                result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
                print("Buying power insufficient for minimum crypto order. Abortando.")
                return result

            stop_price = last_close_btc - (current_atr * SL_ATR_MULT)
            take_profit_price = last_close_btc + (current_atr * 2.0)
            result["stop_loss"] = float(stop_price)
            result["take_profit"] = float(take_profit_price)

            print(f"Enviando ordem de COMPRA...")
            print(f"   Entry: ~{last_close_btc:.2f}")
            print(f"   Stop Loss: {stop_price:.2f}")
            print(f"   Take Profit: {take_profit_price:.2f}")
            print(f"   Tamanho: {qty:.4f} BTC (Lev: {leverage:.2f}x)")

            # Place main BUY order
            req = MarketOrderRequest(
                symbol=SYMBOL_BTC,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )

            order = trading_client.submit_order(req)
            result["action"] = "BUY_ORDER_SENT"
            result["order_id"] = str(order.id)
            print(f"Ordem enviada com sucesso! ID: {order.id}")

            # Place separate STOP LOSS order
            try:
                sl_req = StopOrderRequest(
                    symbol=SYMBOL_BTC,
                    qty=qty,
                    side=OrderSide.SELL,
                    stop_price=stop_price,
                    time_in_force=TimeInForce.GTC,
                )
                sl_order = trading_client.submit_order(sl_req)
                print(f"Stop Loss criado! ID: {sl_order.id} @ ${stop_price:.2f}")
            except Exception as e:
                print(f"Erro ao criar Stop Loss: {e}")

            # Place separate TAKE PROFIT order
            try:
                tp_req = LimitOrderRequest(
                    symbol=SYMBOL_BTC,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=take_profit_price,
                    time_in_force=TimeInForce.GTC,
                )
                tp_order = trading_client.submit_order(tp_req)
                print(f"Take Profit criado! ID: {tp_order.id} @ ${take_profit_price:.2f}")
            except Exception as e:
                print(f"Erro ao criar Take Profit: {e}")

            result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})

        except Exception as e:
            result["action"] = "ORDER_ERROR"
            result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
            print(f"Erro ao enviar ordem: {e}")

    elif prediction == 0:
        print("Sinal de VENDA/NEUTRO. Fechando posicoes longas se houver...")
        try:
            positions = trading_client.get_all_positions()
            closed = False
            for p in positions:
                if p.symbol == "BTC/USD" and p.side == 'long':
                    trading_client.close_position("BTC/USD")
                    closed = True
                    print("Posicao fechada com sucesso.")

            result["action"] = "CLOSE_POSITION" if closed else "NO_POSITION_TO_CLOSE"
            result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        except Exception as e:
            result["action"] = "CLOSE_ERROR"
            result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
            print(f"Erro ao fechar posicoes: {e}")
    else:
        result["action"] = "NO_SIGNAL"
        result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print("Condicoes nao atendidas para entrada. Aguardando proximo ciclo.")

    return result


if __name__ == "__main__":
    trade_logic_multi()

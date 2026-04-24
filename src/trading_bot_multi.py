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
from alpaca.trading.requests import MarketOrderRequest, StopLimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
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
TP_ATR_MULT = 2.0
SL_LIMIT_SLIPPAGE = 0.005  # 0.5% slack below stop_price for the stop-limit fill price
MAX_RISK_PER_TRADE = 0.05
CONFIDENCE_THRESHOLD = 0.55
SENTIMENT_FLOOR = -0.5
FILL_POLL_ATTEMPTS = 10
FILL_POLL_INTERVAL_S = 1.0
MODEL_PATH = 'models/rf_model.pkl'
FEATURES_PATH = 'models/model_features.pkl'


import math

def floor_to_precision(value, precision):
    factor = 10 ** precision
    return math.floor(value * factor) / factor

def get_latest_data(symbol, lookback_days=30, max_retries=5):
    now = datetime.now(pytz.UTC)
    start = now - timedelta(days=lookback_days)
    req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TIMEFRAME, start=start, end=now)
    
    for attempt in range(max_retries):
        try:
            # Re-instantiate client each time to clear any stale connections
            client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
            bars = client.get_crypto_bars(req)
            
            if not bars or bars.df.empty:
                raise ValueError(f"No data returned for {symbol}")
                
            df = bars.df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol].copy()

            return df
        except Exception as e:
            wait = (attempt + 1) * 3  # 3s, 6s, 9s, 12s, 15s
            if attempt < max_retries - 1:
                print(f"⚠️ Tentativa {attempt+1}/{max_retries} falhou ({symbol}): {e}. Retentando em {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Todas as {max_retries} tentativas falharam para {symbol}.")
                raise e


def get_latest_news(symbol="BTC", max_retries=3):
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
    params = {"symbols": symbol, "limit": 5, "include_content": False}
    
    for attempt in range(max_retries):
        try:
            import requests
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return [n['headline'] for n in response.json().get('news', [])]
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
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
    # Alpaca crypto minimum notional is ~$10; require a small buffer.
    if capital < 11.0:
        return 0, 0

    dist_stop = atr * SL_ATR_MULT
    stop_pct = dist_stop / current_price
    if stop_pct == 0:
        return 0, 0

    risk_amount = capital * MAX_RISK_PER_TRADE
    position_value = risk_amount / stop_pct

    # Alpaca crypto is cash-only: leverage > 1 is rejected as insufficient buying power.
    max_pos = capital * 0.95
    min_pos = capital * 0.05
    final_pos_value = min(max(position_value, min_pos), max_pos)
    final_pos_value = max(final_pos_value, 11.0)
    final_pos_value = min(final_pos_value, max_pos)

    # Use 6 decimal places for qty to ensure high-priced assets like BTC 
    # can meet the $10 minimum order size accurately.
    qty = round(final_pos_value / current_price, 6)
    if qty * current_price < 10.1:
        qty = round(10.5 / current_price, 6)
    if qty * current_price > max_pos:
        return 0, 0

    leverage = (qty * current_price) / capital if capital > 0 else 0
    return qty, leverage


def wait_for_fill(trading_client, order_id, attempts=FILL_POLL_ATTEMPTS, interval=FILL_POLL_INTERVAL_S):
    """Poll an order until it reaches FILLED status. Returns the final order object or None on timeout."""
    for _ in range(attempts):
        order = trading_client.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return order
        if order.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
            return order
        time.sleep(interval)
    return trading_client.get_order_by_id(order_id)


def cancel_open_orders_for_symbol(trading_client, symbol):
    """Cancel any resting (non-filled) orders on a symbol. Returns count cancelled.
    Symbol matching is slash-insensitive (BTC/USD == BTCUSD)."""
    try:
        open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    except Exception as e:
        print(f"Could not list open orders: {e}")
        return 0
    target = symbol.replace("/", "").upper()
    cancelled = 0
    for o in open_orders:
        if o.symbol.replace("/", "").upper() != target:
            continue
        try:
            trading_client.cancel_order_by_id(o.id)
            cancelled += 1
        except Exception as e:
            print(f"Failed to cancel order {o.id}: {e}")
    return cancelled


def trade_logic_multi():
    """
    Main bot execution function.
    Returns a dict with all pipeline data for the web dashboard.
    """
    result = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "steps": [],
        "error": None,
        "prediction": 0,
        "prediction_label": "FLAT",
        "confidence": 0.0,
        "sentiment_score": 0.0,
        "action": "WAITING",
        "position_qty": 0.0,
        "leverage": 0.0,
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

        # Store last 500 candles for charting (TradingView style)
        chart_df = df_btc.tail(500).copy()
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

        # Store indicator series for chart overlays (last 500 bars)
        chart_indicators = df_full.tail(500).copy()
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
        result["prediction_label"] = "LONG" if prediction == 1 else "FLAT"
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
    if prediction == 1 and confianca > CONFIDENCE_THRESHOLD and sentiment_score >= SENTIMENT_FLOOR:
        print("SINAL DE COMPRA VALIDADO!")
        try:
            positions = trading_client.get_all_positions()
            btc_pos = next((p for p in positions if p.symbol == "BTC/USD"), None)

            if btc_pos:
                # Soft take-profit: SL is the only resting SELL we can have on crypto,
                # so if price has reached our TP target relative to entry, close manually.
                entry = float(btc_pos.avg_entry_price)
                tp_target = entry + (current_atr * TP_ATR_MULT)
                
                # Try to find SL from open orders to pass to dashboard
                open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
                current_sl = None
                for o in open_orders:
                    if o.symbol.replace("/", "").upper() == "BTCUSD" and "stop" in str(o.order_type).lower():
                        current_sl = float(o.stop_price) if o.stop_price else None
                        break
                
                # Recovery: if position exists but no SL order found, try to attach one
                if current_sl is None:
                    print("⚠️ Posicao ativa sem Stop Loss detectada. Tentando anexar...")
                    try:
                        entry = float(btc_pos.avg_entry_price)
                        # We don't have the original ATR here, but we can use the current one
                        rec_stop_price = round(entry - (current_atr * SL_ATR_MULT), 2)
                        rec_limit_price = round(rec_stop_price * (1 - SL_LIMIT_SLIPPAGE), 2)
                        
                        # Use actual position qty with safety buffer
                        rec_qty = floor_to_precision(float(btc_pos.qty) * 0.9999, 6)
                        
                        sl_req = StopLimitOrderRequest(
                            symbol=SYMBOL_BTC,
                            qty=rec_qty,
                            side=OrderSide.SELL,
                            stop_price=rec_stop_price,
                            limit_price=rec_limit_price,
                            time_in_force=TimeInForce.GTC,
                        )
                        sl_order = trading_client.submit_order(sl_req)
                        current_sl = rec_stop_price
                        print(f"✅ Stop Loss de recuperacao anexado @ ${rec_stop_price:.2f}")
                    except Exception as e:
                        print(f"❌ Falha na recuperacao do Stop Loss: {e}")

                if last_close_btc >= tp_target:
                    cancel_open_orders_for_symbol(trading_client, "BTC/USD")
                    trading_client.close_position("BTC/USD")
                    result["action"] = "TAKE_PROFIT_HIT"
                    result["take_profit"] = float(tp_target)
                    result["stop_loss"] = current_sl
                    result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
                    print(f"Take-profit alcancado (${last_close_btc:.2f} >= ${tp_target:.2f}). Fechando.")
                    return result

                result["action"] = "ALREADY_POSITIONED"
                result["stop_loss"] = current_sl
                result["take_profit"] = float(tp_target)
                result["position_qty"] = float(btc_pos.qty)
                result["leverage"] = (float(btc_pos.qty) * last_close_btc) / capital if capital > 0 else 0
                result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
                print("Ja existe uma posicao em BTC/USD. Mantendo (SL ativo, TP monitorado).")
                return result

            # No position but maybe stale SELL orders from a manually closed trade are
            # holding qty/buying_power. Clear them so the BUY does not fail.
            orphan = cancel_open_orders_for_symbol(trading_client, "BTC/USD")
            if orphan:
                print(f"Limpas {orphan} ordens SELL orfas antes de comprar.")

            qty, leverage = calculate_position_size(capital, last_close_btc, current_atr)
            result["position_qty"] = float(qty)
            result["leverage"] = float(leverage)

            if qty <= 0:
                result["action"] = "INSUFFICIENT_FUNDS"
                result["error"] = "Not enough buying power to meet minimum order size ($10)."
                result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
                print("Buying power insufficient for minimum crypto order. Abortando.")
                return result

            print(f"Enviando ordem de COMPRA: {qty:.4f} BTC @ ~${last_close_btc:.2f}")

            buy_req = MarketOrderRequest(
                symbol=SYMBOL_BTC,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            buy_order = trading_client.submit_order(buy_req)
            result["order_id"] = str(buy_order.id)
            print(f"Ordem BUY enviada. ID: {buy_order.id}")

            # Wait for fill so we can size SL/TP against the real filled price and qty.
            filled = wait_for_fill(trading_client, buy_order.id)
            if filled is None or filled.status != OrderStatus.FILLED:
                result["action"] = "BUY_ORDER_SENT"
                result["error"] = f"BUY not filled in time (status={getattr(filled, 'status', None)}). SL/TP skipped."
                result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
                print(result["error"])
                return result

            entry_price = float(filled.filled_avg_price) if filled.filled_avg_price else last_close_btc
            
            # Fetch the actual position to get the exact quantity available for SELL (accounts for fees)
            actual_pos = None
            try:
                actual_pos = trading_client.get_open_position(SYMBOL_BTC)
            except:
                pass
            
            filled_qty = float(actual_pos.qty) if actual_pos else (float(filled.filled_qty) if filled.filled_qty else qty)
            
            # Small safety buffer: Alpaca sometimes has tiny rounding differences in 'available'
            # We use 99.9% of the quantity if it's crypto to avoid "insufficient balance" errors
            sl_qty = filled_qty
            if "USD" in SYMBOL_BTC: # It's crypto
                 sl_qty = floor_to_precision(filled_qty * 0.9999, 6)

            stop_price = round(entry_price - (current_atr * SL_ATR_MULT), 2)
            take_profit_price = round(entry_price + (current_atr * TP_ATR_MULT), 2)
            result["stop_loss"] = float(stop_price)
            result["take_profit"] = float(take_profit_price)
            result["action"] = "BUY_ORDER_SENT"

            print(f"BUY preenchido @ ${entry_price:.2f} (Posicao: {filled_qty} BTC). Configurando SL para {sl_qty}...")

            # Alpaca crypto only allows ONE resting SELL order against the position
            # qty (no OCO/bracket). We pick the SL because downside protection is
            # the priority. Take-profit is enforced softly: each hourly run checks
            # whether price >= take_profit_price and closes the position via market.
            sl_error = None
            try:
                sl_limit_price = round(stop_price * (1 - SL_LIMIT_SLIPPAGE), 2)
                sl_req = StopLimitOrderRequest(
                    symbol=SYMBOL_BTC,
                    qty=sl_qty,
                    side=OrderSide.SELL,
                    stop_price=stop_price,
                    limit_price=sl_limit_price,
                    time_in_force=TimeInForce.GTC,
                )
                sl_order = trading_client.submit_order(sl_req)
                result["sl_order_id"] = str(sl_order.id)
                print(f"Stop Loss trigger=${stop_price:.2f} limit=${sl_limit_price:.2f} | order {sl_order.id}")
            except Exception as e:
                sl_error = str(e)
                print(f"Falha ao criar Stop Loss: {e}")

            if sl_error:
                result["error"] = f"SL: {sl_error}"
                result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
            else:
                result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})

        except Exception as e:
            result["action"] = "ORDER_ERROR"
            result["error"] = str(e)
            result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
            print(f"Erro ao enviar ordem: {e}")

    elif prediction == 0:
        print("Sinal FLAT. Fechando posicao BTC e cancelando SL/TP pendentes...")
        try:
            positions = trading_client.get_all_positions()
            has_position = any(p.symbol == "BTC/USD" for p in positions)

            cancelled = cancel_open_orders_for_symbol(trading_client, "BTC/USD")
            if cancelled:
                print(f"Cancelled {cancelled} open SL/TP order(s) for BTC/USD.")

            if has_position:
                trading_client.close_position("BTC/USD")
                result["action"] = "CLOSE_POSITION"
                print("Posicao fechada com sucesso.")
            else:
                result["action"] = "NO_POSITION_TO_CLOSE"

            result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        except Exception as e:
            result["action"] = "CLOSE_ERROR"
            result["error"] = str(e)
            result["steps"].append({"name": "Execute Order", "status": "error", "duration_ms": int((time.time() - t0) * 1000)})
            print(f"Erro ao fechar posicoes: {e}")
    else:
        result["action"] = "NO_SIGNAL"
        result["steps"].append({"name": "Execute Order", "status": "ok", "duration_ms": int((time.time() - t0) * 1000)})
        print("Condicoes nao atendidas para entrada. Aguardando proximo ciclo.")

    return result


if __name__ == "__main__":
    trade_logic_multi()

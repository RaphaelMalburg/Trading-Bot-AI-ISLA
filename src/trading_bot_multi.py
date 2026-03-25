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

# Adicionar diretório raiz ao path para garantir que imports locais funcionem no cron/terminal
sys.path.append(os.getcwd())

# Importações da Alpaca API para operações em conta real/paper
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Importações internas e de indicadores técnicos
from src.sentiment_analysis import analyze_sentiment
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Carregar variáveis de ambiente (API Keys da Alpaca)
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# --- Configurações Gerais do Bot ---
SYMBOL_BTC = "BTC/USD"
SYMBOL_ETH = "ETH/USD"
TIMEFRAME = TimeFrame.Hour        # Operando no gráfico de 1 Hora
SL_ATR_MULT = 1.0                 # Multiplicador do ATR para Stop Loss
MAX_RISK_PER_TRADE = 0.05         # Risco máximo: Arriscar apenas 5% do capital total por trade
MODEL_PATH = 'models/rf_model.pkl'# Caminho do modelo Random Forest treinado
FEATURES_PATH = 'models/model_features.pkl'

def get_latest_data(symbol, lookback_days=30): 
    """
    Baixa o histórico de preços OHLCV recente diretamente da corretora (Alpaca).
    Usado para construir os indicadores técnicos em tempo real.
    """
    client = CryptoHistoricalDataClient()
    now = datetime.now(pytz.UTC)
    start = now - timedelta(days=lookback_days)
    
    # Requisição dos candles (barras) de 1 hora
    req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TIMEFRAME, start=start, end=now)
    bars = client.get_crypto_bars(req)
    df = bars.df.reset_index()
    
    # Padronizar colunas para minúsculas
    df.columns = [c.lower() for c in df.columns]
    
    # Garantir que temos os dados apenas do símbolo solicitado
    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()
        
    return df

def get_latest_news(symbol="BTC"):
    """
    Busca as últimas notícias financeiras relacionadas ao ativo na API de notícias da Alpaca.
    Usado posteriormente para a análise de sentimento via LLM.
    """
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
    """
    Calcula os mesmos indicadores técnicos (RSI, MACD, EMA, Bollinger, ATR) 
    que foram utilizados durante o treinamento do modelo.
    """
    df = df.set_index('timestamp').sort_index()
    
    # Calcular Retornos percentuais
    df['returns'] = df['close'].pct_change()
    
    # Indicadores de Momento e Tendência
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()
    
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    ema20 = EMAIndicator(close=df['close'], window=20)
    df['ema20'] = ema20.ema_indicator()
    
    # Indicadores de Volatilidade
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    
    # Renomear colunas adicionando o prefixo do ativo (ex: btc_rsi, eth_macd)
    df = df.add_prefix(f'{asset_name}_')
    
    return df

def prepare_features(df_btc, df_eth):
    """
    Prepara o dataset final combinando dados de múltiplos ativos e criando 
    features de correlação/lags (Usado se o modelo Multi-Ativo estiver ativo).
    """
    df_btc_proc = process_single_asset(df_btc, 'btc')
    df_eth_proc = process_single_asset(df_eth, 'eth')
    
    df = pd.merge(df_btc_proc, df_eth_proc, left_index=True, right_index=True, how='inner')
    
    # Feature Engineering Avançada (Interação BTC x ETH)
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
    """
    Lógica de Gestão de Risco (Fractional Kelly inspirado).
    Calcula quantos bitcoins comprar baseando-se na distância do Stop Loss.
    """
    dist_stop = atr * SL_ATR_MULT
    stop_pct = dist_stop / current_price # Ex: 0.02 (Queda de 2% atinge o stop)
    
    if stop_pct == 0: return 0, 0
    
    # Valor máximo em Dólares que aceitamos perder neste trade
    risk_amount = capital * MAX_RISK_PER_TRADE
    
    # Tamanho Financeiro Total da Posição = Risco Máximo / Queda Percentual
    position_value = risk_amount / stop_pct
    
    # Travas de segurança: Alavancagem Máxima de 5x, Mínima de 0.1x
    max_pos = capital * 5.0
    min_pos = capital * 0.1 
    final_pos_value = min(max(position_value, min_pos), max_pos)
    
    # Converter valor financeiro em quantidade de moeda e calcular alavancagem
    qty = final_pos_value / current_price
    leverage = final_pos_value / capital
    
    return qty, leverage

def trade_logic_multi():
    """
    Função principal de execução do Bot.
    Deve ser chamada periodicamente (ex: a cada hora pelo Cron).
    1. Conecta na corretora.
    2. Baixa dados e calcula indicadores.
    3. Analisa o sentimento das notícias.
    4. Gera previsão com Random Forest.
    5. Executa ordens de compra/venda com base nos sinais e no risco.
    """
    print(f"\n--- 🤖 Bot Multi-Ativo (BTC+ETH): {datetime.now()} ---")
    
    # 1. Conexão Alpaca (Usando Paper Trading)
    try:
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        capital = float(account.buying_power) 
        equity = float(account.equity)
        print(f"💰 Equity Total: ${equity:.2f}")
    except Exception as e:
        print(f"❌ Erro ao conectar na Alpaca: {e}")
        return

    # 2. Baixar Dados Recentes do Mercado
    print("📥 Baixando dados BTC e ETH...")
    try:
        df_btc = get_latest_data(SYMBOL_BTC)
        df_eth = get_latest_data(SYMBOL_ETH)
        last_close_btc = df_btc.iloc[-1]['close']
        print(f"📉 BTC Close: {last_close_btc:.2f}")
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        return
    
    # 3. Processar Dados e Preparar o Input do Modelo (X_latest)
    print("🧮 Calculando features...")
    try:
        # Atualmente o bot está configurado para usar o modelo base do BTC (Random Forest)
        df_full = process_single_asset(df_btc, 'btc')
        df_full.columns = [c.replace('btc_', '') for c in df_full.columns]
        
        current_state = df_full.iloc[[-1]].copy()
        current_atr = current_state['atr'].values[0]
        
        # Carregar a ordem exata das features usadas no treinamento
        feature_cols = joblib.load(FEATURES_PATH)
        
        # Prevenir quebras caso falte alguma coluna de lag no dado atual
        for col in feature_cols:
            if col not in current_state.columns:
                current_state[col] = 0.0 
        
        X_latest = current_state[feature_cols]
        print(f"📊 ATR Atual (BTC): {current_atr:.2f}")
    except Exception as e:
        print(f"❌ Erro no processamento de features: {e}")
        return
    
    # 4. Análise de Sentimento via IA Generativa (Filtro)
    print("🧠 Analisando Sentimento (Gemini)...")
    try:
        headlines = get_latest_news("BTC")
        sentiment_score = analyze_sentiment(headlines) if headlines else 0.0
        print(f"🌡️ Score de Sentimento: {sentiment_score:.2f}")
    except Exception as e:
        print(f"⚠️ Erro no sentimento (usando neutro): {e}")
        sentiment_score = 0.0
    
    # 5. Predição Direcional com Machine Learning
    print("🤖 Executando Modelo Preditivo...")
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        # Obter o nível de certeza do modelo para a classe escolhida
        confianca = probability[prediction]
        
        print(f"🔮 Previsão: {'SUBIR (1)' if prediction == 1 else 'CAIR (0)'}")
        print(f"🎯 Confiança: {confianca:.2%}")
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return
    
    # 6. Lógica de Execução e Envio de Ordens
    # Regras para entrar Long (Comprado):
    # - O Random Forest deve prever Alta (1).
    # - A probabilidade deve ser maior que 53% (filtro de ruído).
    # - As notícias não podem estar extremamente pessimistas (Sentimento >= -0.5).
    
    if prediction == 1 and confianca > 0.53 and sentiment_score >= -0.5:
        print("🚀 SINAL DE COMPRA VALIDADO!")
        
        try:
            # Impedir a abertura de múltiplas posições do mesmo ativo
            positions = trading_client.get_all_positions()
            has_btc_position = any(p.symbol == "BTC/USD" for p in positions)
            
            if has_btc_position:
                print(f"⚠️ Já existe uma posição em BTC/USD. Mantendo.")
                return
            
            # Calcular a quantidade exata a ser comprada baseada no ATR
            qty, leverage = calculate_position_size(equity, last_close_btc, current_atr)
            print(f"📏 Tamanho calculado: {qty:.4f} BTC (Lev: {leverage:.2f}x)")
            
            if qty <= 0:
                print("⚠️ Quantidade calculada é zero ou negativa. Abortando.")
                return

            # Definir os alvos dinâmicos da operação
            stop_price = last_close_btc - (current_atr * SL_ATR_MULT)
            take_profit_price = last_close_btc + (current_atr * 2.0)
            
            print(f"🛒 Enviando ordem de COMPRA...")
            print(f"   Entry: ~{last_close_btc:.2f}")
            print(f"   Stop Loss: {stop_price:.2f}")
            print(f"   Take Profit: {take_profit_price:.2f}")

            # Bracket Order: Envia a ordem de compra principal atrelada aos limites de perda e ganho
            req = MarketOrderRequest(
                symbol=SYMBOL_BTC,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': take_profit_price}
            )
            
            order = trading_client.submit_order(req)
            print(f"✅ Ordem enviada com sucesso! ID: {order.id}")
            
        except Exception as e:
            print(f"❌ Erro ao enviar ordem: {e}")
            
    # Lógica de Saída/Venda
    elif prediction == 0:
        print("📉 Sinal de VENDA/NEUTRO. Fechando posições longas se houver...")
        # Se o modelo detecta queda, fechamos a posição imediatamente para evitar bater no Stop Loss
        try:
            positions = trading_client.get_all_positions()
            for p in positions:
                if p.symbol == "BTC/USD" and p.side == 'long':
                    print("Fechando posição (Liquidando ativos)...")
                    trading_client.close_position("BTC/USD")
                    print("✅ Posição fechada com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao fechar posições: {e}")
            pass
    else:
        print("⏸️ Condições não atendidas para entrada. Aguardando próximo ciclo.")

if __name__ == "__main__":
    # Ponto de entrada principal ao executar o script diretamente
    trade_logic_multi()

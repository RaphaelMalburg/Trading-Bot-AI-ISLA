import os
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# Carrega variáveis de ambiente (API Keys) do arquivo .env
load_dotenv()

# ==========================================
# CONFIGURAÇÕES GERAIS DE INGESTÃO
# ==========================================
SYMBOL = "BTC/USD"  # Ativo alvo
TIMEFRAME = TimeFrame.Hour  # Intervalo de tempo (H1 = 1 Hora)
START_DATE = datetime(2020, 1, 1, tzinfo=pytz.UTC)  # Início do período de dados
END_DATE = datetime(2023, 12, 31, tzinfo=pytz.UTC)  # Fim do período de dados
DATA_PATH = "data/btc_usd_hourly.csv"  # Caminho para salvar o CSV bruto

def fetch_historical_data():
    """
    Coleta dados históricos completos (OHLCV) da Alpaca API.
    Estes dados serão usados posteriormente para treinar o modelo de Machine Learning.
    """
    try:
        # Instancia o cliente da Alpaca para dados históricos de criptomoedas
        client = CryptoHistoricalDataClient()
        
        print(f"⏳ Iniciando coleta de dados para {SYMBOL}...")
        print(f"📅 Período: {START_DATE.date()} a {END_DATE.date()}")
        
        # Cria a requisição formatada para a API
        req = CryptoBarsRequest(
            symbol_or_symbols=[SYMBOL],
            timeframe=TIMEFRAME,
            start=START_DATE,
            end=END_DATE
        )
        
        # Executa a chamada à API
        bars = client.get_crypto_bars(req)
        
        # Verifica se a API retornou dados vazios
        if bars.df.empty:
            print("⚠️ Nenhum dado encontrado.")
            return None
            
        # Converte a resposta em um DataFrame do Pandas
        df = bars.df.reset_index()
        
        # Padroniza os nomes das colunas para minúsculo (facilita a manipulação)
        df.columns = [c.lower() for c in df.columns]
        
        # Cria o diretório 'data/' caso não exista
        os.makedirs("data", exist_ok=True)
        
        # Salva o DataFrame bruto em um arquivo CSV
        df.to_csv(DATA_PATH, index=False)
        print(f"✅ Dados salvos com sucesso em: {DATA_PATH}")
        print(f"📊 Total de registros coletados: {len(df)}")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"❌ Erro na coleta de dados: {e}")
        return None

if __name__ == "__main__":
    fetch_historical_data()

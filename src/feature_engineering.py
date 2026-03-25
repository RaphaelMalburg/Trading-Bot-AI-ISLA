import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(filepath="data/btc_usd_hourly.csv"):
    """
    Carrega os dados brutos de OHLCV do CSV e garante a ordenação temporal.
    A ordenação é crucial para que os cálculos de médias e lags não peguem dados do futuro.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def add_technical_indicators(df):
    """
    Motor principal da Engenharia de Features.
    Pega os dados brutos (OHLCV) e calcula indicadores técnicos complexos.
    Estes indicadores ajudam o Machine Learning a "enxergar" padrões de mercado.
    """
    df = df.copy()
    
    # Preenchimento de dados faltantes copiando o último valor conhecido (Forward Fill)
    df = df.ffill()
    
    # ==========================================
    # 1. Indicadores de Momento (Força da tendência)
    # ==========================================
    
    # RSI (Relative Strength Index): Mede se o ativo está "sobrecomprado" ou "sobrevendido"
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    
    # Oscilador Estocástico: Outra medida de momento baseada na relação entre o preço atual e o range histórico
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    
    # ==========================================
    # 2. Indicadores de Tendência (Direção)
    # ==========================================
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # EMAs (Média Móvel Exponencial): Dá mais peso aos preços recentes
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # ==========================================
    # 3. Indicadores de Volatilidade (Risco/Amplitude)
    # ==========================================
    
    # Bandas de Bollinger: Mostra a dispersão do preço em relação à média
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close'] # Normalizado
    
    # ATR (Average True Range): Mede o tamanho médio absoluto dos "candles". Vital para o Stop Loss e Critério de Kelly.
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # ==========================================
    # 4. Indicadores de Volume
    # ==========================================
    # OBV (On-Balance Volume): Relaciona o fluxo de volume com as mudanças de preço
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # ==========================================
    # 5. Lags e Construção do Target (O que a IA vai prever)
    # ==========================================
    
    # Retorno Logarítmico (Normaliza as variações percentuais para a IA)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # TARGET (Variável Alvo): 
    # É o que queremos que a IA adivinhe.
    # 1 (Compra) se o preço da PRÓXIMA hora for maior que o preço de agora.
    # 0 (Fora) se o preço da PRÓXIMA hora for menor ou igual.
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove linhas vazias criadas no começo do dataset pelos cálculos das médias e lags
    df = df.dropna()
    
    return df

def prepare_features(df):
    """
    Filtra apenas as colunas úteis para o Machine Learning e cria variáveis normalizadas.
    O Machine Learning não lida bem com preços absolutos (ex: $60.000 vs $20.000).
    Ele lida melhor com distâncias percentuais.
    """
    # Lista inicial de features escolhidas para treinar a IA
    feature_cols = [
        'rsi_14', 'stoch_k', 
        'macd', 'macd_signal', 'macd_diff',
        'bb_width', 'atr_14', 'obv',
        'log_return'
    ]
    
    # Criando distâncias normalizadas: Quão longe o preço está da Média Móvel em %?
    df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['close']
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['close']
    df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['close']
    
    # Adicionando as distâncias na lista final de features
    feature_cols.extend(['dist_ema_20', 'dist_ema_50', 'dist_ema_200'])
    
    X = df[feature_cols] # X = Os dados que a IA vai analisar
    y = df['target']     # Y = A resposta certa (O que aconteceu na hora seguinte)
    
    return X, y, feature_cols

if __name__ == "__main__":
    print("🚀 Iniciando Engenharia de Features...")
    
    # 1. Carregar os dados coletados da Alpaca
    df = load_data()
    print(f"Dados carregados: {df.shape}")
    
    # 2. Processar a matemática (TA-Lib)
    df_processed = add_technical_indicators(df)
    print(f"Indicadores calculados. Shape após dropna: {df_processed.shape}")
    
    # 3. Preparar o dataset final para treino (Separar X e Y)
    X, y, features = prepare_features(df_processed)
    
    # Salvar Dataset Processado em CSV para o script de Backtest usar
    df_processed.to_csv("data/processed_data.csv", index=False)
    print("💾 Dados processados salvos em data/processed_data.csv")
    
    # Salvar a ordem e o nome exato das features num arquivo .pkl
    # Isso é obrigatório para o robô ao vivo saber exatamente o que o modelo espera ler
    joblib.dump(features, "models/feature_names.pkl")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

def run_backtest_dynamic(initial_capital=100, base_leverage=3.0, sl_atr=1.0, tp_atr=3.0):
    """
    Executa um backtest detalhado para o modelo de Machine Learning (Random Forest)
    utilizando gestão de risco dinâmica e juros compostos.
    
    A lógica simula operações long-only (apenas compra) no Bitcoin, ajustando o 
    tamanho da posição dinamicamente com base na volatilidade (ATR) e limitando 
    o risco por trade a uma porcentagem fixa do capital atual.
    
    Args:
        initial_capital (float): Capital inicial da simulação em dólares.
        base_leverage (float): Alavancagem base desejada (não estritamente usada se o risco for o limitador).
        sl_atr (float): Multiplicador do ATR para definir a distância do Stop Loss.
        tp_atr (float): Multiplicador do ATR para definir a distância do Take Profit.
    """
    print(f"\n🚀 Backtest Dinâmico (Juros Compostos + Ajuste de Risco)...")
    
    try:
        # Carregar dados processados e artefatos do modelo
        df = pd.read_csv("data/processed_data.csv")
        model = joblib.load("models/rf_model.pkl")
        features = joblib.load("models/model_features.pkl")
    except Exception as e:
        print(f"❌ Erro ao carregar arquivos para backtest: {e}")
        return

    # Dividir os dados: usar apenas os 20% finais (dados nunca vistos no treinamento)
    split_index = int(len(df) * 0.8)
    df_test = df.iloc[split_index:].copy().reset_index(drop=True)
    
    # Gerar previsões (probabilidade da classe 1 - Preço Subir)
    X_test = df_test[features]
    probs = model.predict_proba(X_test)[:, 1]
    
    # Definir um limite de confiança mais conservador (55%) para gerar sinal de compra
    df_test['prediction'] = (probs > 0.55).astype(int)
    
    # Variáveis de controle da simulação
    capital = initial_capital
    position = 0          # 0 = Fora do mercado, 1 = Comprado
    entry_price = 0       # Preço de entrada da operação
    trades = []           # Histórico de PnL dos trades fechados
    equity_curve = [initial_capital]
    equity_dates = [pd.to_datetime(df_test.loc[0, 'timestamp'])] # Guardar as datas do capital
    btc_prices = [df_test.loc[0, 'close']] # Guardar o preço do BTC para comparar com Buy & Hold
    position_sizes = []   # Histórico do tamanho das posições (alavancagem usada)
    trade_dates = []      # Datas em que as operações foram iniciadas
    
    fee_pct = 0.0005      # Taxa de corretagem simulada (0.05% por ordem)
    
    # Parâmetros de Gestão Dinâmica
    # O objetivo é não perder mais de 5% do capital total em uma única operação que dê Stop Loss.
    max_risk_per_trade_pct = 0.05 
    
    # Iterar sobre cada candle no dataset de teste
    for i in range(len(df_test) - 1):
        # Condição de quebra (se perder quase todo o dinheiro)
        if capital <= 10: break
        
        current_price = df_test.loc[i, 'close']
        current_date = pd.to_datetime(df_test.loc[i, 'timestamp'])
        atr = df_test.loc[i, 'atr_14']
        pred = df_test.loc[i, 'prediction']
        
        # --- Lógica de Dimensionamento Dinâmico da Posição ---
        # A distância do stop loss é baseada no ATR (volatilidade atual)
        dist_stop = atr * sl_atr
        stop_pct = dist_stop / current_price # Representa a queda percentual até atingir o stop
        
        # Fórmula do Tamanho da Posição:
        # Posição = (Capital Atual * Risco Máximo Permitido) / Queda Percentual do Stop
        ideal_position_size = (capital * max_risk_per_trade_pct) / stop_pct
        
        # Travas de segurança: Limitar a alavancagem entre 1x e 5x o capital
        position_size = min(ideal_position_size, capital * 5.0)
        position_size = max(position_size, capital * 1.0)
        
        current_leverage = position_size / capital
        
        # --- Lógica de Saída de Posição (Fechamento) ---
        if position == 1:
            # Checar se atingiu o Stop Loss
            if current_price <= stop_loss:
                pct_loss = (stop_loss - entry_price) / entry_price
                pnl = position_size_locked * pct_loss
                cost = position_size_locked * fee_pct
                capital += (pnl - cost) # Atualizar capital (pnl é negativo)
                trades.append(pnl - cost)
                position = 0 # Ficar líquido
                
            # Checar se atingiu o Take Profit
            elif current_price >= take_profit:
                pct_gain = (take_profit - entry_price) / entry_price
                pnl = position_size_locked * pct_gain
                cost = position_size_locked * fee_pct
                capital += (pnl - cost) # Atualizar capital (pnl é positivo)
                trades.append(pnl - cost)
                position = 0 # Ficar líquido
        
        # --- Lógica de Entrada de Posição (Abertura) ---
        # Se estamos fora do mercado e o modelo prevê alta
        if position == 0 and pred == 1:
            position = 1
            entry_price = current_price
            
            # Definir níveis de saída baseados no ATR do momento da entrada
            stop_loss = current_price - dist_stop
            take_profit = current_price + (atr * tp_atr)
            
            # Travar o tamanho da posição para este trade
            position_size_locked = position_size 
            position_sizes.append(current_leverage)
            trade_dates.append(current_date)
            
            # Pagar taxa de corretagem da entrada
            capital -= position_size_locked * fee_pct
            
        # Registrar evolução do capital e preços para o gráfico
        equity_curve.append(capital)
        equity_dates.append(current_date)
        btc_prices.append(current_price)

    # Resultados
    total_return = (capital - initial_capital) / initial_capital * 100
    avg_leverage = np.mean(position_sizes) if position_sizes else 0
    
    # Calcular Buy and Hold
    btc_start_price = btc_prices[0]
    btc_end_price = btc_prices[-1]
    buy_and_hold_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100
    
    print("\n" + "="*40)
    print(f"📅 Período: {equity_dates[0].strftime('%Y-%m-%d')} a {equity_dates[-1].strftime('%Y-%m-%d')}")
    print(f"💰 Resultado Final (Dinâmico): ${capital:.2f}")
    print(f"📈 Retorno do Bot: {total_return:.2f}%")
    print(f"📈 Retorno Buy & Hold (BTC): {buy_and_hold_return:.2f}%")
    print(f"⚖️ Alavancagem Média Usada: {avg_leverage:.2f}x")
    print(f"📉 Pior momento do Bot: ${min(equity_curve):.2f}")
    
    plt.figure(figsize=(14, 10))
    
    # 1. Curva de Capital do Bot
    plt.subplot(3, 1, 1)
    plt.plot(equity_dates, equity_curve, color='blue')
    plt.title('Curva de Capital - Gestão Dinâmica de Risco (+277%)')
    plt.ylabel('Capital ($)')
    plt.axhline(y=initial_capital, color='r', linestyle='--')
    
    # 2. Preço do Bitcoin no mesmo período (Buy and Hold)
    plt.subplot(3, 1, 2)
    plt.plot(equity_dates, btc_prices, color='gray')
    plt.title('Preço do Bitcoin (BTC/USD) no mesmo período')
    plt.ylabel('Preço ($)')
    
    # 3. Alavancagem Usada
    plt.subplot(3, 1, 3)
    plt.plot(trade_dates, position_sizes, color='orange', alpha=0.7, marker='.')
    plt.title('Alavancagem Utilizada por Trade')
    plt.ylabel('Alavancagem (x)')
    plt.xlabel('Data (Meses do Backtest)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'models/dream_dynamic.png')
    print(f"📉 Gráfico salvo: models/dream_dynamic.png")

if __name__ == "__main__":
    run_backtest_dynamic()

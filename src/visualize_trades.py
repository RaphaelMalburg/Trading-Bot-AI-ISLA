import pandas as pd
import joblib
import mplfinance as mpf
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_trades(limit=20):
    """
    Gera imagens individuais de gráficos de candlestick detalhados para os trades 
    simulados pelo modelo Random Forest. 
    
    Isso é extremamente útil para "Visual Debugging" - permitindo que o desenvolvedor
    veja visualmente onde o robô decidiu entrar e porquê ele saiu (Stop Loss ou Take Profit),
    em vez de apenas olhar para métricas brutas.
    
    Args:
        limit (int): Número máximo de imagens de trades a serem geradas.
    """
    print("🎨 Gerando Imagens de Trades Detalhadas (Simulação)...")
    
    # 1. Carregar dados históricos e o modelo treinado
    try:
        df = pd.read_csv("data/processed_data.csv")
        # O mplfinance exige que o índice do DataFrame seja no formato Datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp') 
        
        model = joblib.load("models/rf_model.pkl")
        features = joblib.load("models/model_features.pkl")
    except Exception as e:
        print(f"❌ Erro ao carregar arquivos: {e}")
        return

    # Usar apenas o conjunto de teste (os 20% finais) para evitar visualizar dados viciados
    split_index = int(len(df) * 0.8)
    df_test = df.iloc[split_index:].copy()
    
    # Gerar previsões com probabilidade
    X_test = df_test[features]
    probs = model.predict_proba(X_test)[:, 1]
    
    # Limite de confiança para entrada (55%)
    df_test['prediction'] = (probs > 0.55).astype(int)
    
    trades_found = 0
    os.makedirs("reports/trade_samples", exist_ok=True)
    
    indices = df_test.index
    capital = 100.0 # Simulação de saldo inicial estático apenas para compor o título da imagem
    
    # Parâmetros de risco para calcular SL/TP (mesmos do bot em produção)
    sl_mult = 1.0
    tp_mult = 3.0
    
    # i = 40 para garantir que haja candles suficientes no passado para plotar o gráfico antes da entrada
    i = 40 
    while i < len(df_test) - 40:
        if trades_found >= limit:
            break
            
        current_idx = indices[i]
        row = df_test.loc[current_idx]
        
        # Sinal de Compra Identificado
        if row['prediction'] == 1:
            trades_found += 1
            
            entry_price = row['close']
            atr = row['atr_14']
            
            # Calcular os níveis exatos de saída com base na volatilidade do momento
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
            
            # --- Simular o andamento temporal do trade para encontrar o momento de saída ---
            exit_idx = None
            exit_price = None
            exit_reason = ""
            
            # Varrer os candles do futuro até atingir o alvo ou o stop
            for j in range(i + 1, len(df_test)):
                future_row = df_test.iloc[j]
                
                # Checar se a mínima (Low) do candle furou o Stop Loss
                if future_row['low'] <= stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "Stop Loss ❌"
                    break
                
                # Checar se a máxima (High) do candle atingiu o Take Profit
                if future_row['high'] >= take_profit:
                    exit_idx = j
                    exit_price = take_profit
                    exit_reason = "Take Profit ✅"
                    break
            
            # Se chegamos ao fim do dataset e o trade não fechou, ignora e continua
            if exit_idx is None:
                i += 1
                continue
                
            # Calcular lucro/prejuízo fictício (Alavancagem Fixa 5x)
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_usd = capital * pnl_pct * 5.0 
            capital += pnl_usd
            
            # --- Preparar Recorte do Gráfico (Crop) ---
            # Mostra 10 candles antes da entrada e 15 candles após a saída
            start_pos = max(0, i - 10)
            end_pos = min(len(df_test), exit_idx + 15)
            
            chart_data = df_test.iloc[start_pos:end_pos]
            
            # Inicializar arrays com NaNs (para que o mplfinance desenhe linhas/setas apenas nos pontos corretos)
            len_data = len(chart_data)
            ap_entry_line = [np.nan] * len_data
            ap_sl_line = [np.nan] * len_data
            ap_tp_line = [np.nan] * len_data
            ap_buy_marker = [np.nan] * len_data
            ap_sell_marker = [np.nan] * len_data
            
            # Índices relativos na janela recortada
            rel_entry = i - start_pos
            rel_exit = exit_idx - start_pos
            
            # Traçar as linhas horizontais de suporte e resistência do trade
            for k in range(rel_entry, min(rel_exit + 1, len_data)):
                ap_entry_line[k] = entry_price
                ap_sl_line[k] = stop_loss
                ap_tp_line[k] = take_profit
            
            # Adicionar Marcadores Visuais (Setas)
            # Seta Azul apontando para cima (Compra) levemente abaixo do preço
            ap_buy_marker[rel_entry] = entry_price * 0.99 
            
            # Seta Vermelha apontando para baixo (Venda da Posição) levemente acima do preço
            ap_sell_marker[rel_exit] = exit_price * 1.01
            
            # Construir título dinâmico com métricas da operação
            title = (f"Trade #{trades_found} | {current_idx.strftime('%Y-%m-%d %H:%M')}\n"
                     f"Entry: ${entry_price:.2f} | Exit: {exit_reason} (${exit_price:.2f})\n"
                     f"PnL: {pnl_pct*100:.2f}% | Balance: ${capital:.2f}")

            # Montar camadas adicionais do gráfico (AddPlots do mplfinance)
            plots = [
                # Linha de Entrada Amarela Sólida
                mpf.make_addplot(ap_entry_line, color='yellow', width=1.5, linestyle='-'),
                # Linhas de Saída Tracejadas (Vermelho=Stop, Verde=Alvo)
                mpf.make_addplot(ap_sl_line, color='red', width=1.0, linestyle='--'),
                mpf.make_addplot(ap_tp_line, color='green', width=1.0, linestyle='--'),
                # Markers Geométricos
                mpf.make_addplot(ap_buy_marker, type='scatter', markersize=100, marker='^', color='blue'),
                mpf.make_addplot(ap_sell_marker, type='scatter', markersize=100, marker='v', color='red')
            ]
            
            # Definir estilo estético inspirado no Yahoo Finance
            style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 9, 'figure.facecolor': '#f0f0f0'})
            
            # Nome do arquivo estruturado para fácil busca
            filename = f"reports/trade_samples/trade_{trades_found:02d}_{current_idx.date()}_{exit_reason.split()[0]}.png"
            
            # Renderizar e salvar imagem
            mpf.plot(
                chart_data,
                type='candle',
                style=style,
                title=title,
                ylabel='Preço (USD)',
                addplot=plots,
                volume=True,
                savefig=dict(fname=filename, dpi=100, pad_inches=0.25),
                tight_layout=True
            )
            print(f"📸 Trade {trades_found}: {exit_reason} -> {filename}")
            
            # Avançar o loop temporal para o candle após a saída deste trade (Simulando 1 trade por vez)
            i = exit_idx + 1
        else:
            i += 1

    print(f"\n✅ Concluído! {trades_found} imagens detalhadas geradas.")

if __name__ == "__main__":
    visualize_trades()

import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderStatus
from dotenv import load_dotenv
from datetime import datetime

# Adicionar diretório src ao path do sistema para permitir importações relativas
sys.path.append(os.getcwd())

# Carregar variáveis de ambiente (Chaves de API)
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Caminho do arquivo de saída do Dashboard
DASHBOARD_FILE = "dashboard.html"

def generate_dashboard():
    """
    Conecta-se à API da Alpaca, busca o saldo atual da conta e o histórico de ordens,
    e gera um arquivo HTML estático (dashboard.html) contendo métricas de performance
    e gráficos interativos utilizando a biblioteca Plotly.
    """
    print("📊 Gerando Dashboard...")
    
    try:
        # Inicializar cliente da Alpaca em modo simulação (paper=True)
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        
        # 1. Coleta de Dados da Conta
        equity = float(account.equity) # Patrimônio líquido total
        buying_power = float(account.buying_power) # Poder de compra disponível
        
        # Calcular Lucro/Prejuízo (PnL) do dia atual em dólares e porcentagem
        pnl = float(account.equity) - float(account.last_equity) 
        pnl_pct = (pnl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0
        
        # 2. Histórico de Ordens (Buscar as últimas 50 ordens fechadas)
        req = GetOrdersRequest(status="closed", limit=50, nested=True)
        orders = trading_client.get_orders(req)
        
        # Estruturar os dados das ordens em uma lista de dicionários
        trades_data = []
        for o in orders:
            trades_data.append({
                "symbol": o.symbol,
                "side": o.side,
                "qty": float(o.qty) if o.qty else 0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                "filled_at": o.filled_at,
                "status": o.status
            })
        
        # Converter para DataFrame do Pandas para facilitar a manipulação
        df_trades = pd.DataFrame(trades_data)
        
        # 3. Histórico de Portfólio
        # (Nesta versão simplificada, apenas registramos o ponto atual.
        # Em uma versão avançada, puxaríamos a curva de equity histórica completa da Alpaca)
        portfolio_dates = [datetime.now()]
        portfolio_equity = [equity]

        # --- GERAÇÃO DOS GRÁFICOS (Plotly) ---
        
        # Gráfico 1: Curva de Capital (Equity Curve)
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=portfolio_dates, y=portfolio_equity, mode='lines+markers', name='Equity'))
        fig_equity.update_layout(title="Curva de Capital (Tempo Real)", xaxis_title="Data", yaxis_title="Valor ($)")
        
        # Gráfico 2: Tabela de Distribuição de Trades Recentes
        if not df_trades.empty:
            fig_trades = go.Figure(data=[go.Table(
                header=dict(values=list(df_trades.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[df_trades.symbol, df_trades.side, df_trades.qty, df_trades.filled_avg_price, df_trades.filled_at, df_trades.status],
                           fill_color='lavender',
                           align='left'))
            ])
        else:
            # Fallback caso a conta seja nova e não tenha trades
            fig_trades = go.Figure()
            fig_trades.add_annotation(text="Nenhum trade fechado ainda.", showarrow=False)

        # --- GERAÇÃO HTML ---
        # Constrói o documento HTML injetando os valores processados e os gráficos do Plotly
        html_content = f"""
        <html>
        <head>
            <title>🤖 ML Trading Bot - Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                h1 {{ color: #333; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
                .stat-item {{ text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>🤖 ML Trading Bot Dashboard</h1>
            
            <div class="card">
                <h2>Status da Conta</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${equity:.2f}</div>
                        <div class="stat-label">Equity Total</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${buying_power:.2f}</div>
                        <div class="stat-label">Poder de Compra</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value {('positive' if pnl >= 0 else 'negative')}">${pnl:.2f}</div>
                        <div class="stat-label">PnL Hoje ($)</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value {('positive' if pnl_pct >= 0 else 'negative')}">{pnl_pct:.2f}%</div>
                        <div class="stat-label">PnL Hoje (%)</div>
                    </div>
                </div>
            </div>

            <div class="card">
                {fig_equity.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="card">
                <h2>Últimos Trades</h2>
                {fig_trades.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <p style="text-align: center; color: #999;">Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        # Salvar o HTML no disco
        with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"✅ Dashboard gerado com sucesso: {os.path.abspath(DASHBOARD_FILE)}")
        
    except Exception as e:
        print(f"❌ Erro ao gerar dashboard: {e}")

if __name__ == "__main__":
    generate_dashboard()

# ISLA Bot — Trading Algorítmico com Machine Learning

> **Projeto académico** — desenvolvido para avaliação de Engenharia de Software e IA (Março 2026).  
> Não é recomendação de investimento. Paper trading apenas.

**Grupo:** Raphael Malburg · André Neves · Vasco · Beatriz Ferreira

---

## Índice

1. [Descrição](#descrição)
2. [Resultados](#resultados)
3. [Arquitetura](#arquitetura)
4. [Pipeline de Decisão](#pipeline-de-decisão)
5. [Modelo de Machine Learning](#modelo-de-machine-learning)
6. [Gestão de Risco](#gestão-de-risco)
7. [Dashboard](#dashboard)
8. [Instalação e Configuração](#instalação-e-configuração)
9. [Como Executar](#como-executar)
10. [Estrutura do Projeto](#estrutura-do-projeto)
11. [API Endpoints](#api-endpoints)
12. [Aviso Legal](#aviso-legal)

---

## Descrição

O **ISLA Bot** é um sistema de trading algorítmico para BTC/USD que combina:

- **Machine Learning** (Random Forest) para prever sinais de entrada com base em 17 indicadores técnicos
- **Análise de sentimento** via LLM (Gemini Flash) para filtrar entradas em contexto de mercado muito bearish
- **Gestão de risco automática** com Stop-Loss/Take-Profit baseados em ATR, circuit breaker e exit watchdog
- **Dashboard web em tempo real** com gráficos interativos, métricas de ML e histórico de trades

O bot opera em **paper trading** via [Alpaca Markets](https://alpaca.markets/), sem risco financeiro real.

---

## Resultados

Backtest simulado walk-forward — **6 meses** (dados históricos reais 1H BTC/USD):

| Métrica | Valor |
|---|---|
| Retorno da estratégia | −0.50% |
| Buy & Hold BTC (mesmo período) | −10.74% |
| **Outperformance** | **+10.25%** |
| Win Rate | 52.2% |
| Total de trades | 23 |
| Profit Factor | 1.05 |
| Max Drawdown | −12.34% |
| Sharpe Ratio | −0.05 |

> O outperformance num período fortemente bearish resulta da **selectividade do modelo** — o bot fica maioritariamente em FLAT, evitando as perdas da queda do mercado.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                         ISLA Bot                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  src/data.py │ src/model.py │ src/strategy │  src/broker.py │
│  Fetch bars  │  RF Predict  │  Sizing / SL │  Alpaca API    │
│  Indicators  │  Load/Save   │  TP / Checks │  Orders        │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              src/trading.py  (orquestração)                  │
├─────────────────────────────────────────────────────────────┤
│   src/app.py  Flask · bot_loop · exit_watchdog · REST API   │
├───────────────────────────┬─────────────────────────────────┤
│  templates/dashboard.html │     data/trading_bot.db         │
│  Plotly.js · Polling 15s  │     SQLite · trades / runs      │
└───────────────────────────┴─────────────────────────────────┘
```

**Stack tecnológico:**

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.11 · Flask |
| Machine Learning | Scikit-learn · Random Forest |
| Exchange | Alpaca Markets SDK (paper trading) |
| Sentimento | OpenRouter API · Google Gemini Flash |
| Base de dados | SQLite |
| Frontend | Jinja2 · Plotly.js · JavaScript |
| Configuração | python-dotenv |

---

## Pipeline de Decisão

A cada fecho de candle de **1 hora**, o bot executa a seguinte sequência:

```
1. FETCH MARKET DATA
   └─ 90 dias de candles 1H BTC/USD via Alpaca (~2.160 barras)

2. FEATURE ENGINEERING (17 indicadores)
   └─ RSI-14, MACD, EMA-20/50, Bollinger Bands, ATR-14, OBV,
      Volume SMA ratio, Log return, Candle body ratio, EMA-20 slope

3. ML PREDICTION (Random Forest)
   └─ P(TP atingido antes do SL em 12H) ≥ 55% → LONG
      caso contrário → FLAT

4. DRIFT DETECTION
   └─ Confiança média das últimas 30 runs < 50% → aviso de drift

5. SENTIMENT GATE (opcional)
   └─ Titulares BTC avaliados pelo Gemini Flash (−1 a +1)
      Score ≤ −0.5 → bloqueia entrada

6. CIRCUIT BREAKER
   └─ Perda diária acumulada > 10% → para trading pelo resto do dia

7. POSITION SIZING
   └─ position_value = equity × 5% ÷ (ATR × 1.0 ÷ preço)

8. EXECUÇÃO
   └─ Market BUY + disaster stop 3×ATR na Alpaca
      Exit watchdog verifica SL/TP/max-hold a cada 10 segundos
```

---

## Modelo de Machine Learning

### Objetivo (target)

> *"O Take-Profit (2×ATR acima da entrada) será atingido **antes** do Stop-Loss (1×ATR abaixo) nos próximos 12 candles?"*

Este target está directamente alinhado com a lógica real de saída do bot — precisão = taxa de vitória nas posições abertas.

### Algoritmo

- **Random Forest Classifier** — 200 árvores, profundidade máxima 8, min. 3 amostras por folha
- **StandardScaler** aplicado a todos os features
- **Walk-forward validation** com `TimeSeriesSplit` (5 folds) para evitar data leakage temporal

### Métricas (holdout)

| Métrica | Valor | Notas |
|---|---|---|
| Accuracy | 67.91% | Alta porque maioria das barras são FLAT |
| Precision | 23.56% | Taxa de vitória real · break-even = 33.3% |
| Recall | 10.91% | % de oportunidades identificadas |
| F1 | 0.149 | Baixo — modelo muito conservador |

### Features mais importantes

| Feature | Importância |
|---|---|
| OBV (On-Balance Volume) | 11.9% |
| ATR-14 | 10.0% |
| EMA-20 | 9.6% |
| EMA-50 | 8.9% |
| BB High / Low / Width | ~8% cada |

### Treinar o modelo

```bash
python run_pipeline.py
```

---

## Gestão de Risco

| Mecanismo | Detalhe |
|---|---|
| **Stop-Loss** | 1×ATR abaixo da entrada · gerido pelo exit watchdog (10s) |
| **Take-Profit** | 2×ATR acima da entrada · rácio RR = 2:1 |
| **Disaster Stop** | Ordem stop-limit na Alpaca a 3×ATR · proteção se servidor cair |
| **Circuit Breaker** | Para trading se perda diária > 10% do capital inicial |
| **Max Hold** | Fecha posição após 12 candles (12H) independentemente do P&L |
| **Position Sizing** | Risco fixo 5% por trade · ajustado pela volatilidade (ATR) |
| **Sinal flip** | Fecha posição se o modelo mudar para FLAT no candle seguinte |

---

## Dashboard

Acesso: `http://localhost:5000`

**Funcionalidades:**
- Preço BTC e sinal actualizados a cada **15 segundos** (sem page refresh)
- Gráfico de velas interactivo com zoom/pan/scroll (estilo TradingView)
- Linhas de SL e TP sobrepostas no gráfico com etiqueta de preço ao vivo
- RSI (14) e MACD sincronizados
- Posição aberta com P&L não realizado em tempo real
- **Showcase 6 meses** — equity curve simulada + log de trades
- Métricas do modelo ML + gráfico de feature importance
- Kill switch para fechar todas as posições imediatamente

---

## Instalação e Configuração

### Requisitos

- Python 3.11+
- Conta Alpaca Markets (paper trading — gratuita)
- (Opcional) Chave OpenRouter para sentimento LLM

### 1. Clonar e instalar dependências

```bash
git clone <repo>
cd isla-bot-v2
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

Criar ficheiro `.env` na raiz do projecto:

```env
# Obrigatório — Alpaca paper trading
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Opcional — análise de sentimento
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxx

# Parâmetros do bot (valores por omissão)
TIMEFRAME_HOURS=1
CONFIDENCE_THRESHOLD=0.55
SENTIMENT_FLOOR=-0.5
SL_ATR_MULT=1.0
TP_ATR_MULT=2.0
MAX_RISK_PER_TRADE=0.05
DAILY_LOSS_LIMIT_PCT=0.10
MAX_HOLD_CANDLES=12
```

### 3. Treinar o modelo

```bash
python run_pipeline.py
```

### 4. (Opcional) Gerar showcase 6 meses

```bash
python run_showcase_backtest.py
```

---

## Como Executar

### Iniciar o bot + dashboard

```bash
python main.py
```

O servidor inicia em `http://localhost:5000`.  
O bot loop corre em background e executa a cada fecho de candle de 1H.

### Apenas backtest de avaliação

```bash
python run_backtest_eval.py
```

### Teste Kelly Criterion vs Fixed Fraction

```bash
python run_kelly_test.py
```

---

## Estrutura do Projeto

```
isla-bot-v2/
├── main.py                      # Entry point
├── run_pipeline.py              # Treino do modelo + backtest
├── run_backtest_eval.py         # Backtest com métricas detalhadas
├── run_showcase_backtest.py     # Backtest 6 meses para showcase
├── run_kelly_test.py            # Comparação Kelly vs Fixed Fraction
│
├── src/
│   ├── app.py                   # Flask server + bot loop + exit watchdog
│   ├── trading.py               # Pipeline de decisão principal
│   ├── data.py                  # Fetch de dados + feature engineering
│   ├── model.py                 # Treino, predição, drift detection
│   ├── strategy.py              # Position sizing, SL/TP, circuit breaker
│   ├── broker.py                # Wrapper Alpaca API
│   ├── sentiment.py             # Análise de sentimento via LLM
│   ├── database.py              # SQLite — trades e runs
│   ├── backtest.py              # Motor de backtest walk-forward
│   └── config.py                # Settings (dataclass + .env)
│
├── templates/
│   ├── _base.html               # Layout base
│   ├── dashboard.html           # Dashboard principal
│   └── backtest.html            # Página de backtest
│
├── models/                      # Artefactos ML (gerados)
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── model_features.pkl
│   ├── ml_metrics.json
│   ├── showcase_results.json
│   └── backtest_results.json
│
├── data/                        # Base de dados e dados históricos
│   └── trading_bot.db
│
├── logs/                        # Ficheiros de log
├── presentation.html            # Apresentação do projecto
├── README.md                    # Este ficheiro
└── .env                         # Variáveis de ambiente (não commitar)
```

---

## API Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Dashboard principal |
| GET | `/health` | Health check |
| GET | `/api/latest` | Último ciclo do bot |
| GET | `/api/live_stats` | Equity, posições e preço BTC em tempo real |
| GET | `/api/dashboard_data` | Dados completos para polling |
| GET | `/api/trades` | Histórico de trades (paginado) |
| GET | `/api/trades.csv` | Export CSV de todos os trades |
| GET | `/api/showcase` | Resultados do backtest 6 meses |
| POST | `/api/kill_switch` | Fecha todas as posições imediatamente |
| POST | `/api/run_showcase` | Regenera o backtest showcase em background |

---

## Aviso Legal

Este projecto foi desenvolvido exclusivamente para fins académicos no âmbito da disciplina de Engenharia de Software e IA.

- **Não é recomendação de investimento**
- Utiliza exclusivamente **paper trading** (dinheiro simulado, sem risco real)
- Os resultados de backtest são **simulados** e não garantem performance futura
- Os autores não são responsáveis por qualquer decisão de investimento tomada com base neste projecto

---

*ISLA Bot · Engenharia de Software e IA · Março 2026*  
*Raphael Malburg · André Neves · Vasco · Beatriz Ferreira*

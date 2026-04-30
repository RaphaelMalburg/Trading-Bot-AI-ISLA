# ML Trading Bot - Crypto & Multi-Asset

Este projeto é um sistema de trading algorítmico autônomo para criptomoedas (BTC/USD), utilizando **Machine Learning (Random Forest)**, **Análise de Sentimento (LLMs via Google Gemini)** e **Gestão de Risco Dinâmica (Critério de Kelly Adaptado)**.

## 🚀 Status do Projeto
- **Ativo Principal:** BTC/USD (H1)
- **Estratégia Atual:** Análise Multi-Ativos (Correlação BTC x ETH)
- **Performance (Backtest 2023):** +277.67% (Alpha massivo sobre o Buy & Hold)
- **Acurácia do Modelo:** ~54% (Random Forest)

## 🧠 Inteligência do Sistema

### 1. Análise Técnica & Quantitativa
Utiliza um modelo **Random Forest** treinado em dados históricos de preço e volume, enriquecidos com indicadores técnicos (RSI, MACD, Bollinger Bands, ATR).
*   **Multi-Asset:** O modelo monitora o Ethereum (ETH) para identificar correlações e movimentos antecipados que impactam o Bitcoin.

### 2. Análise Fundamentalista (Sentiment Analysis)
Integração com o modelo LLM **`google/gemini-2.5-flash-lite` (via OpenRouter)** para analisar manchetes de notícias financeiras em tempo real e gerar um "Score de Sentimento" (-1 a +1), atuando como um filtro contra operações em momentos de pânico de mercado.

### 3. Gestão de Risco (Dynamic Position Sizing)
- **Stop Loss / Take Profit:** Dinâmicos, baseados na volatilidade atual do ativo (ATR).
- **Fractional Kelly Criterion:** Ajusta o tamanho da posição/alavancagem com base no risco, maximizando o crescimento geométrico da conta e protegendo o capital contra *drawdowns*.

## ✨ Melhorias Recentes (v2.0)

### 🔐 Infrastructure & Reliability
- **Configuração Centralizada** (Pydantic Settings) — validação de variáveis de ambiente.
- **Graceful Shutdown** — limpeza de posições ao receber SIGTERM/SIGINT.
- **Circuit Breaker** — limite de perda diária configurável (default -10%).
- **Model Drift Detection** — monitora confiança das predições para alertar quando retreinar.

### 📊 ML Engineering
- **StandardScaler Persistente** — normalização adequada de features (sem data leakage).
- **Walk-Forward Validation** — backtest com retreinamento em janela rolante (robustez acadêmica).
- **SHAP Explainability** — interpretabilidade do modelo (feature importance global/local).
- **Calibration Curve** — verifica se confiança preditiva está bem calibrada.
- **Model Comparison** — framework para comparar RF vs XGBoost vs LSTM.

### 🧪 Testes
- **Testes Unitários** (pytest) para feature engineering, position sizing, database.
- **Testes de API** (Flask test client) para endpoints.
- **Testes de Integração** com fixtures de dados sintéticos.

### 📈 Dashboard
- **Card de Performance do Modelo** — acurácia out-of-sample, precision, recall.
- **Alertas de Drift** — banner amarelo quando confiança cai.
- **API `/api/charts`** agora inclui `ml_metrics` no JSON.

## 🛠️ Arquitetura do Código

- `src/data_ingestion.py`: Ingestão de dados históricos OHLCV via Alpaca API.
- `src/feature_engineering.py`: Criação de indicadores técnicos usando `ta-lib`.
- `src/model_training.py`: Pipeline de treinamento com normalização e scaler.
- `src/sentiment_analysis.py`: Integração com LLM para análise qualitativa de notícias.
- `src/trading_bot_multi.py`: Robô principal que executa a lógica de trade 24/7.
- `src/backtest_dynamic.py`: Simulação da curva de capital com gestão de risco dinâmica.
- `src/plot_ml_metrics.py`: Gera SHAP, confusion matrix, calibration curve.
- `src/compare_models.py`: compara RF, XGBoost, LSTM.
- `src/config.py`: Configuração centralizada com Pydantic.

## 📊 Como Rodar

1. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar Credenciais:**
   Renomeie o `.env.example` para `.env` e insira suas chaves da Alpaca e OpenRouter.

3. **Treinar Modelo (inicial):**
   ```bash
   python src/model_training.py
   # Isso gera: models/rf_model.pkl, models/scaler.pkl, models/model_features.pkl
   ```

4. **Gerar Métricas de ML (SHAP, matriz de confusão):**
   ```bash
   python src/plot_ml_metrics.py
   ```

5. **Executar Backtest:**
   ```bash
   python src/backtest_dynamic.py              # single-split
   python src/backtest_dynamic.py --walkforward  # walk-forward
   ```

6. **Comparar Modelos (opcional):**
   ```bash
   python src/compare_models.py   # Requer xgboost e torch instalados
   ```

7. **Executar o Bot (Live/Paper Trading):**
   ```bash
   python src/main.py
   ```

8. **Visualizar a Dashboard:**
   Abra `http://localhost:5000` no navegador.

## 🧪 Testes

```bash
pytest tests/ -v
```

Cobertura atual:
- `test_feature_engineering.py` — 6 testes (indicadores, alvo, normalização)
- `test_database.py` — 4 testes (ledger, estatísticas, drift)
- `test_api.py` — 10 testes (endpoints Flask)

## 📚 Documentação

- `DOCUMENTACAO_TECNICA.md` — fundamentação científica.
- `presentation.html` — slides do projeto.
- `methodology.html` — pipeline detalhado (acessível via `/methodology`).

## ⚠️ Disclaimer

**Projeto acadêmico** — desenvolvido para avaliação de Engenharia de Software e IA (Março 2026).
Não é recomendação de investimento. Paper trading only.

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

## 🛠️ Arquitetura do Código

- `src/data_ingestion.py`: Ingestão de dados históricos OHLCV via Alpaca API.
- `src/feature_engineering.py`: Criação de indicadores técnicos usando `ta-lib`.
- `src/model_training.py`: Pipeline de treinamento do modelo Random Forest.
- `src/sentiment_analysis.py`: Integração com LLM para análise qualitativa de notícias.
- `src/trading_bot_multi.py`: Robô principal que executa a lógica de trade 24/7.
- `src/backtest_dynamic.py`: Script para simulação da curva de capital com gestão de risco dinâmica.
- `presentation.html`: Apresentação completa sobre o funcionamento técnico do projeto.

## 📊 Como Rodar

1. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar Credenciais:**
   Renomeie o `.env.example` para `.env` e insira suas chaves da Alpaca e OpenRouter.

3. **Executar o Bot (Live/Paper Trading):**
   ```bash
   python src/main.py
   ```

4. **Visualizar a Apresentação:**
   Abra o arquivo `presentation.html` em seu navegador.

---
**Autores:** Raphael Malburg, Vasco e André Neves
**Contexto:** Projeto Final de Engenharia de Software e IA (Março 2026)

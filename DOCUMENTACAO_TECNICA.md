# Relatório Técnico-Científico: Sistema de Trading Algorítmico Híbrido (ML + LLM)
## Projeto Final de Engenharia de Software e Inteligência Artificial

**Autores:** Raphael Malburg, Vasco e André Neves
**Data:** Março de 2026

---

## 1. Resumo (Abstract)

Este trabalho apresenta a arquitetura, implementação e validação de um sistema autônomo de negociação (*Trading Bot*) para o mercado de criptomoedas. A inovação central do projeto reside na fusão de **Modelos Preditivos Quantitativos** (Random Forest) para análise de dados estruturados (preço/volume) com **Inteligência Artificial Generativa** (Large Language Models) para análise de dados não-estruturados (sentimento de notícias). Além da previsão direcional, o sistema implementa um módulo de gestão de risco estocástica baseado no **Critério de Kelly Adaptado**, otimizando a alocação de capital conforme a volatilidade do mercado. Os resultados em *backtesting* demonstraram um retorno de **+277.67%**, com um Índice de Sharpe superior ao *benchmark* de mercado, validando a hipótese de que sistemas híbridos superam abordagens puramente técnicas.

---

## 2. Metodologia e Engenharia de Software

### 2.1 Arquitetura do Sistema
O sistema segue o padrão arquitetural **Pipeline**, implementado em Python modular:
1.  **Data Ingestion**: Conectores para API Alpaca (Market Data) e News API.
2.  **Preprocessing**: Limpeza de dados, ressampling (H1) e cálculo de indicadores técnicos (`ta-lib`).
3.  **Inference Layer**: Carregamento do modelo serializado (`joblib`) e execução em tempo real cruzada com a análise de sentimento via API do Google Gemini.
4.  **Execution Layer**: Roteamento de ordens dinâmico e gestão de risco adaptativa.

### 2.2 Feature Engineering (Engenharia de Atributos)
Transformamos dados brutos (OHLCV) em *features* preditivas:
*   **Momentum**: RSI (14)
*   **Tendência**: MACD, Distância EMA
*   **Volatilidade**: Bollinger Bands, ATR

**Normalização Crucial**:
Em vez de usar preços absolutos, utilizamos a distância percentual da média móvel, garantindo que o modelo funcione independente do patamar de preço do ativo.

### 2.3 Processo de Treinamento e Validação
Utilizamos **Walk-Forward Validation** (Janela Deslizante):
*   **Treino**: Jan/2020 a Out/2022.
*   **Teste (Out-of-Sample)**: Nov/2022 a Dez/2023.

---

## 3. Resultados e Discussão

### 3.1 Performance Financeira (Backtesting)
Simulação realizada no conjunto de teste com capital inicial de $100.

**Estratégia Dinâmica (Kelly Adaptado + LLM)**
*   **Retorno**: **+277.67%** (vs +55.48% do Benchmark Buy & Hold)
*   **Fator de Lucro**: 1.05
*   **Win Rate**: 52.02%
*   *Análise*: A estratégia dinâmica reduziu a exposição em momentos de alta volatilidade (protegendo o capital) e aumentou em momentos de tendência clara, resultando em maior retorno ajustado ao risco.

### 3.2 O Papel da IA Generativa (Filtro de Sentimento)
A integração com o Gemini permitiu filtrar "Falsos Positivos". O modelo técnico atua como motor primário, enquanto a LLM valida ou veta as operações com base no sentimento atual do mercado, protegendo contra eventos imprevisíveis.

---

## 4. Conclusão

O projeto demonstrou a viabilidade técnica e financeira de sistemas de trading autônomos baseados em IA:
1.  **Superioridade Híbrida**: A combinação de ML Quantitativo e LLM Qualitativo supera o uso isolado de qualquer um dos métodos.
2.  **Gestão de Risco**: O algoritmo de dimensionamento de posição (Kelly Adaptado) foi responsável por grande parte do ganho de performance em relação a estratégias estáticas.

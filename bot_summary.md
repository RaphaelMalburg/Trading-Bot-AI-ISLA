# 🤖 Resumo do ML Trading Bot (ISLA)

Este documento foca na arquitetura e nas lógicas de **Machine Learning** que movem o bot de trading quantitativo. Todo o código não-ML (integração com corretora, dashboard, etc.) foi omitido para destacar o "cérebro" do sistema.

## 1. Visão Geral do Pipeline de ML
1. **Coleta:** O bot consome velas (OHLCV) históricas.
2. **Engenharia de Features:** Calcula volatilidade, tendência e momentum (RSI, MACD, BB, ATR).
3. **Sentimento (Opcional):** Um LLM converte notícias em um score numérico (-1.0 a 1.0).
4. **Predição (XGBoost):** O modelo avalia o array final de features e retorna a probabilidade de alta.
   * **Nota de Arquitetura (XGBoost vs Random Forest):** O bot utiliza o Extreme Gradient Boosting (XGBoost). Diferente do Random Forest (que cria árvores independentes simultaneamente e tira a média), o XGBoost constrói as árvores de decisão de forma *sequencial* — cada nova árvore é focada em corrigir especificamente os erros matemáticos (resíduos) da árvore anterior. Isso o torna excepcionalmente preciso, rápido e ideal para dados financeiros estruturados.
5. **Decisão:** Se a probabilidade de alta ultrapassar um limiar configurado (`CONFIDENCE_THRESHOLD`), o sinal é disparado.

---

## 2. Lógica de Machine Learning (Snippets)

### A. Construção das Features (Entrada do Modelo)
O modelo exige que as features estejam na mesma ordem exata do treinamento. Elas normalizam a ação de preço e medem a força do mercado.

```python
# src/trading_bot_multi.py
# Criação do array de features da última vela fechada
current_features = [
    float(df['RSI'].iloc[-1]),                  # Momentum
    float(df['MACD'].iloc[-1]),                 # Tendência de curto prazo
    float(df['MACD_Signal'].iloc[-1]),          # Tendência de longo prazo
    float(df['close'].iloc[-1] - df['EMA_20'].iloc[-1]), # Distância da média
    float(df['BB_Width'].iloc[-1]),             # Compressão/Expansão da Volatilidade
    float(df['ATR'].iloc[-1]),                  # Volatilidade bruta
    float(sentiment_score)                      # Contexto macro/notícias
]

# Conversão para o formato otimizado do XGBoost
import xgboost as xgb
dmatrix = xgb.DMatrix([current_features])
```

### B. Carregamento do Modelo e Predição
O bot carrega o arquivo pré-treinado `.json` do XGBoost e avalia a probabilidade de o preço subir na próxima hora.

```python
# src/trading_bot_multi.py
import xgboost as xgb

# 1. Carrega o modelo treinado offline
model = xgb.Booster()
model.load_model("models/xgboost_model.json")

# 2. Faz a predição da probabilidade (0.0 a 1.0)
probabilities = model.predict(dmatrix)
prob_up = float(probabilities[0])

# 3. Classificação baseada no Threshold Otimizado (ex: 0.55)
CONFIDENCE_THRESHOLD = 0.55
action = "BUY" if prob_up >= CONFIDENCE_THRESHOLD else "HOLD/SELL"
```

### C. Treinamento do Modelo (Pipeline de Treino)
O modelo aprende a classificar se a próxima vela terá retorno positivo (1) ou negativo (0) com base no histórico passado.

```python
# src/train_model.py (Pseudo-código do processo de treino)

# 1. Criação do Target (O que o modelo deve prever)
# Se o fechamento da próxima vela for maior que a atual, Target = 1
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 2. Divisão de Treino e Teste (sem vazamento de dados futuros)
X = df[feature_columns]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Definição dos Hiperparâmetros do XGBoost
params = {
    'objective': 'binary:logistic', # Saída de probabilidade
    'eval_metric': 'logloss',       # Métrica de erro
    'max_depth': 4,                 # Profundidade da árvore (evita overfitting)
    'learning_rate': 0.05,          # Taxa de aprendizado
    'subsample': 0.8                # Amostragem para robustez
}

# 4. Treinamento
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, num_boost_round=100)

# 5. Salvando o "cérebro"
model.save_model("models/xgboost_model.json")
```

## 3. Notas de Otimização Recentes
- **Backtest Dinâmico:** Foi comprovado que aumentar o `CONFIDENCE_THRESHOLD` para >0.60 filtra muito ruído, mas corta drasticamente a rentabilidade do juro composto. O "sweet spot" atual (Treino de 15/Abr/2026) se encontra em **0.55**.
- **Model Drift:** A probabilidade atual gravita em ~57%. Se a acurácia cair, o pipeline de treino (`train_model.py`) precisará ser re-executado com a safra de dados mais recente para recalibrar as árvores de decisão.

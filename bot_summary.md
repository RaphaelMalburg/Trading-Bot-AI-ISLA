# 🤖 Resumo do ML Trading Bot (ISLA)

Este documento foca na arquitetura e nas lógicas de **Machine Learning** que movem o bot de trading quantitativo. Todo o código não-ML (integração com corretora, dashboard, etc.) foi omitido para destacar o "cérebro" do sistema.

## 1. Visão Geral do Pipeline de ML
1. **Coleta:** O bot consome velas (OHLCV) históricas.
2. **Engenharia de Features:** Calcula volatilidade, tendência e momentum (RSI, MACD, BB, ATR).
3. **Sentimento (Opcional):** Um LLM converte notícias em um score numérico (-1.0 a 1.0).
4. **Predição (Random Forest):** O modelo avalia o array final de features e retorna a probabilidade de alta.
    * **Nota de Arquitetura (Random Forest):** O bot utiliza Random Forest que cria múltiplas árvores de decisão durante o treinamento e retorna a média das predições dessas árvores. Este método reduz overfitting através da média das previsões e é robusto para dados financeiros com ruído.
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
```

### B. Carregamento do Modelo e Predição
O bot carrega o arquivo pré-treinado do Random Forest e avalia a probabilidade de o preço subir na próxima hora.

```python
# src/trading_bot_multi.py
import joblib

# 1. Carrega o modelo treinado offline
model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/model_features.pkl")

# 2. Faz a predição da probabilidade (0.0 a 1.0) - classe 1 é "alta"
probabilities = model.predict_proba([current_features])[0]
prob_up = probabilities[1]  # Probabilidade da classe 1 (alta)

# 3. Classificação baseada no Threshold Otimizado (ex: 0.55)
CONFIDENCE_THRESHOLD = 0.55
action = "BUY" if prob_up >= CONFIDENCE_THRESHOLD else "HOLD/SELL"
```

### C. Treinamento do Modelo (Pipeline de Treino)
O modelo aprende a classificar se a próxima vela terá retorno positivo (1) ou negativo (0) com base no histórico passado.

```python
# src/model_training.py (Processo de treino real)

# 1. Criação do Target (O que o modelo deve prever)
# Se o fechamento da próxima vela for maior que a atual, Target = 1
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 2. Engenharia de Features (RSI, MACD, EMAs, BB, ATR, etc.)
# Features já calculadas em src/feature_engineering.py

# 3. Divisão de Treino e Teste (sem vazamento de dados futuros)
# Usando 80% para treino, 20% para teste (dados temporais não embaralhados)
split_index = int(len(df) * 0.8)
X_train = df[feature_cols].iloc[:split_index]
X_test = df[feature_cols].iloc[split_index:]
y_train = df['Target'].iloc[:split_index]
y_test = df['Target'].iloc[split_index:]

# 4. Definição dos Hiperparâmetros do Random Forest
model = RandomForestClassifier(
    n_estimators=100,        # Número de árvores
    max_depth=10,            # Profundidade máxima das árvores
    min_samples_leaf=5,      # Mínimo de amostras por folha
    random_state=42,         # Para reproducibilidade
    n_jobs=-1                # Usar todos os cores disponíveis
)

# 5. Treinamento
model.fit(X_train, y_train)

# 6. Avaliação e Salvando o modelo
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Salvando o modelo e lista de features
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(feature_cols, "models/model_features.pkl")
```

## 3. Notas de Otimização Recentes
- **Backtest Dinâmico:** Foi comprovado que aumentar o `CONFIDENCE_THRESHOLD` para >0.60 filtra muito ruído, mas corta drasticamente a rentabilidade do juro composto. O "sweet spot" atual (Treino de 15/Abr/2026) se encontra em **0.55**.
- **Model Drift:** A probabilidade atual gravita em ~57%. Se a acurácia cair, o pipeline de treino (`train_model.py`) precisará ser re-executado com a safra de dados mais recente para recalibrar as árvores de decisão.

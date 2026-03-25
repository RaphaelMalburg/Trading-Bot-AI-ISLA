import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    """
    Motor de Treinamento do Machine Learning.
    Carrega os dados com os indicadores matemáticos e treina o algoritmo Random Forest
    para prever se a próxima hora será de alta (1) ou baixa (0).
    """
    # 1. Carregar dados processados (contém OHLCV + Indicadores Técnicos)
    print("📂 Carregando dados processados...")
    df = pd.read_csv("data/processed_data.csv")
    
    # 2. Definir Features (X) e Target (y)
    # Excluir colunas que são textos ou preços brutos que a IA não deve usar diretamente
    drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    
    # As features são todas as colunas que restam (RSI, MACD, ATR, distâncias das EMAs, etc.)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols] # Matriz de dados de entrada
    y = df['target']     # Vetor de respostas corretas
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 3. Divisão Treino/Teste (Walk-Forward Validation)
    # Importante: Em séries temporais (finanças), NÃO podemos embaralhar os dados.
    # O modelo precisa aprender com o passado (primeiros 80%) e provar que funciona no futuro (últimos 20%).
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # 4. Instanciar e Treinar o Modelo
    print("🧠 Treinando Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,      # Cria 100 árvores de decisão em paralelo
        max_depth=10,          # Profundidade máxima de cada árvore para evitar decorar os dados (overfitting)
        min_samples_leaf=5,    # Mínimo de exemplos por folha para generalizar melhor
        random_state=42,       # Semente para garantir que o resultado seja sempre igual
        n_jobs=-1              # Usa todos os núcleos do processador do computador
    )
    # Aqui a mágica acontece: o algoritmo estuda as features para tentar acertar o 'y'
    model.fit(X_train, y_train)
    
    # 5. Avaliação do Modelo (Prova de Fogo)
    print("📊 Avaliando Modelo no ambiente de Teste (Out-of-Sample)...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Acurácia no Teste: {acc:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 6. Importância das Features (Feature Importance)
    # Descobre o que o robô considerou mais importante na hora de tomar a decisão
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    print("\nTop 5 Features Mais Importantes:")
    print(feature_imp.head(5))
    
    # 7. Exportar a Inteligência Artificial
    # Salva o "cérebro" treinado em um arquivo para que o robô de trade consiga usar sem precisar treinar de novo
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(feature_cols, "models/model_features.pkl")
    print("\n💾 Modelo salvo em models/rf_model.pkl")

if __name__ == "__main__":
    train_model()

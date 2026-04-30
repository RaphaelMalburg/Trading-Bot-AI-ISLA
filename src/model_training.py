import logging
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def train_model():
    """
    ML training pipeline.
    Loads the feature-engineered dataset and trains a Random Forest classifier
    to predict whether the next hour's BTC close will be higher (1) or lower/flat (0).
    """
    # 1. Load processed data (OHLCV + technical indicators)
    logger.info("Loading processed data...")
    df = pd.read_csv("data/processed_data.csv")

    # 2. Define features (X) and target (y)
    # Drop raw price/text columns the model should not see directly
    drop_cols = ['timestamp', 'symbol', 'target', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df['target']

    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    # 3. Walk-forward validation: train on first 80%, test on last 20%.
    # Time-series data must NOT be shuffled.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # 4. Fit the Random Forest
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Out-of-sample evaluation
    logger.info("Evaluating model out-of-sample...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 6. Feature importance
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)

    print("\nTop 5 most important features:")
    print(feature_imp.head(5))

    # 7. Persist the trained model + feature list for the live bot
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(feature_cols, "models/model_features.pkl")
    logger.info("Model saved to models/rf_model.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_model()

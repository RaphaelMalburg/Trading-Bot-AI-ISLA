import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load data
df = pd.read_csv("data/processed_data.csv")
drop_cols = [
    "timestamp",
    "symbol",
    "target",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["target"]

# Split as in training (80/20)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Load model
model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/model_features.pkl")
# Ensure feature order
X_test = X_test[features]

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy bar
axes[0].bar(["Accuracy"], [acc], color="skyblue")
axes[0].set_ylim(0, 1)
axes[0].set_title(f"Model Accuracy: {acc:.4f}")
axes[0].set_ylabel("Score")

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Feature importance
importances = model.feature_importances_
feat_imp = pd.DataFrame({"Feature": features, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)
axes[2].barh(feat_imp["Feature"], feat_imp["Importance"], color="teal")
axes[2].set_title("Top 10 Feature Importances")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("models/ml_metrics.png", dpi=150)
print("Saved ML metrics plot to models/ml_metrics.png")
plt.show()

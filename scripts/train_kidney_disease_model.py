"""
Chronic Kidney Disease (CKD) Random Forest Model Training
UCI ML Repository benchmark
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_utils import ensure_kidney_data

print("=" * 60)
print("KIDNEY DISEASE MODEL TRAINING")
print("=" * 60)

df = ensure_kidney_data()
print(f"[+] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

df = df.replace('?', np.nan).dropna()

for col in df.columns:
    if col != df.columns[-1] and (df[col].dtype == object or isinstance(df[col].iloc[0], str)):
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.iloc[:, :-1].copy()
y_raw = df.iloc[:, -1].astype(str).str.strip().str.lower()
y = np.where(y_raw.str.contains('ckd') & ~y_raw.str.contains('not'), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

feature_importance = dict(zip(X.columns, model.feature_importances_))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

print("\nTop 5 Important Features:")
for feature, importance in feature_importance.items():
    print(f"  {feature}: {importance:.4f}")

metrics = {
    'disease': 'Kidney Disease',
    'dataset': 'UCI Chronic Kidney Disease (CKD)',
    'samples': df.shape[0],
    'features': X.shape[1],
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'auc_roc': round(auc, 4),
    'top_features': {k: round(float(v), 4) for k, v in feature_importance.items()}
}

out_file = os.path.join(os.path.dirname(__file__), 'kidney_disease_metrics.json')
with open(out_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n[+] Metrics saved to: {os.path.basename(out_file)}")
print("=" * 60)

"""
MedAssist AI - Model Training Script
=====================================
Trains a Random Forest classifier on the healthcare dataset.
Run this once to generate model.pkl before launching the Streamlit app.

Usage:
    python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("📂 Loading dataset...")
data = pd.read_csv("healthcare_dataset_5000_people.csv")

print(f"   ✅ Rows: {len(data)}")
print(f"   ✅ Symptoms: {len(data.columns) - 1}")
print(f"   ✅ Diseases: {data['Disease'].nunique()}")

# ── 2. Prepare Features & Labels ─────────────────────────────────────────────
X = data.drop("Disease", axis=1)
y = data["Disease"]

symptoms = list(X.columns)  # save symptom order for prediction

# ── 3. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 4. Train Random Forest ───────────────────────────────────────────────────
print("\n🌲 Training Random Forest (200 trees)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1        # use all CPU cores
)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ── 6. Save Model & Symptom List ─────────────────────────────────────────────
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(symptoms, open("symptoms.pkl", "wb"))

print("\n✅ Saved: model.pkl")
print("✅ Saved: symptoms.pkl")
print("\n🚀 Ready! Now run:  streamlit run app.py")

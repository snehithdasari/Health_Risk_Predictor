"""
Train ML models for Health Risk Prediction.
Generates synthetic health data and trains Random Forest, XGBoost, and SVM
for Diabetes, Hypertension, and Heart Disease prediction.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Ensure ml_models directory exists
os.makedirs('ml_models', exist_ok=True)

np.random.seed(42)
N = 10000


def generate_synthetic_data(n=N):
    """Generate synthetic health data with realistic correlations."""
    data = {}

    # Demographics
    data['age'] = np.random.randint(18, 85, n)
    data['gender'] = np.random.choice([0, 1], n)  # 0=Female, 1=Male

    # Health metrics
    data['bmi'] = np.round(np.random.normal(26, 5, n).clip(15, 50), 1)
    data['blood_pressure_systolic'] = np.random.randint(90, 200, n)
    data['blood_pressure_diastolic'] = np.random.randint(50, 130, n)
    data['cholesterol'] = np.random.randint(120, 350, n)
    data['glucose'] = np.random.randint(60, 300, n)

    # Lifestyle factors (encoded)
    data['smoking'] = np.random.choice([0, 1, 2], n)  # 0=Non, 1=Former, 2=Current
    data['alcohol'] = np.random.choice([0, 1, 2], n)  # 0=None, 1=Moderate, 2=Heavy
    data['physical_activity'] = np.random.choice([0, 1, 2], n)  # 0=Low, 1=Moderate, 2=High
    data['family_history'] = np.random.choice([0, 1, 2, 3], n)  # 0=None, 1=Diabetes, 2=Heart, 3=Hypertension

    df = pd.DataFrame(data)

    # --- Generate realistic target labels based on risk factors ---

    # Diabetes risk
    diabetes_score = (
        (df['age'] > 45).astype(float) * 0.2 +
        (df['bmi'] > 30).astype(float) * 0.25 +
        (df['glucose'] > 140).astype(float) * 0.3 +
        (df['physical_activity'] == 0).astype(float) * 0.1 +
        (df['family_history'] == 1).astype(float) * 0.15 +
        np.random.normal(0, 0.1, n)
    )
    df['diabetes'] = (diabetes_score > 0.45).astype(int)

    # Hypertension risk
    hypertension_score = (
        (df['age'] > 50).astype(float) * 0.2 +
        (df['blood_pressure_systolic'] > 140).astype(float) * 0.3 +
        (df['blood_pressure_diastolic'] > 90).astype(float) * 0.15 +
        (df['bmi'] > 28).astype(float) * 0.1 +
        (df['smoking'] == 2).astype(float) * 0.1 +
        (df['alcohol'] == 2).astype(float) * 0.05 +
        (df['family_history'] == 3).astype(float) * 0.1 +
        np.random.normal(0, 0.1, n)
    )
    df['hypertension'] = (hypertension_score > 0.45).astype(int)

    # Heart disease risk
    heart_score = (
        (df['age'] > 55).astype(float) * 0.2 +
        (df['cholesterol'] > 240).astype(float) * 0.2 +
        (df['blood_pressure_systolic'] > 140).astype(float) * 0.15 +
        (df['bmi'] > 30).astype(float) * 0.1 +
        (df['smoking'] == 2).astype(float) * 0.15 +
        (df['gender'] == 1).astype(float) * 0.05 +
        (df['physical_activity'] == 0).astype(float) * 0.1 +
        (df['family_history'] == 2).astype(float) * 0.1 +
        np.random.normal(0, 0.1, n)
    )
    df['heart_disease'] = (heart_score > 0.5).astype(int)

    return df


def train_and_save_models(df):
    """Train RF, XGB, SVM for each disease and save models."""
    feature_cols = [
        'age', 'gender', 'bmi', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'cholesterol', 'glucose',
        'smoking', 'alcohol', 'physical_activity', 'family_history'
    ]
    diseases = ['diabetes', 'hypertension', 'heart_disease']

    X = df[feature_cols]

    # Scale features for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'ml_models/scaler.pkl')
    print("Saved scaler.pkl")

    for disease in diseases:
        y = df[disease]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        print(f"\n{'='*60}")
        print(f"Training models for: {disease.upper()}")
        print(f"{'='*60}")
        print(f"Class distribution: {dict(y.value_counts())}")

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        rf_roc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        joblib.dump(rf, f'ml_models/rf_{disease}.pkl')
        print(f"  Random Forest Accuracy: {rf_acc:.4f} | ROC-AUC: {rf_roc:.4f}")

        # Gradient Boosting
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42
        )
        gbm.fit(X_train, y_train)
        gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
        gbm_roc = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])
        joblib.dump(gbm, f'ml_models/gbm_{disease}.pkl')
        print(f"  Gradient Boosting Acc:  {gbm_acc:.4f} | ROC-AUC: {gbm_roc:.4f}")

        # SVM (uses scaled features)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
        svm_roc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])
        joblib.dump(svm, f'ml_models/svm_{disease}.pkl')
        print(f"  SVM Accuracy:           {svm_acc:.4f} | ROC-AUC: {svm_roc:.4f}")

    print(f"\n{'='*60}")
    print("All models trained and saved to ml_models/")
    print(f"{'='*60}")


if __name__ == '__main__':
    print("Generating synthetic health data...")
    df = generate_synthetic_data()
    print(f"Generated {len(df)} samples")
    train_and_save_models(df)

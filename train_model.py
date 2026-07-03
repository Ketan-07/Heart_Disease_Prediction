"""
train_model.py
Trains a heart disease prediction model on the UCI Cleveland dataset (14 features).

Run:
    python train_model.py

Outputs:
    model/heart_model.pkl  -> dict with preprocessor, calibrated model, SHAP-ready model,
                               isolation forest anomaly detector, and metadata.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "data/heart.csv"
MODEL_PATH = "model/heart_model.pkl"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # '?' marks missing values in the raw UCI data (mainly in 'ca' and 'thal')
    df = df.replace("?", np.nan)
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert multi-class diagnosis (0-4) into binary target: 0 = no disease, 1 = disease
    df["target"] = (df["diagnosis"] > 0).astype(int)
    df = df.drop(columns=["diagnosis"])
    return df


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def main():
    print("Loading data...")
    df = load_data()
    X = df[ALL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        ),
    }

    print("\nComparing models via 5-fold cross-validated ROC-AUC:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_model = None, -1, None
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()
        print(f"  {name:20s} ROC-AUC = {mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_name, best_score, best_model = name, mean_score, model

    print(f"\nBest model: {best_name} (ROC-AUC = {best_score:.4f})")

    # Fit the uncalibrated best model on full training data -> used later for SHAP explanations
    shap_model = candidates[best_name]
    shap_model.fit(X_train_proc, y_train)

    # Fit a calibrated version for trustworthy probability outputs
    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(candidates[best_name], cv=5, method="sigmoid")
    calibrated_model.fit(X_train_proc, y_train)

    # Evaluate on held-out test set
    y_pred = calibrated_model.predict(X_test_proc)
    y_proba = calibrated_model.predict_proba(X_test_proc)[:, 1]
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Anomaly detector to flag unusual input combinations at inference time
    print("Fitting anomaly detector...")
    anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(X_train_proc)

    # Feature names after one-hot encoding, for SHAP labeling
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(cat_feature_names)

    # Small background sample for SHAP (needed for linear/kernel explainers)
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_train_proc.shape[0], size=min(100, X_train_proc.shape[0]), replace=False)
    background_sample = X_train_proc[bg_idx]

    artifact = {
        "preprocessor": preprocessor,
        "calibrated_model": calibrated_model,
        "shap_model": shap_model,
        "anomaly_detector": anomaly_detector,
        "background_sample": background_sample,
        "model_name": best_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "encoded_feature_names": all_feature_names,
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved trained artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()

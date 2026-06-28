# ============================================================
#   models/train.py — Train All ML Models
# ============================================================

import os
import sys
import joblib
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PARAMS, OUTPUT_DIR, RANDOM_STATE

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not found. Install with: pip install xgboost")


def build_models():
    """
    Instantiate all ML models using config hyperparameters.

    Returns:
        dict: {model_name: model_object}
    """
    models = {
        "Linear Regression":  LinearRegression(**MODEL_PARAMS["Linear Regression"]),
        "Ridge Regression":   Ridge(**MODEL_PARAMS["Ridge Regression"]),
        "Lasso Regression":   Lasso(**MODEL_PARAMS["Lasso Regression"]),
        "Random Forest":      RandomForestRegressor(**MODEL_PARAMS["Random Forest"]),
        "Gradient Boosting":  GradientBoostingRegressor(**MODEL_PARAMS["Gradient Boosting"]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(**MODEL_PARAMS["XGBoost"])

    return models


def train_all(models, X_train, X_train_scaled, y_train, verbose=True):
    """
    Train every model in the dictionary.
    Linear models use scaled features; tree models use raw features.

    Args:
        models         : dict of {name: model}
        X_train        : Raw training features
        X_train_scaled : Scaled training features
        y_train        : Training labels

    Returns:
        trained_models : dict of {name: fitted_model}
    """
    LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
    trained = {}

    print("\n🤖 Training Models...")
    print("-" * 40)
    for name, model in models.items():
        X = X_train_scaled if name in LINEAR_MODELS else X_train
        print(f"   ⏳ {name:<25}", end=" ", flush=True)
        model.fit(X, y_train)
        trained[name] = model
        print("✅ Done")

    return trained


def save_model(model, name, verbose=True):
    """Save a single model to disk."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_name}.pkl")
    joblib.dump(model, path)
    if verbose:
        print(f"   💾 Saved: {path}")
    return path


def save_all_models(trained_models):
    """Save all trained models to the outputs directory."""
    print("\n💾 Saving All Models...")
    paths = {}
    for name, model in trained_models.items():
        paths[name] = save_model(model, name)
    return paths


def load_model(name):
    """Load a previously saved model by name."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    print(f"   ✅ Loaded: {path}")
    return model


if __name__ == "__main__":
    from data.data_loader import load_data
    from utils.preprocessing import split_data, scale_features

    X, y, df = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    models  = build_models()
    trained = train_all(models, X_train, X_train_sc, y_train)
    save_all_models(trained)

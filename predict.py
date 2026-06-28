# ============================================================
#   models/predict.py — Predict on New House Data
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, SAMPLE_HOUSE

LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}


def load_best_model(model_name, scaler_path=None):
    """
    Load a saved model and optionally its scaler.

    Args:
        model_name   : Name of the model (e.g., 'random_forest')
        scaler_path  : Path to scaler.pkl (needed for linear models)

    Returns:
        model, scaler (or None)
    """
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}\n"
                                 "   Please run main.py first to train & save models.")
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {model_path}")

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded: {scaler_path}")

    return model, scaler


def predict_single(model, model_name, house_data, scaler=None):
    """
    Predict the price for a single house.

    Args:
        model       : Trained model object
        model_name  : Model name (to decide scaling)
        house_data  : dict or pd.DataFrame of house features
        scaler      : StandardScaler (required for linear models)

    Returns:
        float: Predicted house value in dollars
    """
    if isinstance(house_data, dict):
        house_df = pd.DataFrame([house_data])
    else:
        house_df = house_data.copy()

    if model_name in LINEAR_MODELS:
        if scaler is None:
            raise ValueError("Scaler is required for linear models.")
        X = scaler.transform(house_df)
    else:
        X = house_df.values

    price_100k = model.predict(X)[0]
    price_usd  = price_100k * 100_000
    return price_usd


def predict_batch(model, model_name, df_new, scaler=None):
    """
    Predict prices for a batch of houses from a DataFrame or CSV.

    Args:
        model       : Trained model
        model_name  : Model name
        df_new      : pd.DataFrame of new house records
        scaler      : Scaler (required for linear models)

    Returns:
        pd.DataFrame: Original df with 'PredictedPrice_USD' column added
    """
    if model_name in LINEAR_MODELS:
        if scaler is None:
            raise ValueError("Scaler is required for linear models.")
        X = scaler.transform(df_new)
    else:
        X = df_new.values

    preds = model.predict(X) * 100_000
    result = df_new.copy()
    result["PredictedPrice_USD"] = preds.round(2)
    return result


def predict_from_csv(csv_path, model_name, scaler=None):
    """
    Load a CSV file of houses and predict prices.

    Args:
        csv_path   : Path to CSV file
        model_name : Saved model name (without .pkl)
        scaler     : Scaler object (for linear models)
    """
    model, _ = load_best_model(model_name)
    df = pd.read_csv(csv_path)
    result = predict_batch(model, model_name, df, scaler)
    out_path = csv_path.replace(".csv", "_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"✅ Predictions saved → {out_path}")
    return result


if __name__ == "__main__":
    # ── Example: predict on a single sample house ────────────
    SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
    MODEL_NAME  = "random_forest"   # change as needed

    model, scaler = load_best_model(MODEL_NAME, SCALER_PATH)

    price = predict_single(model, MODEL_NAME.replace("_", " ").title(),
                           SAMPLE_HOUSE, scaler)

    print(f"\n🏠 Sample House Details:")
    for k, v in SAMPLE_HOUSE.items():
        print(f"   {k:<15} : {v}")
    print(f"\n💰 Predicted House Value : ${price:,.2f}")

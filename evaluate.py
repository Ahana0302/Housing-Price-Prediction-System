# ============================================================
#   models/evaluate.py — Evaluate & Compare Models
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}


def evaluate_model(name, model, X_test, X_test_scaled, y_test, verbose=True):
    """
    Evaluate a single model on the test set.

    Args:
        name          : Model name (str)
        model         : Trained model object
        X_test        : Raw test features
        X_test_scaled : Scaled test features
        y_test        : True labels

    Returns:
        dict with MAE, RMSE, R², and predictions
    """
    X = X_test_scaled if name in LINEAR_MODELS else X_test
    preds = model.predict(X)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    if verbose:
        print(f"\n  [{name}]")
        print(f"   MAE  : {mae:.4f}  ≈  ${mae * 100_000:,.0f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   R²   : {r2:.4f}")

    return {
        "Model":       name,
        "MAE":         round(mae, 4),
        "RMSE":        round(rmse, 4),
        "R2":          round(r2, 4),
        "predictions": preds,
        "model_obj":   model
    }


def evaluate_all(trained_models, X_test, X_test_scaled, y_test):
    """
    Evaluate all trained models and return a list of result dicts.

    Returns:
        list of result dicts
    """
    print("\n📋 Evaluating All Models...")
    print("=" * 45)
    results = []
    for name, model in trained_models.items():
        res = evaluate_model(name, model, X_test, X_test_scaled, y_test)
        results.append(res)
    return results


def summarize_results(results):
    """
    Build a sorted DataFrame summary of all model results.

    Returns:
        pd.DataFrame, best_result (dict)
    """
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("predictions", "model_obj")}
        for r in results
    ]).sort_values("R2", ascending=False).reset_index(drop=True)

    print("\n\n🏆 MODEL LEADERBOARD")
    print("=" * 55)
    print(summary.to_string(index=False))

    best_name   = summary.iloc[0]["Model"]
    best_result = next(r for r in results if r["Model"] == best_name)
    print(f"\n🥇 Best Model : {best_name}")
    print(f"   R²         : {summary.iloc[0]['R2']:.4f}")
    print(f"   MAE        : ${summary.iloc[0]['MAE'] * 100_000:,.0f}")

    return summary, best_result


def cross_validate_model(name, model, X, y, cv=5):
    """
    Run K-Fold cross-validation on a model.

    Args:
        name  : Model name
        model : Model object (unfitted)
        X, y  : Full feature matrix and labels
        cv    : Number of folds (default: 5)
    """
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="r2", n_jobs=-1)
    print(f"\n🔄 Cross-Validation ({cv}-Fold) — {name}")
    print(f"   Scores : {np.round(scores, 4)}")
    print(f"   Mean   : {scores.mean():.4f}")
    print(f"   Std    : {scores.std():.4f}")
    return scores


def save_results(summary, filename="results.csv"):
    """Save model comparison table to CSV."""
    path = os.path.join(OUTPUT_DIR, filename)
    summary.to_csv(path, index=False)
    print(f"\n💾 Results saved → {path}")
    return path


if __name__ == "__main__":
    print("Run main.py to evaluate all models.")

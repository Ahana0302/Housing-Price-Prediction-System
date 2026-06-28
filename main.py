# ============================================================
#   main.py — Run the Full House Price Prediction Pipeline
#
#   Usage:
#       python main.py
# ============================================================

import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, SAMPLE_HOUSE
from data.data_loader import load_data, explore_data
from utils.preprocessing import split_data, scale_features, get_feature_info
from utils.visualize import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
)
from models.train import build_models, train_all, save_all_models
from models.evaluate import evaluate_all, summarize_results, save_results
from models.predict import predict_single


def main():
    print("\n" + "=" * 60)
    print("   🏠  HOUSE PRICE PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # ── STEP 1 : Load Data ───────────────────────────────────
    print("\n📦 STEP 1: Loading Data")
    X, y, df = load_data(verbose=True)
    explore_data(df)
    get_feature_info(X)

    # ── STEP 2 : Visualise Raw Data ──────────────────────────
    print("\n📊 STEP 2: Exploratory Data Analysis (EDA)")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_distribution(y)

    # ── STEP 3 : Preprocess ──────────────────────────────────
    print("\n⚙️  STEP 3: Preprocessing")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    # ── STEP 4 : Build & Train Models ───────────────────────
    print("\n🤖 STEP 4: Building & Training Models")
    models         = build_models()
    trained_models = train_all(models, X_train, X_train_sc, y_train)

    # ── STEP 5 : Evaluate ────────────────────────────────────
    print("\n📋 STEP 5: Evaluating Models")
    results              = evaluate_all(trained_models, X_test, X_test_sc, y_test)
    summary, best_result = summarize_results(results)

    # ── STEP 6 : Visualise Results ───────────────────────────
    print("\n📈 STEP 6: Generating Result Plots")
    plot_model_comparison(summary)
    plot_actual_vs_predicted(
        y_test,
        best_result["predictions"],
        best_result["Model"]
    )
    plot_residuals(
        y_test,
        best_result["predictions"],
        best_result["Model"]
    )

    # Feature importance for tree models
    rf_result = next((r for r in results if r["Model"] == "Random Forest"), None)
    if rf_result:
        plot_feature_importance(
            rf_result["model_obj"], X.columns, "Random Forest"
        )

    # ── STEP 7 : Save Models & Results ──────────────────────
    print("\n💾 STEP 7: Saving Models & Results")
    save_all_models(trained_models)
    save_results(summary)

    # Save best model separately
    best_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(best_result["model_obj"], best_path)
    print(f"   🏆 Best model saved → {best_path}")

    # ── STEP 8 : Predict on New Data ────────────────────────
    print("\n🔮 STEP 8: Sample Prediction")
    print("\n   House Features:")
    for k, v in SAMPLE_HOUSE.items():
        print(f"   {k:<15} : {v}")

    price = predict_single(
        model       = best_result["model_obj"],
        model_name  = best_result["Model"],
        house_data  = SAMPLE_HOUSE,
        scaler      = scaler
    )
    print(f"\n   💰 Predicted House Value : ${price:,.2f}")

    # ── DONE ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   ✅  PIPELINE COMPLETE!")
    print(f"   All outputs saved in: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

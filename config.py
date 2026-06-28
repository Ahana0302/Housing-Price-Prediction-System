# ============================================================
#   config.py — All Settings & Hyperparameters
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data Settings ────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Model Hyperparameters ────────────────────────────────────
MODEL_PARAMS = {
    "Linear Regression": {},

    "Ridge Regression": {
        "alpha": 1.0
    },

    "Lasso Regression": {
        "alpha": 0.01
    },

    "Random Forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },

    "Gradient Boosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": RANDOM_STATE
    },

    "XGBoost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": 0
    }
}

# ── Plot Settings ────────────────────────────────────────────
PLOT_DPI    = 150
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
COLORS      = {
    "primary":   "steelblue",
    "secondary": "darkorange",
    "accent":    "teal",
    "error":     "#e74c3c",
    "success":   "#27ae60"
}

# ── Sample New House for Prediction ─────────────────────────
SAMPLE_HOUSE = {
    "MedInc":    8.3252,
    "HouseAge":  41.0,
    "AveRooms":  6.984,
    "AveBedrms": 1.024,
    "Population":322.0,
    "AveOccup":  2.555,
    "Latitude":  37.88,
    "Longitude": -122.23
}

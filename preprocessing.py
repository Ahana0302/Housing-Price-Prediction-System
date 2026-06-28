# ============================================================
#   utils/preprocessing.py — Data Cleaning & Scaling
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SIZE, RANDOM_STATE, OUTPUT_DIR


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, verbose=True):
    """
    Split dataset into training and test sets.

    Args:
        X             : Feature matrix
        y             : Target vector
        test_size     : Fraction for test set (default: 0.2)
        random_state  : Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"\n✂️  Train/Test Split")
        print(f"   Train : {X_train.shape[0]:,} samples")
        print(f"   Test  : {X_test.shape[0]:,} samples")
        print(f"   Ratio : {int((1-test_size)*100)}/{int(test_size*100)}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save=True, verbose=True):
    """
    Apply StandardScaler to features (needed for linear models).

    Args:
        X_train  : Training features
        X_test   : Test features
        save     : Save scaler to disk (default: True)

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if save:
        path = os.path.join(OUTPUT_DIR, "scaler.pkl")
        joblib.dump(scaler, path)
        if verbose:
            print(f"\n💾 Scaler saved → {path}")

    return X_train_scaled, X_test_scaled, scaler


def remove_outliers(df, target_col="MedHouseVal", z_thresh=3.0):
    """
    Remove outliers from DataFrame using Z-score method.

    Args:
        df         : Full DataFrame
        target_col : Column to check for outliers
        z_thresh   : Z-score threshold (default: 3.0)

    Returns:
        df_clean   : Cleaned DataFrame
    """
    from scipy import stats
    z_scores  = np.abs(stats.zscore(df[target_col]))
    mask      = z_scores < z_thresh
    df_clean  = df[mask].reset_index(drop=True)
    removed   = len(df) - len(df_clean)
    print(f"\n🧹 Outlier Removal (Z-score > {z_thresh})")
    print(f"   Removed : {removed} rows")
    print(f"   Remaining: {len(df_clean):,} rows")
    return df_clean


def get_feature_info(X):
    """Print feature names, types, and stats."""
    print("\n📌 Feature Summary")
    print("-" * 55)
    info = pd.DataFrame({
        "Feature": X.columns,
        "Dtype"  : X.dtypes.values,
        "Min"    : X.min().values.round(3),
        "Max"    : X.max().values.round(3),
        "Mean"   : X.mean().values.round(3),
        "Std"    : X.std().values.round(3)
    })
    print(info.to_string(index=False))


if __name__ == "__main__":
    from data.data_loader import load_data
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)
    get_feature_info(X)

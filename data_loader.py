# ============================================================
#   data/data_loader.py — Load & Explore Dataset
# ============================================================

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data(verbose=True):
    """
    Load the California Housing dataset.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series)   : Target vector (median house value in $100K)
        df (pd.DataFrame): Full dataframe including target
    """
    raw  = fetch_california_housing(as_frame=True)
    df   = raw.frame
    X    = raw.data
    y    = raw.target

    if verbose:
        print("=" * 55)
        print("  DATASET : California Housing")
        print("=" * 55)
        print(f"  Shape    : {df.shape}")
        print(f"  Features : {list(X.columns)}")
        print(f"  Target   : MedHouseVal  (median value × $100K)")
        print(f"  Samples  : {len(df):,}")
        print()

    return X, y, df


def explore_data(df):
    """
    Print summary statistics and data quality report.

    Args:
        df (pd.DataFrame): Full dataframe

    Returns:
        None
    """
    print("\n📋 Basic Info")
    print("-" * 40)
    print(df.dtypes)

    print("\n📊 Descriptive Statistics")
    print("-" * 40)
    print(df.describe().round(3).to_string())

    print("\n🔍 Missing Values")
    print("-" * 40)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values found!")
    else:
        print(missing[missing > 0])

    print("\n🔢 Duplicate Rows")
    print("-" * 40)
    dupes = df.duplicated().sum()
    print(f"  Duplicates: {dupes}")

    return None


if __name__ == "__main__":
    X, y, df = load_data(verbose=True)
    explore_data(df)

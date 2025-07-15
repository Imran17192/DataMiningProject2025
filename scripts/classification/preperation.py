# -----------------------------------------------------------------------------
# scripts/classification/preperation.py
# -----------------------------------------------------------------------------
"""Dataset helpers: index‑alignment + stratified splits."""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

def align_x_y(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X_aligned, y_aligned = X.align(y, join="inner", axis=0)
    return X_aligned, y_aligned.iloc[:, 0] if isinstance(y_aligned, pd.DataFrame) else y_aligned

def make_splits(X: pd.DataFrame, y: pd.Series, *, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss1.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    val_frac = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_state
    )
    tr_idx2, val_idx = next(sss2.split(X_train, y_train))
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    X_train, y_train = X_train.iloc[tr_idx2], y_train.iloc[tr_idx2]

    return X_train, X_val, X_test, y_train, y_val, y_test

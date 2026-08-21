from pathlib import Path
import pandas as pd
from src.features import FEATURES, FINAL_STATUSES

def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, usecols=FEATURES + ["loan_status"], low_memory=False)
    df = df[df["loan_status"].isin(FINAL_STATUSES)].copy()
    y = df["loan_status"].map(FINAL_STATUSES).astype("int8")
    X = df[FEATURES].copy()
    for column in ("int_rate", "revol_util"):
        X[column] = pd.to_numeric(X[column].astype("string").str.rstrip("%"), errors="coerce")
    if y.nunique() != 2:
        raise ValueError("Target must contain paid and defaulted loans")
    return X, y

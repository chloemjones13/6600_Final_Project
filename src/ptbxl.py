"""PTB-XL metadata, labels and splits, matching notebooks/01_data_cleaning.ipynb.

Only the two metadata CSVs in data/ are needed here; signals are loaded separately.
"""
import ast
import os

import numpy as np
import pandas as pd

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
Y_COLS = ["y_" + c for c in CLASSES]
P_COLS = ["p_" + c for c in CLASSES]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS = range(1, 8), [8], [9, 10]


def fold_to_split(fold):
    if fold in TRAIN_FOLDS:
        return "train"
    return "val" if fold in VAL_FOLDS else "test"


def load_metadata(data_dir="data/"):
    """Cleaned modeling table: ages 20-85, 5 diagnostic superclasses, no-label records dropped.

    Returns one row per ECG (index = ecg_id) with sex (0 = male, 1 = female), age,
    device, filename_lr, strat_fold, split and one 0/1 column per class.
    """
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    df = df[(df["age"] >= 20) & (df["age"] <= 85)].copy()

    scp = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    diagnostic = scp.loc[scp["diagnostic"] == 1, "diagnostic_class"]
    codes = df["scp_codes"].apply(ast.literal_eval)
    labels = codes.apply(lambda c: sorted({diagnostic[k] for k in c if k in diagnostic.index}))
    for c in CLASSES:
        df[c] = labels.apply(lambda x, c=c: c in x).astype(int)

    df = df[df[CLASSES].sum(axis=1) > 0].copy()
    df["split"] = df["strat_fold"].map(fold_to_split)
    return df


def group_mask(meta, group):
    """Boolean mask for a training group: 'all', 'female' or 'male'."""
    if group == "all":
        return np.ones(len(meta), bool)
    return (meta["sex"] == (1 if group == "female" else 0)).values

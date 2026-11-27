"""Rebuild every results table from the saved prediction CSVs. No data download or GPU needed.

    python src/evaluate.py                    # all models found in the results folders
    python src/evaluate.py --out results/final

Each *_preds.csv has one row per ECG with split, sex (0 = male, 1 = female), true labels y_<class>,
predicted probabilities p_<class>, model and seed. Only test rows (folds 9-10) are scored.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptbxl import CLASSES, P_COLS, Y_COLS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIRS = ["cnn_results", "results", "hubert_results"]
SEXES = {"male": 0, "female": 1}


def find_prediction_files(repo=REPO):
    files = []
    for d in PRED_DIRS:
        files += glob.glob(os.path.join(repo, d, "**", "*_preds.csv"), recursive=True)
    return sorted(set(files))


def load_predictions(files):
    """Test-set predictions for every (model, seed), as one long table."""
    frames = [pd.read_csv(f) for f in files]
    preds = pd.concat(frames, ignore_index=True)
    return preds[preds["split"] == "test"]


def check_same_test_set(preds):
    """Every model must be scored on the same ECGs with the same labels."""
    ref = None
    for key, t in preds.groupby(["model", "seed"]):
        t = t.set_index("ecg_id")[Y_COLS + ["sex"]].sort_index()
        if ref is None:
            ref = t
        elif not t.equals(ref):
            raise ValueError(f"{key}: test records or labels differ from the other models")
    return len(ref)


def run_metrics(t):
    """Macro AUROC/AUPRC overall and by sex for one model + seed."""
    r = {}
    for name, mask in [("all", np.ones(len(t), bool))] + [(s, (t["sex"] == v).values) for s, v in SEXES.items()]:
        y, p = t.loc[mask, Y_COLS].values, t.loc[mask, P_COLS].values
        r[f"auroc_{name}"] = roc_auc_score(y, p, average="macro")
        r[f"auprc_{name}"] = average_precision_score(y, p, average="macro")
    r["gap_m_minus_f"] = r["auroc_male"] - r["auroc_female"]
    return r


def summary_table(preds):
    runs = pd.DataFrame([dict(model=m, seed=s, **run_metrics(t)) for (m, s), t in preds.groupby(["model", "seed"])])
    agg = runs.drop(columns="seed").groupby("model").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", runs.groupby("model")["seed"].nunique())
    return agg.fillna(0.0), runs


def per_class_gap(preds):
    """Per-class AUROC gap (male - female), mean over seeds."""
    rows = []
    for (m, s), t in preds.groupby(["model", "seed"]):
        for c in CLASSES:
            auc = {sex: roc_auc_score(t.loc[t["sex"] == v, "y_" + c], t.loc[t["sex"] == v, "p_" + c])
                   for sex, v in SEXES.items()}
            rows.append(dict(model=m, cls=c, seed=s, gap=auc["male"] - auc["female"]))
    gaps = pd.DataFrame(rows).groupby(["cls", "model"])["gap"].mean().unstack()
    return gaps.reindex(CLASSES)


def female_miss_rates(preds, seed=0, cutoff=0.5):
    """Share of true positives in women scored below the cutoff (single seed, as in the write-ups)."""
    t = preds[(preds["seed"] == seed) & (preds["sex"] == SEXES["female"])]
    rows = []
    for m, tm in t.groupby("model"):
        for c in CLASSES:
            pos = tm[tm["y_" + c] == 1]
            rows.append(dict(model=m, cls=c, miss_rate=(pos["p_" + c] < cutoff).mean()))
    return pd.DataFrame(rows).pivot(index="cls", columns="model", values="miss_rate").reindex(CLASSES)


def to_markdown(df, digits=3):
    df = df.reset_index()
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "final"), help="folder for the output CSVs")
    ap.add_argument("--files", nargs="*", help="specific *_preds.csv files (default: search the results folders)")
    args = ap.parse_args()

    files = args.files or find_prediction_files()
    if not files:
        sys.exit("No *_preds.csv files found.")
    preds = load_predictions(files)
    n_test = check_same_test_set(preds)

    summary, runs = summary_table(preds)
    gaps, misses = per_class_gap(preds), female_miss_rates(preds)

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary_by_model.csv"))
    runs.to_csv(os.path.join(args.out, "metrics_by_seed.csv"), index=False)
    gaps.to_csv(os.path.join(args.out, "per_class_gap.csv"))
    misses.to_csv(os.path.join(args.out, "female_miss_rates_seed0.csv"))

    cols = ["n_seeds", "auroc_all_mean", "auroc_all_std", "auroc_male_mean", "auroc_female_mean",
            "gap_m_minus_f_mean", "auprc_female_mean"]
    print(f"{len(files)} prediction files, {summary.shape[0]} models, {n_test} shared test ECGs\n")
    print("Macro AUROC / AUPRC on the test set (mean and std over seeds)\n")
    print(to_markdown(summary[cols].sort_values("auroc_all_mean", ascending=False)))
    print("\nPer-class AUROC gap, male - female (mean over seeds)\n")
    print(to_markdown(gaps))
    print("\nMiss rate in women at a 0.5 cutoff (seed 0)\n")
    print(to_markdown(misses))
    print(f"\nCSVs written to {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()

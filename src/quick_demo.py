"""Small end-to-end demo: load a few hundred PTB-XL ECGs, train the CNN for a few epochs on CPU,
and score it by sex with the same metrics as the full project.

    python src/quick_demo.py                  # ~1,000 ECGs streamed from PhysioNet, a few minutes on a laptop
    python src/quick_demo.py --smoke-test     # random signals, no download: checks the install only

If data/records100/ exists (full PTB-XL download, see README), records are read from disk instead of
streamed. Numbers from this demo are far below the full runs (tiny training set, few epochs); it shows
the pipeline, not the results. The full experiments are the notebooks.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import ECGResNet1d  # noqa: E402
from evaluate import female_miss_rates, run_metrics, to_markdown  # noqa: E402
from ptbxl import CLASSES, P_COLS, Y_COLS, group_mask, load_metadata  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSIONET_DIR = "ptb-xl/1.0.3"


def sample_records(meta, sizes, group, seed):
    """Random subset per split; train/val follow the training group, test always has both sexes."""
    rng = np.random.default_rng(seed)
    parts = []
    for split, n in sizes.items():
        m = meta[meta["split"] == split]
        if split != "test":
            m = m[group_mask(m, group)]
        parts.append(m.loc[np.sort(rng.choice(m.index, min(n, len(m)), replace=False))])
    return pd.concat(parts)


def load_signals(meta, data_dir, cache_file):
    """(N, 12, 1000) raw mV. Reads data/records100 if present, otherwise streams from PhysioNet."""
    import wfdb

    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if np.array_equal(cached["ids"], meta.index.values):
            return cached["X"]
    local = os.path.isdir(os.path.join(data_dir, "records100"))
    print("Reading signals from", "data/records100" if local else f"PhysioNet ({PHYSIONET_DIR}), streaming")
    X = np.empty((len(meta), 12, 1000), dtype=np.float32)
    for i, fn in enumerate(meta["filename_lr"]):
        if local:
            X[i] = wfdb.rdsamp(os.path.join(data_dir, fn))[0].T
        else:
            X[i] = wfdb.rdsamp(os.path.basename(fn), pn_dir=f"{PHYSIONET_DIR}/{os.path.dirname(fn)}")[0].T
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(meta)}")
    np.savez_compressed(cache_file, X=X, ids=meta.index.values)
    return X


def fake_signals(meta, seed):
    return np.random.default_rng(seed).normal(0, 0.3, (len(meta), 12, 1000)).astype(np.float32)


def train(X_tr, Y_tr, X_va, Y_va, epochs, seed):
    torch.manual_seed(seed)
    model = ECGResNet1d()
    model.set_normalization(X_tr)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X_tr), torch.from_numpy(Y_tr)
    g = torch.Generator().manual_seed(seed)
    best, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        p_va = predict(model, X_va)
        auc = macro_auroc(Y_va, p_va)
        print(f"  epoch {epoch}: train loss {loss.item():.3f} | val macro AUROC {auc:.3f} | {time.time() - t0:.0f}s")
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([torch.sigmoid(model(xb)) for xb in torch.from_numpy(X).split(256)]).numpy()


def macro_auroc(y, p):
    keep = (y.min(axis=0) == 0) & (y.max(axis=0) == 1)       # skip classes absent from a small sample
    return roc_auc_score(y[:, keep], p[:, keep], average="macro") if keep.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="all", choices=["all", "female", "male"], help="training population")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true", help="random signals instead of real ECGs (no download)")
    ap.add_argument("--out", default=os.path.join(REPO, "demo_output"))
    args = ap.parse_args()

    data_dir = os.path.join(REPO, "data")
    os.makedirs(args.out, exist_ok=True)
    meta = sample_records(load_metadata(data_dir),
                          dict(train=args.n_train, val=args.n_val, test=args.n_test), args.group, args.seed)
    print(pd.crosstab(meta["split"], meta["sex"].map({0: "male", 1: "female"})), "\n")

    if args.smoke_test:
        X = fake_signals(meta, args.seed)
    else:
        X = load_signals(meta, data_dir, os.path.join(args.out, f"signals_seed{args.seed}.npz"))
    Y = meta[CLASSES].values.astype(np.float32)
    s = {k: (meta["split"] == k).values for k in ["train", "val", "test"]}

    print(f"Training cnn_{args.group} for {args.epochs} epochs on {s['train'].sum()} ECGs (CPU)")
    model = train(X[s["train"]], Y[s["train"]], X[s["val"]], Y[s["val"]], args.epochs, args.seed)

    test = meta[s["test"]][["sex", "age", "device"]].reset_index()
    test["split"] = "test"
    test[Y_COLS] = Y[s["test"]]
    test[P_COLS] = predict(model, X[s["test"]])
    test["model"], test["seed"] = f"demo_cnn_{args.group}", args.seed
    pred_file = os.path.join(args.out, f"demo_cnn_{args.group}_seed{args.seed}_preds.csv")
    test.to_csv(pred_file, index=False)

    m = run_metrics(test)
    print(f"\nTest macro AUROC: all {m['auroc_all']:.3f} | male {m['auroc_male']:.3f} | "
          f"female {m['auroc_female']:.3f} | gap (M - F) {m['gap_m_minus_f']:+.3f}")
    print("\nMiss rate in women at a 0.5 cutoff\n")
    print(to_markdown(female_miss_rates(test, seed=args.seed)))
    print(f"\nPredictions saved to {os.path.relpath(pred_file, REPO)} (same format as the full runs)")
    if args.smoke_test:
        print("Smoke test used random signals, so AUROC near 0.5 is expected.")


if __name__ == "__main__":
    main()

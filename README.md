# Sex-Matched vs. More Data: What Closes the Subgroup Gap in ECG Classification?

**DSAN 6600 Final Project · Georgetown University · Fall 2026**
Chloe Jones · Olivia Semien

ECG classifiers trained on PTB-XL perform worse for women than for men. We ask whether that gap closes with **sex-matched training data** or with **more data**, holding the model fixed and changing only who is in the training set. We then test whether "more data" borrowed from outside the dataset (a foundation model pretrained on 9.1M ECGs) helps.

> **📄 [Read the final report](reports/final-report.md)** — problem, data, evaluation, models, results, failure analysis, limitations and future work in one document.

---

## Key findings

| | Macro AUROC, men | Macro AUROC, women | Gap (M − F) |
|---|---|---|---|
| CNN, trained on everyone (`cnn_all`, **baseline**) | **0.930** | **0.917** | 0.012 |
| CNN, trained on women only | 0.913 | 0.905 | 0.008 |
| CNN, trained on men only | 0.924 | 0.907 | 0.017 |
| HuBERT-ECG, fine-tuned (**extension**) | 0.903 | 0.902 | 0.002 |
| Logistic regression, lead statistics | 0.768 | 0.763 | 0.005 |

*Test folds 9–10 (2,137 men, 1,842 women), mean over 3 seeds.*

1. **More data beat sex-matched data for women** at the sizes we tested: the mixed CNN is best for women, and a CNN that never saw a female ECG matches the female-only one on women.
2. **Every neural model is worse for women, including the female-only one**, so part of the gap comes from the female test set, not only the training mix.
3. **The MI gap (0.016–0.030) survives sex-matching, more data, and a pretrained transformer.** It most likely comes from the signals or labels, not from training composition.
4. **Errors run one way:** disease in women is missed more often (MI missed 37% of the time in women vs 27% in men).
5. **The foundation model didn't help.** HuBERT-ECG's gap is smaller only because men lost more AUROC than women; it collapses on hypertrophy because its preprocessing erases ECG voltage.

![Test AUROC by sex for every model](figures/final/auroc_by_sex_all_models.png)

---

## Project story

| Stage | What we did | Write-up | Notebooks |
|---|---|---|---|
| **Check-in 1** | Problem framing, PTB-XL audit, EDA (base rates, voltage differences, label bias, device confound), evaluation plan | [check-in-1.md](reports/check-in-1.md) | [01](notebooks/01_data_cleaning.ipynb), [02](notebooks/02_eda_signals.ipynb) |
| **Check-in 2** | Baseline: 1D ResNet CNN trained on female / male / all data (3 seeds each); logistic-regression floor; failure analysis | [check-in-2.md](reports/check-in-2.md) | [03](notebooks/03_cnn_baseline.ipynb), [04](notebooks/04_logistic_baseline.ipynb) |
| **Check-in 3** | Extension: HuBERT-ECG transfer learning (frozen, pooled, fine-tuned) vs the CNN; improved logistic baseline | [check-in-3.md](reports/check-in-3.md) | [05](notebooks/05_hubert_transfer_mean.ipynb), [06](notebooks/06_improving_logistic_baseline.ipynb) |
| **Final** | Full narrative, results rebuilt from saved predictions, limitations, future work | [final-report.md](reports/final-report.md) | [00 demo](notebooks/00_results_demo.ipynb) |

Final report sections: [Problem](reports/final-report.md#1-problem-and-motivation) · [Data](reports/final-report.md#2-data) · [Evaluation](reports/final-report.md#3-evaluation-protocol) · [Models](reports/final-report.md#4-models) · [Results](reports/final-report.md#5-results) · [Failure analysis](reports/final-report.md#6-failure-analysis) · [Conclusions](reports/final-report.md#7-what-we-conclude) · [Limitations](reports/final-report.md#8-limitations) · [Future work](reports/final-report.md#9-future-work)

Videos: [Check-in 1](https://georgetown.zoom.us/rec/share/U7h6VP5fh0Z8LLdASP5MVAZjTZPWpq3kQSlmA6FquTaVAzA-Bis6J5t6NKSxKQmK.NyYZniEmehRwY_nv?startTime=1790781544000) · [Check-in 2](https://georgetown.zoom.us/rec/share/2w44Ck5Cgy8kCSn4weL3ehHqS19dM76_IRvDnvnVY4d8fZqI003hjPLKibnPOZc4.Li545-hUdWyEy0iD) · [Check-in 3](https://georgetown.zoom.us/rec/share/1J_2v773JVXGr_3BnfCyCEensfzGVF0WfudHIXqFBFY8QfLcxn2Vv2qWtY8nBpbX.LdTYfVzo9Bir0n4l?startTime=1795100795000)

---

## Quickstart: three ways to run it

| Path | What it shows | Needs | Time |
|---|---|---|---|
| **A. Rebuild the results** | every table in the report, from the saved predictions | no data download, no GPU | < 1 min |
| **B. Quick demo** | the CNN trained and scored by sex end to end on ~1,000 real ECGs | internet (streams from PhysioNet), CPU | a few min |
| **C. Full reproduction** | all experiments from raw data | PTB-XL download, GPU (Colab T4) | hours |

### Setup

```bash
git clone https://github.com/chloemjones13/6600_Final_Project.git
cd 6600_Final_Project
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+ (on Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Inside the activated environment `python` works; without it, use `python3` on macOS/Linux.

### A. Rebuild the results (no data, no GPU)

```bash
python src/evaluate.py
```

Reads every `*_preds.csv` in `cnn_results/`, `results/` and `hubert_results/`, checks that all models were scored on the same 3,979 test ECGs with the same labels, prints the summary, per-class gap and miss-rate tables, and writes them to [`results/final/`](results/final/). The rendered version, with figures, is **[`notebooks/00_results_demo.ipynb`](notebooks/00_results_demo.ipynb)**.

### B. Quick demo (small training run on CPU)

```bash
python src/quick_demo.py                   # 500 train / 150 val / 400 test ECGs, 5 epochs
python src/quick_demo.py --group female    # train on women only
python src/quick_demo.py --smoke-test      # random signals: checks the install without any download
```

Streams the sampled records from PhysioNet (or reads `data/records100/` if you've downloaded PTB-XL), trains the same CNN as the baseline, and prints AUROC by sex plus miss rates in women. Predictions go to `demo_output/` in the same format as the full runs, so `python src/evaluate.py --files demo_output/*_preds.csv` scores them. Expect AUROC well below the full runs: it's a tiny training set and a handful of epochs, meant to show the pipeline, not the results.

### C. Full reproduction

**1. Get the data.** PTB-XL v1.0.3 is open access ([PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), CC BY 4.0, no credentialing). The two metadata CSVs are already in `data/`; only the signals need downloading. We use the 100 Hz records (`records100/`).

```bash
# whole dataset (~1.7 GB zipped), into data/
wget -r -N -c -np -nH --cut-dirs=3 -P data/ https://physionet.org/files/ptb-xl/1.0.3/
# you need data/records100/ next to data/ptbxl_database.csv
```

On **Colab**, zip `records100/` and upload it to `MyDrive/6600_project/records100.zip`. The notebooks mount Drive, clone this repo, unzip the signals, and cache them to Drive so later sessions skip the load.

**2. Run the notebooks in order** (each one's first cells handle paths for both Colab and local runs):

| Notebook | Does | Writes | Hardware / time |
|---|---|---|---|
| [`01_data_cleaning`](notebooks/01_data_cleaning.ipynb) | age filter, superclass labels, splits, cohort EDA | `data/ptbxl_model.pkl`, `figures/*.png` | CPU, seconds |
| [`02_eda_signals`](notebooks/02_eda_signals.ipynb) | signal-level EDA | `figures/*.png` | CPU, minutes |
| [`03_cnn_baseline`](notebooks/03_cnn_baseline.ipynb) | CNN × {female, male, all} × 3 seeds | `results/cnn_*` (committed copy in `cnn_results/`), `figures/cnn_figures/` | T4: ~2 min per run, ~20 min total |
| [`04_logistic_baseline`](notebooks/04_logistic_baseline.ipynb) | raw-feature logistic regression | `results/logreg/` | CPU, minutes |
| [`05_hubert_transfer_mean`](notebooks/05_hubert_transfer_mean.ipynb) | HuBERT-ECG frozen / pooled / fine-tuned, comparison vs CNN | `hubert_results/`, `figures/hubert_figures/` | T4: ~32 s/epoch fine-tuned (3 seeds × up to 20 epochs); downloads weights from Hugging Face |
| [`06_improving_logistic_baseline`](notebooks/06_improving_logistic_baseline.ipynb) | lead-statistics logistic regression | `results/logreg_improved/` | CPU, minutes |

Notebooks 03–06 call notebook 01 automatically if `ptbxl_model.pkl` is missing. Training runs skip any (model, seed) whose outputs already exist, so an interrupted Colab session resumes where it stopped. Every run's settings are saved next to its predictions (`*_config.json`).

**Cheaper full-data paths:** in notebook 03, set `CONFIG["seeds"] = [0]` and `CONFIG["train_groups"] = ["all"]` for a single ~2 min run. In notebook 05, the frozen variants train a linear head on cached embeddings in seconds once embeddings are computed; fine-tuning is the expensive part.

---

## Repository layout

```
├── README.md                  ← you are here
├── reports/
│   ├── final-report.md        ← full technical narrative
│   ├── check-in-1.md          ← framing, data audit, EDA, evaluation plan
│   ├── check-in-2.md          ← CNN baseline + failure analysis
│   └── check-in-3.md          ← HuBERT-ECG extension + improved logistic baseline
├── notebooks/
│   ├── 00_results_demo.ipynb  ← demo: rebuilds all results, no data needed
│   └── 01 … 06                ← cleaning → EDA → CNN → logreg → HuBERT → improved logreg
├── src/
│   ├── ptbxl.py               ← metadata cleaning, labels, splits (same as notebook 01)
│   ├── cnn.py                 ← the 1D ResNet (same as notebook 03)
│   ├── evaluate.py            ← rebuilds every results table from saved predictions
│   └── quick_demo.py          ← small CPU training run, end to end
├── data/                      ← ptbxl_database.csv, scp_statements.csv (signals are gitignored)
├── cnn_results/               ← CNN predictions + training histories, 9 runs
├── hubert_results/            ← HuBERT predictions + histories
├── results/
│   ├── logreg/, logreg_improved/
│   └── final/                 ← tables rebuilt by src/evaluate.py
├── figures/                   ← EDA (top level), cnn_figures/, hubert_figures/, logreg*/, final/
└── requirements.txt
```

**Prediction file format** (all `*_preds.csv`): one row per ECG with `ecg_id`, `split`, `sex` (0 = male, 1 = female), `age`, `device`, true labels `y_NORM … y_HYP`, predicted probabilities `p_NORM … p_HYP`, `model`, `train_group`, `n_train`, `seed`. Any new model saved in this format drops straight into `src/evaluate.py`.

---

## Data and licenses

- **PTB-XL** v1.0.3, Wagner et al. (2020), CC BY 4.0. Not redistributed here beyond the two metadata CSVs.
- **HuBERT-ECG** weights (`Edoardo-Coppola/hubert-ecg-small`), Coppola et al. (2024), CC BY-NC 4.0 (non-commercial). Downloaded at run time, not stored here.

Full references and our AI-use disclosure are at the end of the [final report](reports/final-report.md#ai-disclosure).

# Check-in 3 — DSAN 6600

**Sex-Matched vs. More Data: What Closes the Subgroup Gap in ECG Classification?**

Chloe Jones · Olivia Semien

> Check-in 1: [`check-in-1.md`](check-in-1.md) · Check-in 2: [`README.md`](README.md)

<!-- TODO: every [TBD] below gets filled from notebooks/05_hubert_transfer.ipynb outputs (hubert_results/*.csv). -->

---

## 1. The extension

**Transfer learning from a self-supervised ECG foundation model.** Check-in 2 asked whether *more data* closes the male–female gap within PTB-XL. Pretraining is more data borrowed from outside PTB-XL: HuBERT-ECG (Coppola et al.) was pretrained with self-supervision on 9.1 million unlabeled 12-lead ECGs from four countries. We take the SMALL model (8 transformer layers, hidden size 512) and compare it to our Check-in 2 CNN.

| Regime | What trains | Model |
|---|---|---|
| From scratch (Check-in 2 baseline) | everything | `cnn_all` (1D ResNet, ~1.2M params) |
| Frozen encoder (linear probe) | new 5-class linear head only, on mean-pooled embeddings | `hubert_frozen_all` |
| Fine-tuned encoder | all 8 transformer layers + head (conv feature extractor frozen) | `hubert_finetuned_all` |

Notebook: [`notebooks/05_hubert_transfer.ipynb`](notebooks/05_hubert_transfer.ipynb)

### Why this model
- Openly released weights (safetensors, CC BY-NC 4.0) that load with Hugging Face `AutoModel`.
- Pretrained on large, multi-country data, so it tests "borrowed" scale against the within-PTB-XL scale question.
- Runs on a Colab T4.

### Input pipeline
A pretrained encoder expects inputs like its pretraining data, so we use the authors' preprocessing instead of our CNN's z-score: 0.05–47 Hz FIR bandpass → resample to 500 Hz → per-lead min-max to [−1, 1] → two 5 s halves, each flattened and decimated to 6,000 samples. We checked our implementation against the authors' code (max difference < 1e-7). Training uses a random half per record; evaluation averages the predicted probabilities over both halves. All of this is per-record, so no statistics are shared across records or sexes.

### Training
Fine-tuning follows the paper's PTB-XL recipe (AdamW, encoder LR 1e-5, weight decay 0.01, conv extractor frozen, layerdrop/dropout off), with two deliberate changes: the randomly initialized head gets LR 1e-3, and the loss is **unweighted** like the CNN's (positive-class weights would distort the calibration-by-sex comparison). We also turned off the encoder's pretraining-time masking, which the Hugging Face version otherwise applies during training. Checkpoint selection, LR schedule and early stopping use validation macro AUROC, as for the CNN. Budget: max 20 epochs, patience 5 (CNN: 40/7), to fit Colab time.

---

## 2. Comparison (same protocol as Check-in 2)

Same splits (train folds 1–7, val fold 8, test folds 9–10), same labels, same test records (2,137 male / 1,842 female; the notebook asserts this against the saved CNN predictions), same metrics, 3 seeds each. Mean ± std over seeds.

### Macro AUROC by test sex

| Model | Test: male | Test: female | Gap (M − F) |
|---|---|---|---|
| `cnn_all` | 0.930 ± 0.003 | 0.917 ± 0.003 | 0.012 ± 0.001 |
| `hubert_frozen_all` | [TBD] | [TBD] | [TBD] |
| `hubert_finetuned_all` | [TBD] | [TBD] | [TBD] |

### Macro AUPRC

| Model | All | Male | Female |
|---|---|---|---|
| `cnn_all` | 0.813 | 0.823 | 0.801 |
| `hubert_frozen_all` | [TBD] | [TBD] | [TBD] |
| `hubert_finetuned_all` | [TBD] | [TBD] | [TBD] |

![AUROC by sex](figures/hubert_figures/auroc_by_sex_cnn_vs_hubert.png)

### Ablation: frozen vs fine-tuned encoder

The only thing that changes is whether the pretrained encoder weights update on PTB-XL. [TBD: gain from fine-tuning over the frozen probe, overall and for women.]

### Per-class gap (male − female AUROC)

| Class | `cnn_all` | `hubert_frozen_all` | `hubert_finetuned_all` |
|---|---|---|---|
| NORM | 0.007 | [TBD] | [TBD] |
| MI | 0.021 | [TBD] | [TBD] |
| STTC | 0.012 | [TBD] | [TBD] |
| CD | 0.015 | [TBD] | [TBD] |
| HYP | 0.007 | [TBD] | [TBD] |

![Per-class gap](figures/hubert_figures/per_class_gap_cnn_vs_hubert.png)

### Miss rates in women at a 0.5 cutoff (seed 0)

| Class | `cnn_all` | `hubert_finetuned_all` |
|---|---|---|
| MI | 0.374 | [TBD] |
| STTC | 0.390 | [TBD] |
| CD | 0.327 | [TBD] |
| HYP | 0.545 | [TBD] |

---

## 3. What got better or worse, and why

[TBD after running. Questions to answer:]
- Did pretraining beat the CNN overall, by more than the seed std (~0.003–0.005)?
- Did it help women more than men, i.e. did the gap shrink? Especially MI, where sex-matching didn't help in Check-in 2.
- Does the frozen encoder, which never saw our labels or sex split, show the same gap as the models trained on PTB-XL? If so, the gap is more likely in the signal/labels than in the training composition.
- Did the female miss rates for MI / STTC / CD drop?

---

## 4. Remaining limitations

- **Pretraining overlap with our test set.** HuBERT-ECG's pretraining data includes PTB-XL with labels removed, and the paper doesn't say test folds were held out, so the encoder has likely seen our test ECGs (never their labels or our sex split). Absolute AUROCs may be slightly optimistic. The gap comparison is less affected because the exposure is the same for both sexes.
- **Not a pure architecture comparison.** The CNN and HuBERT differ in architecture, size (~1.2M vs ~30M+ parameters), input preprocessing and training budget at once. The frozen-vs-fine-tuned ablation is the controlled comparison; an optional control in the notebook (`RUN_SCRATCH = True`) trains the same HuBERT architecture from random init to separate "pretraining helps" from "a transformer helps".
- **Only the `all` training group** was run for the transfer arm, so this doesn't yet answer whether pretraining changes the sex-matched vs mixed result.
- **Shorter training budget** than the paper (≈20 epochs vs 70k steps), which may understate fine-tuning.
- Check-in 2 limitations still apply: labels are expert reads that may under-diagnose women (HYP), and the device confound hasn't been checked against the gap.

---

## 5. Video

~5 min, linked [HERE]([TBD])

---

## 6. AI Disclosure

We used AI as an assistant on this check-in, specifically for:

- **Code:** drafting the transfer-learning notebook (preprocessing, model wrapper, training loop, comparison tables and figures) and checking our preprocessing against the authors' implementation
- **Research:** reading the HuBERT-ECG paper and code to find the pretraining data and preprocessing
- **Writing:** drafting parts of this document

**Credit to the HuBERT-ECG authors:** the pretrained model (architecture and weights) is theirs, loaded from Hugging Face. Our preprocessing re-implements theirs, and our fine-tuning settings (learning rate, optimizer, frozen conv layers, dropout/layerdrop off) follow their released PTB-XL scripts, both from https://github.com/Edoar-do/HuBERT-ECG. The training loop, classification wrapper and comparison analysis are our own, adapted from our Check-in 2 CNN notebook.

All code was run by us on the real data, and all numbers come from our notebook outputs. We reviewed and edited the AI-assisted code and text, and we are responsible for the final content.

---

## 7. References

Coppola, E., Savardi, M., Massussi, M., Adamo, M., Metra, M., & Signoroni, A. (2024). HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications. *medRxiv*. https://doi.org/10.1101/2024.11.14.24317328

Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM TASLP*, 29, 3451–3460. https://doi.org/10.1109/TASLP.2021.3122291

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*. https://doi.org/10.1038/s41597-020-0495-6

Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE JBHI*, 25(5), 1519–1528. https://doi.org/10.1109/JBHI.2020.3022989

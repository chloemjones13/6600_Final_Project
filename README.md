# Check-in 1 — DSAN 6600

**Sex-Matched vs. More Data: What Closes the Subgroup Gap in ECG Classification?**

Chloe Jones · Olivia Semien · Due Sept 30, 2026

---

## 1. Problem framing + scope

**Task.** Multi-label classification of 12-lead ECGs into PTB-XL's five diagnostic superclasses (NORM, MI, STTC, CD, HYP). We hold the model fixed and vary the *training data* — how much of it, and how much comes from women — then measure performance on male and female test sets separately.

**Questions.**

- **1 — Trade-off.** At a fixed budget of N samples, is female performance better served by all-female data or by a mixed set? We try to state an exchange rate: how many mixed samples equal one sex-matched one.
- **2 — Does scale fix it?** The default answer to a subgroup gap is "collect more data." We train across a ladder of sizes and check whether the male–female gap actually shrinks. A flat line means more data doesn't fix it.
- **3 — Why?** If a gap survives, is it a different learned representation (needs a separate model) or a mis-set threshold (needs recalibration)? Tested by recalibrating on women only.

**Success criteria.** We succeed if we produce (1) a gap-vs-size plot with error bars, (2) an answer on the RQ1 trade-off, (3) a recalibration result, and (4) honest reporting against seed noise — including "the effect is smaller than the noise," which is a real finding, not a failure. We are not claiming a new architecture.

**Scope.** Open dataset, no credentialing. 100 Hz signals train in minutes per run. Grid bounded at ~45 runs. Work splits cleanly into a data track and a modeling track.

**Why us.** Both of us care about what happens when a model meets a population it wasn't trained on. ECG is a case where the clinical criteria already have that problem — thresholds derived from male-dominated cohorts, women's cardiac events documented as underdiagnosed. A pooled AUROC can look healthy while the female subgroup underneath it doesn't.

---

## 2. Dataset access + documentation

**PTB-XL v1.0.3** — PhysioNet. 21,799 clinical 12-lead ECGs from 18,869 patients, 10 seconds each, at 100 Hz and 500 Hz.

| | |
|---|---|
| **Source** | https://physionet.org/content/ptb-xl/1.0.3/ |
| **Size** | 1.7 GB zipped / 3.0 GB unzipped; we use the 100 Hz subset |
| **License** | Creative Commons Attribution 4.0 — open access, no credentialing |
| **Reading** | `wfdb` Python package |

**Why it fits.** `sex` and `age` on every record — the two variables our design needs. `strat_fold` gives 10 stratified folds that keep each patient's records together, so patient-level separation is handled for us. Folds 9–10 received human validation and are our test set.

**Access.**

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
pip install wfdb
```

Data is gitignored. The repo carries code and the metadata CSV only, so anyone can clone and reproduce the pull in one line.

---

## 3. Data audit / EDA

*See `notebooks/eda.ipynb`.*

**What we looked at.**

- Sex and age distributions, overall and crossed
- Class prevalence per superclass, split by sex (multi-label, so co-occurrence too)
- Records per sex × age band × superclass — the feasibility table
- Device and recording site crosstabbed against sex
- Representative 12-lead waveform plots, one per superclass
- Signal-quality flags and human-validation coverage

**What matters for the design.**

*Feasibility.* The training ladder tops out wherever the scarcest sex × age × class cell runs dry after matching. This table sets the real ceiling on our grid, and we'll adjust the ladder to whatever it says.

*Confounds.* Men and women in a clinical cohort differ in age and in disease base rates. Comparing unmatched groups would measure epidemiology, not physiology — so training cells are matched on age and class prevalence, and we report what we couldn't balance. We also check device and site against sex, since a device artifact correlated with sex would look exactly like a sex effect.

*Failure modes.* NORM dominates the label distribution, so accuracy is uninformative and rare classes will be noisy. Not every record is human-validated. Some carry noise and baseline-drift flags.

---

## 4. Evaluation plan

**Metrics.** Macro AUROC and AUPRC, reported separately for male and female test sets. Not accuracy — class imbalance makes it meaningless.

- **Gap metric:** male-minus-female AUROC, plotted against training size. This is the answer to RQ2.
- **Calibration:** reliability curves and Expected Calibration Error per subgroup. A model can match AUROC across groups while under-predicting risk in one of them at a fixed threshold — that's the failure mode that reaches patients.
- **Per-class breakdown:** the trade-off likely differs between MI and conduction disturbance; a pooled average would hide it.
- **Age stratification:** ECG sex signal is reported to weaken with age, predicting a larger gap in younger patients. We test it directly.

**Splits.** Patient-disjoint throughout via `strat_fold`. Folds 1–8 training, one fold held out for validation and model selection, folds 9–10 test. Test sets are fixed before anything else and never touched.

**Noise.** Three seeds per cell, varying both the data subset and weight initialization. Everything reported as means with error bars.

---

## 5. Initial direction

**Primary (Check-in 2).** A 1D convolutional network over the raw 12-lead signal — residual blocks, roughly 5–10 layers, global average pooling, sigmoid outputs for multi-label. This family is the standard for PTB-XL and fast enough at 100 Hz to make ~45 runs realistic.

Architecture stays **fixed** across every cell of the main grid. The independent variables are training size and composition; changing the model would confound them.

**Stretch arm — transfer learning.** RQ2 asks whether more data closes the gap. Pretraining is more data, borrowed rather than collected. If scale within PTB-XL doesn't close the gap, transfer might. We plan a reduced slice comparing three regimes:

| Regime | What trains |
|---|---|
| From scratch | Everything (our main grid) |
| Frozen | Pretrained encoder fixed, new head only |
| Fine-tuned | Pretrained encoder updated end to end |

Run at a subset of training sizes, not the full grid, so the primary comparison stays clean. The encoder would come from a publicly available self-supervised ECG model — these are largely 1D vision-transformer masked autoencoders, which is how a transformer enters the design.

Two conditions on this arm:

- **Leakage check is a prerequisite.** Several public ECG foundation models include PTB-XL in their pretraining corpus. Loading those weights would contaminate our held-out folds. We verify the pretraining corpus before using any checkpoint.
- **Expect a small effect.** Published scaling work on PTB-XL reports that pretraining helps rhythm and form tasks substantially but improves *diagnostic* labels by only about 0.024 AUROC, and that checkpoints pretrained on fewer than roughly 400,000 ECGs often fail to beat a non-pretrained control at all. We run this arm at our larger N values, where seed noise is lowest.

**Non-neural probe.** Logistic regression on simple extracted features, to establish a floor the neural model has to beat.

---

## 6. Video

~5 min, linked [HERE]()
---

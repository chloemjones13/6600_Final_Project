# Project Proposal — DSAN-6600

## Sex-Matched vs. More Data: What Actually Closes the Subgroup Gap in ECG Classification?

**Team:** Chloe Jones + Olivia Semien
**Check-in 1 due:** September 30, 2026
**Dataset:** PTB-XL v1.0.3 (PhysioNet, open access)

---

## 1. One-paragraph summary

Diagnostic ECG models are usually trained on pooled data and reported as a single number. When performance is broken out by sex, gaps often appear. The field's default assumption is that these gaps are a data-scarcity problem that more data will fix. We test that assumption directly. Using PTB-XL, we train diagnostic classifiers across a grid of training-set sizes and sex compositions, and measure whether the male–female performance gap shrinks as training data scales. We then ask whether any remaining gap is a *representation* problem or merely a *calibration* problem — a distinction that determines whether a practitioner needs two models or just two thresholds.

## 2. Why this matters

Diagnostic ECG criteria — voltage thresholds for left ventricular hypertrophy, ST-elevation cutoffs for myocardial infarction — were largely derived from male-dominated cohorts. Women's cardiac events are documented as underdiagnosed. Deep learning inherits this risk when models are trained on pooled data and evaluated on pooled metrics, because an aggregate AUROC can look healthy while the female subgroup underneath it does not.

Published work establishes that ECG carries strong sex-specific signal: models classify sex from raw ECG at roughly 0.93 AUROC, and that accuracy declines with patient age, suggesting sex-specific morphology attenuates over the lifespan. What is *not* established is what that signal costs a diagnostic model, or what fixes it.

## 3. Research questions

**RQ1 — The trade-off.** Given a fixed training budget, is performance on female patients better served by sex-matched training data or by a larger pooled set? Deliverable: how many additional pooled samples are worth one sex-matched sample.

**RQ2 — Scale invariance (primary novelty).** Does the male–female performance gap shrink as training data grows, or is it flat? If flat, the disparity is structural and collecting more data will not fix it.

**RQ3 — Mechanism.** If a gap persists, is it because the learned representation differs by sex, or because a shared representation needs a different decision threshold? Tested by post-hoc per-sex recalibration of the pooled model.

## 4. Dataset

**PTB-XL v1.0.3** — 21,799 clinical 12-lead ECGs from 18,869 patients, 10 seconds each, available at both 100 Hz and 500 Hz. Freely available on PhysioNet with no credentialing requirement.

Relevant properties:

- `sex` and `age` recorded for every record — the two variables our design needs.
- Diagnostic labels are multi-label SCP-ECG statements, aggregated into 5 superclasses (NORM, MI, STTC, CD, HYP).
- `strat_fold` provides 10 stratified folds that keep all of a patient's records within a single fold, so patient-level separation is handled for us.
- Folds 9 and 10 received at least one human evaluation and are therefore the highest-quality labels — we reserve these for test.
- We use the 100 Hz version (~1.7 GB), which fits Colab comfortably.

**Access:** direct download from PhysioNet; `wfdb` Python package for reading. No application, no wait.

## 5. Experimental design

### Held-out test sets (fixed first, never touched)

From folds 9–10, split into a male test set and a female test set. Every model is evaluated on both, separately. No model ever sees these records during training or model selection.

### Training grid

From folds 1–8, draw training sets varying two factors:

| Size (N) | Compositions |
|---|---|
| 500 | all-male, all-female, 50/50 |
| 1,000 | all-male, all-female, 50/50 |
| 2,000 | all-male, all-female, 50/50 |
| 4,000 | all-male, all-female, 50/50 |
| 7,500 | all-male, all-female, 50/50 |

Ceiling note: PTB-XL has roughly 9,000 patients per sex, so after the test holdout the all-female arm caps near 7,500. That sets the top of the ladder.

**Cohort matching.** Within each cell, training sets are matched on age distribution and diagnostic class prevalence. This is non-negotiable — men and women in a clinical cohort differ in base rates and age, and an unmatched comparison measures epidemiology rather than physiology. We report matched cohort characteristics and any residual imbalance we could not correct.

**Seeds.** Three seeds per cell, varying both the random training subset and the weight initialization. At N=500 the between-seed variance will likely exceed the effect we are measuring, so all results are reported as means with error bars. Total: 5 sizes × 3 compositions × 3 seeds = **45 runs**, plus the additional arms below.

### Additional arms

- **Pooled + sex as input feature.** The cheap fix a real deployment would try first. Does appending sex as a covariate recover what sex-specific training buys?
- **Pooled at full N (~15,000).** Reference upper bound, reported separately so it does not contaminate the size-matched comparison.
- **Per-sex recalibration (RQ3).** Take the trained pooled model, fit Platt scaling or isotonic regression on a small held-out female calibration set, and re-evaluate. If this recovers most of the gap, the answer for practitioners is "recalibrate," not "retrain."

## 6. Evaluation plan

- **Primary metrics:** macro AUROC and AUPRC, reported separately for the male and female test sets. Not accuracy — class imbalance makes it uninformative.
- **Gap metric:** male-minus-female AUROC, plotted against training size. This plot is the answer to RQ2.
- **Calibration:** reliability curves and Expected Calibration Error per subgroup. A model can hold AUROC across groups while systematically under-predicting risk in one of them at a fixed threshold, which is the failure mode that actually harms patients.
- **Per-class breakdown:** the trade-off is likely to differ between MI and conduction disturbance; a pooled average would hide that.
- **Age stratification:** published work reports that ECG sex signal weakens with age. That predicts our transfer gap should be larger in younger patients. We test it directly using PTB-XL's `age` field.
- **Splits:** patient-disjoint throughout, via `strat_fold`. Folds 1–8 for training, one held out for validation/model selection, folds 9–10 for test.

## 7. Planned neural approach (Check-in 2)

A 1D convolutional network over the raw 12-lead signal — ResNet-style residual blocks with 1D convolutions, roughly 5–10 layers, global average pooling, sigmoid outputs for multi-label classification. This architecture family is standard for PTB-XL and trains in minutes per run at 100 Hz on a Colab T4, which is what makes 45+ runs feasible.

Architecture is deliberately held **fixed** across all cells. The independent variables are training set size and composition; changing the model would confound them.

**Non-neural baseline (optional, Check-in 1):** logistic regression on simple extracted features, to establish a floor.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Effect is smaller than seed variance | Planned for: 3 seeds/cell, error bars. "The effect is below sampling noise at these scales" is a reportable result, not a failure. |
| Cohort matching leaves residual confounds | Report matched characteristics explicitly; state what we could not balance. |
| Null result on RQ2 | Either direction is publishable-shaped. "Scale closes the gap" and "scale does not close the gap" are both useful answers. |
| Compute overrun | 100 Hz signals, small model, checkpoint to Drive, runs are minutes not hours. Budget verified before scaling the grid. |
| Prior work already covers RQ2 | Literature search in week 1 (see §10). If covered for ECG, we reposition around RQ1/RQ3 and cite. |

## 9. Division of work

**Person A — data and infrastructure**
- PTB-XL download, `wfdb` loading, 100 Hz pipeline
- Cohort matching code (age + class prevalence stratified sampling)
- Grid sampler that generates the 45 training sets reproducibly from a seed
- EDA notebook: sex/age distributions, class prevalence by sex, matched cohort tables

**Person B — modeling and evaluation**
- 1D CNN implementation and training loop
- Experiment runner, checkpointing, results logging
- Metrics: AUROC/AUPRC by subgroup, calibration curves, ECE
- Learning-curve and gap plots

**Shared:** literature search, README/check-in write-up, ~5 min progress video.

The two tracks decouple after week 1 — Person B can develop against a small fixed subset while Person A builds the full sampler.

## 10. Timeline

**To Check-in 1 (Sept 30)**

- **Week 1:** Literature search (ECG subgroup fairness; scaling-and-fairness work in medical imaging — this is where prior work on RQ2 is most likely to exist). Download PTB-XL. Repo setup.
- **Week 2:** EDA — sex and age distributions, class prevalence by sex, missingness, signal quality flags. Cohort matching implemented and matched tables produced.
- **Week 3:** Logistic regression baseline. One pilot CNN run end-to-end to verify the pipeline. Write `check-in-1.md`. Record video.

**Check-in 1 deliverables:** repo, `check-in-1.md`, EDA notebook, dataset access notes, ~5 min video.

**Rest of project**

- Execute the full grid — 45 runs, 3 seeds per cell, results logged and checkpointed.
- Produce the learning curves and the male-minus-female gap plot (RQ1 and RQ2).
- Run the additional arms: pooled-plus-sex-feature, pooled at full N.
- Per-sex recalibration experiment (RQ3).
- Per-class and age-stratified breakdowns; subgroup calibration curves and ECE.
- Final write-up and presentation.

## 11. Open questions for discussion

1. Do we hold the model fixed at one architecture, or add a second (e.g. a small transformer) as a robustness check? Fixed is cleaner; two is more convincing if compute allows.
2. Do we include the pooled-plus-sex-feature arm at every N, or only at the largest? Every N is more informative and costs 15 more runs.
3. Who takes the video?

## 12. Key references

**Dataset**

1. Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., et al. (2020). PTB-XL: A large publicly available ECG dataset. *Scientific Data*. https://doi.org/10.1038/s41597-020-0495-6
2. Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). *PhysioNet*. https://doi.org/10.13026/kfzx-aw45
3. Moody, G., Pollard, T., & Moody, B. (2022). WFDB software package (version 10.7.0). *PhysioNet*. https://doi.org/10.13026/gjvw-1m31

**Closest related work — confirmed non-overlapping**

4. Steinbrinker, T. F. A., Hempel, P., Krefting, D., & Spicher, N. (2025). Explaining a sex-related cardiovascular risk continuum. *Current Directions in Biomedical Engineering*, 11(1), 310–313. https://doi.org/10.1515/cdbme-2025-0179

   Two-headed 1D ResNet predicting ECG-derived age and sex simultaneously; trained on CODE, validated on PTB-XL. Reports 0.93 AUROC for sex classification and 8.85 years MAE for age. **This paper predicts age and sex as targets. It does not perform diagnostic classification and reports no diagnostic performance stratified by sex — confirmed by reading the full text. No overlap with our research questions.**

   Three things we take from it:
   - **Age bands.** They use <40, 40–60, >60, justified by hormonal transition around menopause, and find sex-prediction error increases significantly across these groups (p < 0.001, two-sided t-test). We adopt the same boundaries for comparability.
   - **Directional prediction for our per-class analysis.** Their integrated-gradients analysis localizes sex-discriminative signal in lead V1 to the region after the QRS complex and before the T-wave (~350 ms) — the ST segment — and to the S-wave via narrower female QRS complexes. Because these overlap the diagnostic features for STTC, MI, and CD, we predict our sex transfer gap will be largest for those three superclasses and smallest for NORM and HYP.
   - **Cohort definition.** They exclude patients under 20 and over 85, yielding 20,370 ECGs at 46.64% female. We adopt the same exclusion.

**Fairness and confounds**

5. Liu, M., Ning, Y., Teixayavong, S., Mertens, M., Xu, J., et al. (2023). A translational perspective towards clinical AI fairness. *npj Digital Medicine*, 6(1), 172. https://doi.org/10.1038/s41746-023-00918-4

   Basis for our device/site crosstab in EDA: training biases such as device-specific artifacts can be present even when a training set appears balanced on sensitive variables. PTB-XL has `device` and `site` columns and we check these against sex before attributing any effect to sex.

**Clinical background**

6. Carbone, V., Guarnaccia, F., Carbone, G., Zito, G. B., & Oliviero, U. (2020). Gender differences in the 12-lead electrocardiogram: clinical implications and prospects. *The Italian Journal of Gender-Specific Medicine*, 6(3), 126–141. https://doi.org/10.1723/3432.34217

   Source for the physiological grounding: ST-elevation reported in ~90% of male vs ~20% of female ECGs; narrower QRS complexes in women.

**To be added**

7. Medical imaging subgroup fairness literature, specifically any work on whether subgroup gaps shrink with training scale — this is where prior work on RQ2 is most likely to exist. Week 1 search.

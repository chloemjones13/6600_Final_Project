Dataset access + documentation

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

Data is gitignored. The repo carries code and the metadata CSV only, so anyone can clone and reproduce the pull in two commands.

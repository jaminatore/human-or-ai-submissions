# Team 35 — Human or AI?

Kaggle team name: `team 35` (repository folder and branch use `team-35`).

Team members and NetIDs: **add before marking the draft PR ready for review**.

## Setup (PowerShell)

Use Python 3.14. Run these commands from `submissions/team-35`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the Kaggle `train.csv` and `test.csv` in `data/`. The repository ignores `data/` and CSV files, so the competition data stays local.

## Q1: our character TF-IDF logistic model

```powershell
.\.venv\Scripts\python.exe train.py --train data/train.csv --output-dir data/q1
```

Our team contribution is one model: character 3–5-gram TF-IDF with logistic regression. The script reserves one stratified 2,000-row validation set (seed `20261023`) and keeps it fixed while searching `C` over `{0.1, 1, 10}`. It selects `C` by validation AUROC. `data/q1/q1_search.csv` records every fit, including training and validation AUROC, error counts at a 0.5 threshold, and fit time. `q1_summary.json` records the split and selected setting. Other team members will provide the other approaches for the final Q1 comparison.

### Measured Q1 results

| Setting | Train AUROC | Validation AUROC | Train errors / 8,000 | Validation errors / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| C=0.1 | 0.963835 | 0.938598 | 808 | 260 |
| C=1 | 0.994568 | 0.965867 | 308 | 204 |
| C=10 | 0.999995 | 0.971249 | 14 | 180 |

These results use 8,000 training rows and 2,000 validation rows, stratified 4,000/4,000 and 1,000/1,000 by class. There were no exact text passages shared between the partitions. We chose `C=10` by validation AUROC. Since the same validation set selected `C`, its score may be somewhat optimistic.

## Refit and generate a local Kaggle submission

After reviewing the validation results, refit the selected approach on all 10,000 labeled rows and write 3,000 test probabilities:

```powershell
.\.venv\Scripts\python.exe train.py --train data/train.csv --output-dir data/q1 --refit-all
.\.venv\Scripts\python.exe predict.py --test data/test.csv --model data/q1/model.joblib --out data/submission.csv
```

The CSV columns are `id,label`, where `label` is the predicted probability that the passage is AI-generated. The file remains local until a team member reviews and uploads it. Q2–Q5 experiments are not part of this Q1 script yet.

## AI assistance

Codex helped set up the repository and draft the Q1 training script. Team members should run the experiment, inspect the data schema and results, review the code, and describe any changes and conclusions in the final report.

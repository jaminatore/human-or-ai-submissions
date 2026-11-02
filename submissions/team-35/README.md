# Team 35 — Human or AI?

Kaggle team name: `team 35` (repository folder and branch use `team-35`).

Team members and NetIDs: **add before opening the PR**.

## Setup (PowerShell)

Use Python 3.14. Run these commands from `submissions/team-35`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the Kaggle `train.csv` and `test.csv` in `data/`. The repository ignores `data/` and CSV files, so the competition data stays local.

## Q1: model comparison

```powershell
.\.venv\Scripts\python.exe train.py --train data/train.csv --output-dir data/q1
```

The script reserves one stratified 2,000-row validation set (seed `20261023`) and keeps it fixed for all candidates. It compares word unigram/bigram TF-IDF with logistic regression, character 3–5-gram TF-IDF with logistic regression, and word unigram/bigram TF-IDF with multinomial Naive Bayes. It searches `C` or `alpha` over `{0.1, 1, 10}` and selects the best setting for each approach by validation AUROC. `data/q1/q1_search.csv` records every fit; `q1_models.csv` contains the best row for each approach; `q1_summary.json` records the split and selected model. Training AUROC, validation AUROC, and fit time are measured on the same split.

### Measured Q1 results

| Approach | Chosen setting | Train AUROC | Validation AUROC | Fit time (s) |
| --- | ---: | ---: | ---: | ---: |
| Word TF-IDF + logistic regression | C=10 | 1.0000 | 0.9434 | 1.94 |
| Character TF-IDF + logistic regression | C=10 | 1.0000 | 0.9712 | 8.18 |
| Word TF-IDF + multinomial Naive Bayes | alpha=1 | 0.9896 | 0.9003 | 1.87 |

These results use 8,000 training rows and 2,000 validation rows, stratified 4,000/4,000 and 1,000/1,000 by class. There were no exact text passages shared between the partitions. We chose character TF-IDF + logistic regression (`C=10`) by validation AUROC. Hyperparameters and model family were both selected using this validation set, so the winning score may be somewhat optimistic. Fit times exclude vectorization of the training and validation data for scoring; they include vectorization and classifier fitting during `fit()`.

## Refit and generate a local Kaggle submission

After reviewing the validation results, refit the selected approach on all 10,000 labeled rows and write 3,000 test probabilities:

```powershell
.\.venv\Scripts\python.exe train.py --train data/train.csv --output-dir data/q1 --refit-all
.\.venv\Scripts\python.exe predict.py --test data/test.csv --model data/q1/model.joblib --out data/submission.csv
```

The CSV columns are `id,label`, where `label` is the predicted probability that the passage is AI-generated. The file remains local until a team member reviews and uploads it. Q2–Q5 experiments are not part of this Q1 script yet.

## AI assistance

Codex helped set up the repository and draft the Q1 training script. Team members should run the experiment, inspect the data schema and results, review the code, and describe any changes and conclusions in the final report.

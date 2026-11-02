# Team 35 — Human or AI?

Kaggle team name: `team 35` (repository folder and branch use `team-35`).

Team members and NetIDs: Zhitong Liu (zl2358). Add any other team members before marking the draft PR ready for review.

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

## Exploratory improvement: explicit wording cues

The [Wikipedia field guide on signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) suggests negative parallelism and the rule of three as possible cues. We tested whether counts of `not ... but`, `rather than`, and approximate three-item lists improve our character model. These regex counts are divided by passage length (per 100 words), standardized using training rows only, and appended to the TF-IDF features with a fixed weight of `0.1`. Both runs use the same 8,000/2,000 split and logistic `C=10`.

```powershell
.\.venv\Scripts\python.exe style_experiment.py --train data/train.csv --output data/style/style_ablation.json
```

| Configuration | Train AUROC | Validation AUROC | Validation errors at 0.5 | Fit time (s) |
| --- | ---: | ---: | ---: | ---: |
| Character TF-IDF alone | 0.999995 | 0.971249 | 180 | 8.04 |
| Character TF-IDF + wording cues | 1.000000 | 0.971205 | 173 | 9.95 |

The added cues did not improve validation AUROC, the competition metric, so the final prediction pipeline remains the simpler character TF-IDF model. The lower 0.5-threshold error count is a secondary result. In the training partition, the `not ... but` cue appeared in 5.8% of human and 7.3% of AI passages, and `rather than` appeared in 1.3% and 3.0%, respectively. Our approximate three-item-list pattern appeared in 52.6% of human passages and 45.6% of AI passages. This pattern does not capture every rhetorical rule of three; none of these cues establishes authorship by itself. The field guide also notes that some observations are specific to Wikipedia, so transfer to this dataset is uncertain.

## Refit and generate a local Kaggle submission

After reviewing the validation results, refit the selected approach on all 10,000 labeled rows and write 3,000 test probabilities:

```powershell
.\.venv\Scripts\python.exe train.py --train data/train.csv --output-dir data/q1 --refit-all
.\.venv\Scripts\python.exe predict.py --test data/test.csv --model data/q1/model.joblib --out data/submission.csv
```

The CSV columns are `id,label`, where `label` is the predicted probability that the passage is AI-generated. The file remains local until a team member reviews and uploads it. Q2–Q5 experiments are not part of this Q1 script yet.

## AI assistance

Codex helped set up the repository and draft the Q1 training and wording-cue experiment scripts. Team members should run the experiments, inspect the data schema and results, review the code, and describe any changes and conclusions in the final report.

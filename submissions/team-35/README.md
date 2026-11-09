# Team 35: Human or AI? Text Classification
**Cornell Tech | Applied Machine Learning (CS 5785)**

- **Team Members**: Zhitong Liu (`zl2358`), Sorathorn Thongpitukthavorn (`st2329`), Keyan Miao (`km2299`)
- **Kaggle Team**: `team 35`
- **Final Public Leaderboard AUROC**: **`0.98640`** ([`submission.csv`](submission.csv))
- **Primary Executable Entrypoint**: [`error_directed_model_refinement.ipynb`](error_directed_model_refinement.ipynb)

---

## 1. Directory Structure & File Map

```
submissions/team-35/
├── README.md                                  # Master team reproduction guide & report mapping
├── requirements.txt                           # Unified Python dependencies
├── .gitignore                                 # Ignore rules (unignoring models/)
├── submission.csv                             # Final winning Kaggle predictions (AUROC: 0.98640)
│
├── [Notebook Deliverables]
│   ├── model_selection_v3_high_score.ipynb    # §3: v3 multi-view ensemble (Kaggle: 0.98552)
│   ├── validation_gap_diagnosis.ipynb         # §6: Error diagnostics, length bins & domain shift
│   ├── final_model_improvements.ipynb         # §7: 9-specialist hybrid ensemble (Kaggle: 0.98629)
│   ├── error_directed_model_refinement.ipynb  # §7, §8: Final winning model refinement (Kaggle: 0.98640)
│   └── transformer_hybrid_ensemble.ipynb      # Exploration: DistilBERT hybrid testing (Kaggle: 0.98575)
│
├── models/                                    # Modular YAML hyperparameter configurations
│   ├── final_model_improvements.yaml
│   ├── final_ngram_lm.yaml
│   ├── final_nb_logreg.yaml
│   └── ...
│
├── report/                                    # Course report deliverables
│   ├── team_35_report.tex                     # LaTeX source for completed team report
│   ├── team_35_report.pdf                     # Compiled PDF report (11 pages)
│   └── figures/
│       └── baseline_learning_curve.pdf        # §4: Learning curve plot
│
├── zl2358/                                    # Zhitong's individual baseline scripts (§4, §5)
└── plum/                                      # Plum's individual exploration & artifacts
```

---

## 2. Report Section Mapping & Progress Milestones

The table below maps every section of the course report to its exact runnable code, findings, and metrics:

| Report Section | Focus | Runnable File / Entrypoint | Methodology & Role in Progression | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| **§2** | Final Model Summary | [`submission.csv`](submission.csv) | Error-directed refinement of 9-specialist ensemble | Kaggle **0.98640** |
| **§3** | Model Comparison & Selection | [`model_selection_v3_high_score.ipynb`](model_selection_v3_high_score.ipynb) | v3 multi-view ensemble (word/char TF-IDF, style, topic-masking) | Kaggle 0.98552 |
| **§4** | Training Data Size | [`zl2358/baseline_analysis.py`](zl2358/baseline_analysis.py) | Character TF-IDF learning curve ($N = 500 \to 8,000$) | Val 0.904 $\to$ 0.969 |
| **§5** | Unlabeled Data | [`zl2358/train.py`](zl2358/train.py) | Pseudo-labeling (1,112 rows at $p \le 0.05 / p \ge 0.95$) | Val 0.96845 (-0.0005) |
| **§6** | Generalization & Error Analysis | [`validation_gap_diagnosis.ipynb`](validation_gap_diagnosis.ipynb) | Holdout error autopsy, length bins, formatting sensitivity | 40% error drop |
| **§7 & §8** | Open-Ended Improvements | [`final_model_improvements.ipynb`](final_model_improvements.ipynb) | 9-specialist hybrid with dual generative n-gram LMs & GBDTs | Kaggle 0.98629 |
| **§7 & §8** | Final Winning Entrypoint | [`error_directed_model_refinement.ipynb`](error_directed_model_refinement.ipynb) | Blockquote unmasking (`> `) & rhyme/verse cadence features; outputs `submission.csv` | Kaggle **0.98640** |

### Note on Pre-trained Models:
- **[`transformer_hybrid_ensemble.ipynb`](transformer_hybrid_ensemble.ipynb)**: Explored fine-tuning a pre-trained contextual transformer (`distilbert-base-uncased`) blended with the GBDT ensemble. This was done for testing, but scored lower on the Kaggle public leaderboard (**`0.98575`** vs. `0.98640`) due to extreme logit bimodality displacing test rank calibrations. Because pre-trained models did not give a higher score, pre-trained models were abandoned in favor of our refined classical ensemble.

---

## 3. Reproduction Instructions (Quick Start)

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r submissions/team-35/requirements.txt
```
Place competition data (`train.csv`, `unlabeled.csv`, `test.csv`) in `./data/`.

### 2. Reproduce the Winning Submission (`0.98640`)
Open and run [`error_directed_model_refinement.ipynb`](error_directed_model_refinement.ipynb) end-to-end.
- Evaluates the 20 holdout misclassifications from Section 6.
- Applies blockquote unmasking (`> `), proper-noun repetition differentiation, and poetic rhyme/cadence extraction.
- Outputs: [`submission.csv`](submission.csv) (Kaggle public AUROC: **`0.98640`**).

### 3. Reproduce Section 3 Report Baseline (`0.98552`)
Open and run [`model_selection_v3_high_score.ipynb`](model_selection_v3_high_score.ipynb).

### 4. Reproduce Section 7 Model Improvements (`0.98629`)
Open and run [`final_model_improvements.ipynb`](final_model_improvements.ipynb).

---

## 4. Hardware, Environment & Seeds

- **Hardware**: Standard x86_64 Laptop / Workstation CPU (optionally DirectML/CUDA GPU).
- **Python Version**: 3.12
- **Core Dependencies**: `scikit-learn>=1.5.0`, `lightgbm>=4.3.0`, `catboost>=1.2.5`, `xgboost>=2.0.0`, `pandas>=2.2.0`, `numpy>=1.26.0`.
- **Random Seed**: Fixed to `42` across all data splitters, cross-validation folds, and model initializations for strict determinism.

---

## 5. Chronological Kaggle Submissions & Progression

```
0.86800  [Course baseline benchmark]
0.93670  [Local Reference Baseline: Word TF-IDF + Logistic Regression]
0.98379  [Synthetic formatting stress candidate (further_model_tuning.ipynb)]
0.98552  [model_selection_v3_high_score.ipynb (v3 multi-view ensemble)]
0.98629  [final_model_improvements.ipynb (9-specialist hybrid ensemble)]
0.98604  [best_recipe_seed_average.ipynb (3-seed replicate test)]
0.98640  [error_directed_model_refinement.ipynb (submission.csv - WINNING BEST SUBMISSION)]
0.98575  [transformer_hybrid_ensemble.ipynb (DistilBERT hybrid test - lower score)]
```

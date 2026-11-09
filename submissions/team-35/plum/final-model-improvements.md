# Model Improvement Progress & Empirical Diagnostic Report

**Project**: Human or AI? Text Classification (Cornell Tech CS 5785 / Kaggle)  
**Team**: Team 35 (`plum`)  
**Directory**: `submissions/team-35/plum/`  
**Date**: November 2026  
**Primary Metric**: Area Under the ROC Curve (AUROC)  
**Benchmark Reference Baseline**: Stratified 8,000/2,000 Word $(1,2)$ TF-IDF + Logistic Regression (Val AUROC: `0.9367`, Kaggle Benchmark: `~0.868`)

---

## 1. Executive Summary & Current Standings

Across iterative architectural iterations, feature engineering, and model exploration, our best-performing submissions on the Kaggle public leaderboard are:

1. **`submission_final_model_improvements.csv`** — **`0.98629`** (Top Result to Date)
   - *Architecture*: 9-Specialist Two-Tier Hybrid Ensemble (Pure Lexical/LM View + Structural View) combining Regularized Logistic Regression, Calibrated LinearSVC, Dual Word/Char N-gram LMs, and GBDT Triad (LightGBM, CatBoost, XGBoost) trained on all 10,000 labeled samples.
2. **`submission_best_recipe_seedavg_3.csv`** — **`0.98604`** (Variance-Reduction Replication)
   - *Architecture*: Exact 3-seed arithmetic average ($\text{seeds} \in \{42, 43, 44\}$) of the winning 9-specialist recipe.
3. **Robustness-Constrained Candidates (`candidate_01`, `candidate_04`)** — **`0.98379`**, **`0.98363`**
   - *Architecture*: Models heavily regularized against synthetic formatting perturbations (`canonical`, `wrap60`, `wrap90`).

```
Kaggle AUROC Progress:
  0.86800  [Kaggle Baseline Benchmark]
  0.93670  [Local Reference Baseline: Word TF-IDF + LogReg]
  0.97430  [Multi-View Linear Baseline]
  0.98379  [Candidate 01: Formatting-Stress Robustness]
  0.98604  [3-Seed Average of Best Recipe]
  0.98629  [submission_final_model_improvements.csv - CURRENT BEST]
```

---

## 2. Winning Recipe Architecture (`0.98629`)

The winning model (`submission_final_model_improvements.csv`) was developed in `final_model_improvements.ipynb` and parameterized via `models/final_model_improvements_selected.yaml`. It addresses the problem by recognizing that human vs. AI text detection requires two complementary views:

### 2.1 Multi-View Representation
- **Pure Lexical & Subword Views (Formatting-Free)**:
  - Sublinear Word $(1,3)$-gram TF-IDF (85,000 features, capturing topical phrasing and transition discourse markers).
  - Word-boundary Character `char_wb` $(3,5)$-grams (95,000 features, capturing morphology, prefixes, suffixes, punctuation cadence).
  - Masked Token View (stripping stopwords/formatting to isolate syntactical flow).
  - 72 Dense Pure Stylometric Features (lexical diversity `log_ttr`, `root_ttr`, `yules_k`, repetition frequencies, sentence length coefficient of variation `cv_sent_len`, Coleman-Liau readability, POS ratios).
- **Dual N-gram Generative Language Models**:
  - Fold-locally trained Word Trigram and Character 3-gram generative language models (`final_ngram_lm.yaml`) calculating log-likelihood ratio features $\log P(x \mid \text{AI}) - \log P(x \mid \text{Human})$ and per-token perplexity differences.
- **Structural View**:
  - Paragraph counts, whitespace distributions, line length variances, structural entropy, and 24 genre-conditioned stylometric Z-scores.

### 2.2 Nine-Specialist Hybrid Ensemble
The architecture integrates 9 diverse specialist learners split into two decoupled pipelines:
1. **No-Format Specialists (`nofmt_weights`: 0.62 total weight)**:
   - 4-View Logistic Regression (`lr_4v`, weight $0.22$, $C=2.0$)
   - 4-View Calibrated LinearSVC (`svc_4v`, weight $0.22$, $C=0.5$, Platt sigmoid calibration)
   - Masked-Text Logistic Regression (`mask_lr`, weight $0.14$)
   - Pure GBDT Triad blend (`gbdt_pure`, weight $0.42$: LightGBM $450$ trees, CatBoost $850$ trees, XGBoost $450$ trees)
2. **Structural Specialists (`struct_weights`: 0.38 total weight)**:
   - Structural LightGBM (`lgb_struct`, weight $0.30$)
   - Structural CatBoost (`cat_struct`, weight $0.40$)
   - Structural XGBoost (`xgb_struct`, weight $0.30$)

Final probabilities are fused via calibrated logit-space combination and retrained over the complete 10,000 labeled dataset.

---

## 3. Systematic Record of Subsequent Experiments

To push beyond `0.98629`, four focused experimental campaigns were conducted and recorded across dedicated notebooks in this directory:

### Experiment A: Formatting-Stress & Robustness Candidates (`further_model_tuning.ipynb`)
- **Hypothesis**: The competition assignment warns about synthetic distribution shifts (e.g. line-wrap, stripping `<br>`, canonicalizing quotes/dashes). Training models on augmented data and increasing structural invariance should prevent test-time performance collapse.
- **Implementation**:
  - Evaluated 20 distinct branch configurations over 5 grouped cross-validation folds.
  - Implemented real-time formatting stress evaluators: `canonical`, `typography`, `lowercase`, `wrap60`, and `wrap90`.
  - Selected 5 submission candidates (`candidate_01` to `candidate_05`) based on a balanced metric combining original AUROC and stress AUROC.
- **Kaggle Results**:
  - `candidate_01`: **`0.98379`** (vs `0.98629` baseline)
  - `candidate_04`: **`0.98363`**
- **Post-Mortem & Lesson**:
  - Aggressive formatting-invariance regularization significantly damaged test-set discriminative power ($-0.0025$ to $-0.0027$ AUROC).
  - The actual test distribution retains subtle structural and punctuation signals that artificial augmentation smoothed over or penalized excessively.

### Experiment B: Extended Boosting Iteration Scaling (`longer_training_iterations.ipynb`)
- **Hypothesis**: Tree-based gradient boosters (LightGBM, CatBoost, XGBoost) might underfit subtle tail interactions when constrained to 450–850 trees. Scaling iteration budgets ($2\times, 4\times, 8\times$) could improve classification margins.
- **Implementation**:
  - Scaled LightGBM to 3,600 trees, CatBoost to 6,800 iterations, and XGBoost to 3,600 rounds.
  - Monitored full 5-fold iteration curves and verified exact reproduction at the $1\times$ checkpoint.
- **Validation Outcome**:
  - Pooled local development AUROC increased monotonically from `0.99827` ($1\times$) to `0.99837` ($4\times$) and `0.99839` ($8\times$).
  - Mean holdout AUROC slightly improved from `0.999449` to `0.999486`.
- **Post-Mortem & Lesson**:
  - Despite local CV gains, formatting robustness slightly decreased (`0.98049` $\to$ `0.98003`), and submissions produced lower test rankings.
  - Boosting on lexical TF-IDF features simply Memorizes training-domain co-occurrences without improving out-of-domain generalization. Local CV at $\ge 0.998$ suffered severe saturation.

### Experiment C: Multi-Seed Averaging (`best_recipe_seed_average.ipynb`)
- **Hypothesis**: If tree models exhibit stochastic variance across different random seeds (subsampling, feature bagging, CatBoost random permutations), arithmetic probability averaging across independent runs of the winning recipe (`0.98629`) will reduce variance and raise test AUROC.
- **Implementation**:
  - Kept outer folds and all hyperparameter weights identical to the winning recipe.
  - Trained 5 full independent replicates with seeds $42, 43, 44, 45, 46$.
  - Averaged 3 seeds ($\{42, 43, 44\}$) and 5 seeds ($\{42, 43, 44, 45, 46\}$).
  - Verified numerical reconstruction of the seed 42 baseline within $2.2 \times 10^{-16}$.
- **Kaggle Result**:
  - `submission_best_recipe_seedavg_3.csv`: **`0.98604`** (vs `0.98629`, $-0.00025$).
- **Post-Mortem & Lesson**:
  - Seed averaging of tree models produces negligible difference ($\Delta < 0.0003$). Tree variance is *not* the limiting factor. The limiting factor is a fundamental representation and domain gap.

---

## 4. Diagnostic Root Cause: The Massive Test Domain Shift

In `validation_gap_diagnosis.ipynb`, a deep structural audit of the dataset revealed the exact reason why local validation AUROC ($\approx 0.9985$) diverges from Kaggle leaderboard AUROC ($\approx 0.9863$):

### 4.1 Observable Genre Support Matrix

| Genre / Heuristic Domain | Train Human % | Train AI % | Train Total % | Test Set % | **Domain Shift (pp)** |
|---|---:|---:|---:|---:|---:|
| **3. Poetry / Verse / Lyrics** | **0.02%** | 0.22% | **0.12%** | **14.40%** | **$+14.28$ pp** |
| **4. Book / Novel Summary** | 3.12% | 4.88% | **4.00%** | **13.93%** | **$+9.93$ pp** |
| 1. Recipe / Cooking | 20.72% | 20.92% | 20.82% | 12.77% | $-8.05$ pp |
| 2. Academic / ArXiv Abstract | 18.36% | 21.08% | 19.72% | 13.43% | $-6.29$ pp |
| 6. Encyclopedia / Wikipedia | 12.16% | 9.54% | 10.85% | 6.57% | $-4.28$ pp |
| 5. Film / TV Review | 8.32% | 5.52% | 6.92% | 3.93% | $-2.99$ pp |
| 7. First-Person / Social | 17.34% | 17.16% | 17.25% | 13.13% | $-4.12$ pp |
| 8. General / News / Other | 19.96% | 20.68% | 20.32% | 21.83% | $+1.51$ pp |

### 4.2 Key Findings
1. **The Poetry Blind Spot**:
   - In `train.csv` (10,000 rows), Poetry represents only **0.12%** (12 passages total, with only **1 human poem** in the entire training split!).
   - In `test.csv` (3,000 rows), Poetry constitutes **14.40%** (~432 passages)!
   - Sparse N-grams and surface stylometrics trained on recipes and academic abstracts completely fail to generalize to poetic cadence, meter, stanza formatting, and rhyming patterns.
2. **The Book Summary Shift**:
   - Book summaries triple from 4.0% in train to **13.93%** in test.
   - Combined, **Poetry and Book Summaries make up 28.33% of the Kaggle test set**, but only **4.12%** of the training data.
3. **Implication**:
   - Classical tabular models and bag-of-words classifiers have reached their mathematical limit on the supported 71.7% of the test set, but are operating near chance or with biased priors on the remaining 28.3%.
   - Any further hyperparameter tuning on the existing training distribution will only cause further overfitting to the dominant training genres (recipes, arxiv, wikipedia).

---

## 5. Artifact & Experiment Provenance

| Artifact / Notebook | Purpose / Contents | Primary Metric / Score | Status |
|---|---|---|---|
| `final_model_improvements.ipynb` | Canonical 9-specialist engine, multi-view TF-IDF, fold-local training | Kaggle: **0.98629** | Committed (Baseline Best) |
| `submission_final_model_improvements.csv` | Full 10k retrain submission of the winning 9-specialist model | Kaggle: **0.98629** | Committed |
| `further_model_tuning.ipynb` | Robustness-focused tuning across 5 stress modes, 20 configs | Kaggle: 0.98379 | Untracked $\to$ Staged |
| `longer_training_iterations.ipynb` | Tree iteration scaling ($2\times \dots 8\times$ depth/tree budgets) | Dev: 0.99839 / Kaggle: $\downarrow$ | Untracked $\to$ Staged |
| `validation_gap_diagnosis.ipynb` | Root cause discovery of 28.3% test domain shift (Poetry + Books) | Diagnostic Analysis | Untracked $\to$ Staged |
| `best_recipe_seed_average.ipynb` | Multi-seed replicate averaging ($\text{seeds } 42\dots 46$) | Kaggle: **0.98604** | Untracked $\to$ Staged |
| `submission_best_recipe_seedavg_3.csv` | 3-seed probability average CSV export | Kaggle: **0.98604** | Stored in `artifacts/` |

---

## 6. Next Strategic Directions

See the subsequent brainstorming analysis for our proposed structured experiment loop to break past 0.98629, addressing the domain shift via domain-specific representations, unsupervised feature transfer from test/unlabeled, and pretrained transformer hybridization.

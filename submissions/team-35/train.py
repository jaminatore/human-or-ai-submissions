"""Run the Q1 Human-or-AI baseline comparison on a fixed validation split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/q1"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--refit-all", action="store_true", help="Save the chosen model fitted on all labeled rows")
    return parser.parse_args()


def make_pipeline(name: str, value: float, seed: int) -> Pipeline:
    if name == "word_logreg":
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=100_000, sublinear_tf=True,
        )
        classifier = LogisticRegression(C=value, max_iter=1000, random_state=seed)
    elif name == "char_logreg":
        vectorizer = TfidfVectorizer(
            analyzer="char", lowercase=True, ngram_range=(3, 5), min_df=2,
            max_features=150_000, sublinear_tf=True,
        )
        classifier = LogisticRegression(C=value, max_iter=1000, random_state=seed)
    elif name == "word_nb":
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=100_000, sublinear_tf=True,
        )
        classifier = MultinomialNB(alpha=value)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.train)
    missing = {args.text_column, args.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(frame.columns)}")
    if frame[[args.text_column, args.label_column]].isna().any().any():
        raise ValueError("Text and label columns must not contain missing values")
    labels = pd.to_numeric(frame[args.label_column], errors="raise").to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels 0=human and 1=AI")
    if len(frame) <= args.validation_size:
        raise ValueError("Training data must exceed the validation size")
    texts = frame[args.text_column].astype(str).to_numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(frame)), test_size=args.validation_size,
        stratify=labels, random_state=args.seed,
    )
    x_train, x_val = texts[train_idx], texts[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    overlap = len(set(x_train).intersection(x_val))
    if overlap:
        print(f"Warning: {overlap} exact text passages occur in both partitions")

    grid = {
        "word_logreg": ("C", [0.1, 1.0, 10.0]),
        "char_logreg": ("C", [0.1, 1.0, 10.0]),
        "word_nb": ("alpha", [0.1, 1.0, 10.0]),
    }
    search_rows = []
    best_by_model = {}
    for name, (parameter, values) in grid.items():
        for value in values:
            model = make_pipeline(name, value, args.seed)
            start = time.perf_counter()
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - start
            train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
            val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
            row = {
                "model": name, "parameter": parameter, "value": value,
                "train_auc": float(train_auc), "validation_auc": float(val_auc),
                "train_seconds": round(train_seconds, 3),
            }
            search_rows.append(row)
            print(f"{name:12} {parameter}={value:<4} train={train_auc:.4f} val={val_auc:.4f} fit={train_seconds:.1f}s", flush=True)
            if name not in best_by_model or val_auc > best_by_model[name]["validation_auc"]:
                best_by_model[name] = row

    winner = max(best_by_model.values(), key=lambda row: row["validation_auc"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "q1_search.csv", index=False)
    pd.DataFrame(best_by_model.values()).to_csv(args.output_dir / "q1_models.csv", index=False)
    summary = {
        "seed": args.seed,
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "train_class_counts": np.bincount(y_train.astype(int), minlength=2).tolist(),
        "validation_class_counts": np.bincount(y_val.astype(int), minlength=2).tolist(),
        "exact_text_overlap": overlap,
        "selection_rule": "Highest AUROC on the fixed validation partition",
        "selected_model": winner,
        "note": "Validation was used for hyperparameter and model selection; these are not Kaggle test scores.",
    }
    (args.output_dir / "q1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.refit_all:
        model = make_pipeline(winner["model"], winner["value"], args.seed)
        model.fit(texts, labels)
        joblib.dump(model, args.output_dir / "model.joblib")
        print(f"Saved full-data model to {args.output_dir / 'model.joblib'}")
    print(f"Selected: {winner['model']} ({winner['parameter']}={winner['value']})")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()

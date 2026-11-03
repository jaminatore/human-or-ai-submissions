"""Reproduce Q2–Q4 for the character TF-IDF baseline."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train import make_pipeline


def main():
    data = Path(__file__).resolve().parent / "data"
    output = data / "baseline-analysis"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(data / "train.csv")
    train, val = train_test_split(frame, test_size=2000, stratify=frame.label, random_state=42)
    curve = []
    for size in (500, 1000, 2000, 5000, 8000):
        subset = train if size == len(train) else train_test_split(
            train, train_size=size, stratify=train.label, random_state=42
        )[0]
        model = make_pipeline(10, 42).fit(subset.text, subset.label)
        train_auc = roc_auc_score(subset.label, model.predict_proba(subset.text)[:, 1])
        scores = model.predict_proba(val.text)[:, 1]
        curve.append([size, train_auc, roc_auc_score(val.label, scores)])
        print(curve[-1], flush=True)

    unlabeled = pd.read_csv(data / "unlabeled.csv").dropna(subset=["text"])
    unlabeled = unlabeled.drop_duplicates("text")
    unlabeled = unlabeled.loc[~unlabeled.text.isin(frame.text)]
    probabilities = model.predict_proba(unlabeled.text)[:, 1]
    confident = (probabilities <= 0.05) | (probabilities >= 0.95)
    pseudo = unlabeled.loc[confident].assign(label=(probabilities[confident] >= 0.95).astype(int))
    augmented = pd.concat([train, pseudo], ignore_index=True)
    model = make_pipeline(10, 42).fit(augmented.text, augmented.label)
    semi_auc = roc_auc_score(val.label, model.predict_proba(val.text)[:, 1])

    errors = val.assign(probability=scores, words=val.text.str.split().str.len())
    errors["wrong"] = (scores >= 0.5) != errors.label
    bins = pd.cut(errors.words, bins=[0, 100, 200, np.inf], include_lowest=True)
    groups = []
    for group, rows in errors.groupby(bins, observed=True):
        groups.append([str(group), len(rows), roc_auc_score(rows.label, rows.probability), int(rows.wrong.sum())])

    wrong = errors.loc[errors.wrong]
    fp = wrong.loc[wrong.label == 0].nlargest(5, "probability")
    fn = wrong.loc[wrong.label == 1].nsmallest(5, "probability")
    examples = pd.concat([fp, fn])
    examples.to_csv(output / "examples-local.csv", index=False)
    result = dict(curve=curve, pseudo_counts=np.bincount(pseudo.label, minlength=2).tolist(),
                  semi_auc=semi_auc, length_groups=groups,
                  errors_by_label=np.bincount(wrong.label, minlength=2).tolist(),
                  examples=examples[["id", "label", "probability", "words"]].to_dict("records"))
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    sizes, training, validation = zip(*curve)
    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    ax.plot(sizes, training, marker="o", label="Training")
    ax.plot(sizes, validation, marker="s", label="Validation")
    ax.set(xlabel="Labeled training passages", ylabel="AUROC", xticks=sizes)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "baseline_learning_curve.pdf")


if __name__ == "__main__":
    main()

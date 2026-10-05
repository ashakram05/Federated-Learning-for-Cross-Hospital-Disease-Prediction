"""Run a centralized logistic-regression baseline on processed cardio CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/cardiovascular/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/cardiovascular/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/cardiovascular/centralized_baseline.json"))
    args = parser.parse_args()

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    target = "cardio"
    if target not in train or target not in test:
        raise ValueError(f"Both input files must contain target column {target!r}.")
    X_train, y_train = train.drop(columns=target), train[target]
    X_test, y_test = test.drop(columns=target), test[target]
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns do not match.")

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    result = {
        "experiment": "cardiovascular_centralized_logistic_regression",
        "target": target,
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "source": "Existing stratified 80/20 split from preprocessing notebook (random_state=42)",
        },
        "features": list(X_train.columns),
        "preprocessing": "Uses provided processed CSVs. Notebook fits StandardScaler on training data, then transforms test data.",
        "model": {"name": "LogisticRegression", "max_iter": 2000, "random_state": 42, "decision_threshold": 0.5},
        "metrics": {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix_labels_0_1": cm.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["no cardiovascular disease (0)", "cardiovascular disease (1)"],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "This is a centralized baseline on the provided processed dataset.",
            "The dataset is not hospital-partitioned; this result does not measure federated learning or cross-hospital generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "split": result["split"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()


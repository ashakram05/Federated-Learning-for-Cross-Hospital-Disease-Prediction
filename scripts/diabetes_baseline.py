"""Run matched centralized logistic-regression baselines for diabetes features."""
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

TARGET = "Diabetes_binary"
FEATURE_SETS = ("full", "reduced")


def run_baseline(data_dir: Path, output_dir: Path, feature_set: str) -> dict:
    train_path = data_dir / f"{feature_set}_train.csv"
    test_path = data_dir / f"{feature_set}_test.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)

    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError(f"Both {feature_set} input files must contain target column {TARGET!r}.")

    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"Train and test feature columns do not match for {feature_set!r}.")
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError(f"Both {feature_set} splits must contain target classes 0 and 1.")

    # The preprocessing notebook already scales continuous columns using
    # training-set statistics. Consume those processed values as provided.
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    result = {
        "experiment": "diabetes_centralized_logistic_regression",
        "dataset": "Diabetes health indicators dataset",
        "feature_representation": feature_set,
        "target": TARGET,
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "source": "Existing stratified 80/20 split from preprocessing notebook (random_state=42)",
        },
        "features": list(X_train.columns),
        "preprocessing": "Uses provided processed CSVs. Continuous features were scaled by the preprocessing notebook using training data only.",
        "model": {
            "name": "LogisticRegression",
            "max_iter": 2000,
            "random_state": 42,
            "decision_threshold": 0.5,
        },
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
            target_names=["no diabetes (0)", "diabetes (1)"],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "This is a centralized baseline on the provided processed dataset.",
            "The dataset is not hospital-partitioned; this result does not measure federated learning or cross-hospital generalization.",
        ],
    }
    output_path = output_dir / f"centralized_baseline_{feature_set}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "feature_representation": feature_set,
                      "split": result["split"], "metrics": result["metrics"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/diabetes"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diabetes"))
    args = parser.parse_args()

    for feature_set in FEATURE_SETS:
        run_baseline(args.data_dir, args.output_dir, feature_set)


if __name__ == "__main__":
    main()

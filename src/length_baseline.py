"""Length-only baseline for shortcut detection.

A logistic regression is fitted on normalised peptide length alone. The
training pool is split into 90% fit and 10% threshold-selection subsets,
stratified by label. The selected threshold is then frozen and the test
split is scored once. A high test score indicates that label is predictable
from length alone.

Usage for a pooled FASTA that contains train/validation/test records:
    python src/length_baseline.py \
        --data-dir data/genpept \
        --train genpept_all.fasta \
        --test genpept_all.fasta \
        --out results/length_baseline_genpept.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fasta_utils import parse_fasta
from metrics_utils import compute_classification_metrics, select_threshold

MAX_LENGTH = 200.0
REPORT_KEYS = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC")


def length_matrix(records) -> np.ndarray:
    return np.array(
        [[len(record.sequence) / MAX_LENGTH] for record in records],
        dtype=np.float64,
    )


def labels_of(records) -> np.ndarray:
    return np.array([record.label for record in records], dtype=np.int64)


def records_for_split(records, split: str, source: Path):
    subset = [record for record in records if record.split == split]
    if not subset:
        raise SystemExit(f"{source}: no records with split={split!r}")
    return subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="length-only baseline")
    parser.add_argument("--data-dir", default="data/genpept")
    parser.add_argument("--train", default="genpept_all.fasta",
                        help="FASTA containing the training-pool records")
    parser.add_argument("--test", default="genpept_all.fasta",
                        help="FASTA containing the test records")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/length_baseline.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train
    test_path = data_dir / args.test
    if not train_path.exists():
        raise SystemExit(f"missing training FASTA: {train_path}")
    if not test_path.exists():
        raise SystemExit(f"missing test FASTA: {test_path}")

    train_pool = records_for_split(parse_fasta(train_path), args.train_split, train_path)
    test_records = records_for_split(parse_fasta(test_path), args.test_split, test_path)
    labels = labels_of(train_pool)

    fit_index, val_index = train_test_split(
        np.arange(len(train_pool)),
        test_size=0.1,
        random_state=args.seed,
        shuffle=True,
        stratify=labels,
    )
    features = length_matrix(train_pool)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(features[fit_index], labels[fit_index])

    val_scores = model.predict_proba(features[val_index])[:, 1]
    tau, _ = select_threshold(labels[val_index], val_scores)

    test_labels = labels_of(test_records)
    test_scores = model.predict_proba(length_matrix(test_records))[:, 1]
    metrics = compute_classification_metrics(test_labels, test_scores, threshold=tau)

    payload = {
        "dataset": str(data_dir),
        "protocol": (
            "90:10 stratified fit/threshold split within the training pool; "
            "test scored once at the frozen threshold"
        ),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "n_train_pool": int(len(train_pool)),
        "n_fit": int(len(fit_index)),
        "n_val": int(len(val_index)),
        "n_test": int(len(test_records)),
        "seed": args.seed,
        "tau": float(tau),
        "test_metrics": metrics,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"length-only baseline on {data_dir}")
    print(f"  train pool {len(train_pool)} | fit {len(fit_index)} | "
          f"threshold-selection {len(val_index)} | test {len(test_records)}")
    print(f"  tau*           : {tau:.4f}")
    for key in REPORT_KEYS:
        print(f"  {key:<6}: {metrics[key]:.4f}")
    print(f"written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

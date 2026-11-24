"""Classical machine-learning baselines for the BiFAN revision.

The six estimators, their hyper-parameters, the input representation
(``Length_norm`` + AAC20 + DPC400 = 421 dimensions), the validation-selected
decision threshold and the metric set mirror the published
GenPept-Curated-2025 benchmark so that the numbers are directly comparable.

Usage:
    python baselines_ml.py --dataset genpept --seeds 100 101 102
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fasta_utils import parse_fasta
from metrics_utils import aggregate_metrics, compute_classification_metrics, select_threshold

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}
DPC_TO_INDEX = {
    first + second: i * 20 + j
    for i, first in enumerate(AA_ORDER)
    for j, second in enumerate(AA_ORDER)
}
MIN_LENGTH = 5      # the in-house corpus starts at 5 residues; GenPept starts at 10
MAX_LENGTH = 200
REPORT_KEYS = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC", "Brier", "ECE10")
DEFAULT_MODELS = (
    "logistic_regression",
    "linear_svm_calibrated",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
)


def extract_421_features(sequence: str) -> np.ndarray:
    """Official benchmark representation: Length_norm + AAC20 + DPC400."""
    seq = str(sequence).upper()
    if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
        raise ValueError(f"sequence length {len(seq)} outside [{MIN_LENGTH}, {MAX_LENGTH}]")
    if not set(seq).issubset(set(AA_ORDER)):
        raise ValueError("sequence contains non-canonical residues")

    features = np.zeros(421, dtype=np.float32)
    features[0] = len(seq) / float(MAX_LENGTH)
    for aa in seq:
        features[1 + AA_TO_INDEX[aa]] += 1.0
    features[1:21] /= float(len(seq))
    for position in range(len(seq) - 1):
        features[21 + DPC_TO_INDEX[seq[position] + seq[position + 1]]] += 1.0
    if len(seq) > 1:
        features[21:] /= float(len(seq) - 1)
    return features


def build_feature_matrix(records, representation: str) -> np.ndarray:
    if representation == "official421":
        return np.stack([extract_421_features(record.sequence) for record in records])
    if representation == "project559":
        from protein_extract import build_handcrafted_features

        features, _ = build_handcrafted_features([record.sequence for record in records], gap=3)
        return np.asarray(features, dtype=np.float32)
    raise ValueError(f"unknown representation: {representation}")


def make_estimator(model_id: str, seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if model_id == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                                         tol=1e-4, class_weight=None, random_state=None)),
        ])
    if model_id == "linear_svm_calibrated":
        base = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2", dual="auto", tol=1e-4,
                         max_iter=20000, class_weight=None, random_state=seed)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=folds,
                                            ensemble=True)
        return Pipeline([("scaler", StandardScaler()), ("model", calibrated)])
    if model_id == "random_forest":
        return RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=None,
                                      min_samples_split=2, min_samples_leaf=1,
                                      max_features="sqrt", bootstrap=True, class_weight=None,
                                      n_jobs=-1, random_state=seed)
    if model_id == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, criterion="gini", max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1,
                                    max_features="sqrt", bootstrap=False, class_weight=None,
                                    n_jobs=-1, random_state=seed)
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.05, max_iter=300,
                                              max_leaf_nodes=31, max_depth=None,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              max_bins=255, early_stopping=False,
                                              random_state=seed)
    if model_id == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                                  num_leaves=31, max_depth=-1, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  reg_alpha=0.0, reg_lambda=1.0, class_weight=None, n_jobs=-1,
                                  random_state=seed, deterministic=True, force_col_wise=True,
                                  verbosity=-1, device_type="cpu")
    raise KeyError(model_id)


def positive_probability(model, features: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(features), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = model[-1].classes_
    classes = list(classes)
    if proba.shape[1] == 1:
        return proba[:, 0] if int(classes[0]) == 1 else 1.0 - proba[:, 0]
    return proba[:, classes.index(1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="classical ML baselines (GenPept protocol)")
    parser.add_argument("--dataset", default="genpept")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--representation", default="official421",
                        choices=["official421", "project559"])
    parser.add_argument("--out-dir", default="results/ml")
    return parser.parse_args()


def summarize(rows) -> dict:
    summary = {}
    for model_id in dict.fromkeys(row["model_id"] for row in rows):
        subset = [row["test"] for row in rows if row["model_id"] == model_id]
        summary[model_id] = aggregate_metrics(subset, keys=REPORT_KEYS)
    return summary


def write_table(summary: dict, path: Path, digits: int = 4) -> None:
    header = ["model_id", "n"] + [f"{key}_mean" for key in REPORT_KEYS] + [f"{key}_std" for key in REPORT_KEYS]
    lines = [",".join(header)]
    for model_id, stats in summary.items():
        row = [model_id, str(stats["MCC"]["n"])]
        row += [f"{stats[key]['mean']:.{digits}f}" for key in REPORT_KEYS]
        row += [f"{stats[key]['std']:.{digits}f}" for key in REPORT_KEYS]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(summary: dict, digits: int = 4) -> None:
    print()
    print(f"{'model':<26}{'MCC':>16}{'ACC':>16}{'AUC':>16}{'BACC':>16}")
    ranked = sorted(summary.items(), key=lambda item: -item[1]["MCC"]["mean"])
    for model_id, stats in ranked:
        cells = ""
        for key in ("MCC", "ACC", "AUC", "BACC"):
            cells += f"{stats[key]['mean']:.{digits}f}+-{stats[key]['std']:.{digits}f}".rjust(16)
        print(f"{model_id:<26}{cells}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else Path("data") / args.dataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {name: parse_fasta(data_dir / f"{name}.fasta") for name in ("train", "val", "test")}
    labels = {name: np.array([record.label for record in records], dtype=np.int64)
              for name, records in splits.items()}
    for name in ("train", "val", "test"):
        print(f"{name:<5}: {len(splits[name])} records | positives {int(labels[name].sum())}")

    features = {name: build_feature_matrix(records, args.representation)
                for name, records in splits.items()}
    print(f"representation: {args.representation} {features['train'].shape}")
    print()

    rows = []
    for model_id in args.models:
        for seed in args.seeds:
            started = time.time()
            model = make_estimator(model_id, seed)
            model.fit(features["train"], labels["train"])

            val_probs = positive_probability(model, features["val"])
            tau, val_metrics = select_threshold(labels["val"], val_probs)
            test_probs = positive_probability(model, features["test"])
            test_metrics = compute_classification_metrics(labels["test"], test_probs, threshold=tau)
            elapsed = time.time() - started

            record = {
                "dataset": args.dataset,
                "representation": args.representation,
                "model_id": model_id,
                "seed": seed,
                "tau": tau,
                "seconds": elapsed,
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "val": val_metrics,
                "test": test_metrics,
            }
            stem = f"{args.dataset}_{model_id}_seed{seed}"
            (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            np.savez(out_dir / f"{stem}.npz", val_probs=val_probs, test_probs=test_probs,
                     val_labels=labels["val"], test_labels=labels["test"])
            rows.append(record)
            print(f"{model_id:<24} seed {seed:<4} tau {tau:.3f} | MCC {test_metrics['MCC']:.4f} | "
                  f"AUC {test_metrics['AUC']:.4f} | BACC {test_metrics['BACC']:.4f} | {elapsed:.1f}s")

    summary = summarize(rows)
    stem = f"summary_{args.dataset}_{args.representation}"
    (out_dir / f"{stem}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                          encoding="utf-8")
    write_table(summary, out_dir / f"{stem}.csv")
    print_table(summary)
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate BiFAN, classical and deep baseline runs into one comparison table.

Each method is evaluated with the same frozen splits, validation-selected
decision threshold and metric definitions. Mean and sample standard deviation
are reported over the seeds available for that method. The paired Wilcoxon
test uses only seeds shared with BiFAN.

Usage:
    python src/aggregate_results.py \
        --dataset genpept --seeds 100 101 102 103 104 105 \
        --bifan-tag add --ml-dir results/ml --deep-dir results/deep \
        --representation official421 --out-dir results/summary
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics_utils import METRIC_KEYS, aggregate_metrics, format_mean_std, paired_wilcoxon

TABLE_KEYS = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC")
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "linear_svm_calibrated": "Calibrated linear SVM",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "lightgbm": "LightGBM",
    "mlp": "ESM-2 embedding + MLP probe",
    "amplify": "AMPlify (architecture-level re-implementation)",
    "bilstm": "BiLSTM on ESM-2 embeddings",
    "transformer": "Transformer + attention pooling (AMP-BERT style)",
}


def default_bifan_label(dataset: str) -> str:
    if dataset == "inhouse":
        return "BiFAN (ours; Additive + auxiliary loss, lambda = 0.3)"
    return "BiFAN (ours; Additive)"


def load_bifan(runs_dir: Path, dataset: str, tag: str, seeds) -> dict:
    available = {}
    for seed in seeds:
        path = runs_dir / f"bifan_{dataset}_{tag}_seed{seed}" / "test_metrics.json"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = dict(payload["test_at_tau_star"])
        metrics["tau_star"] = float(payload["tau_star"])
        available[seed] = metrics
    return available


def load_model_runs(results_dir: Path, dataset: str, seeds, representation=None) -> dict:
    per_model = {}
    if not results_dir.exists():
        return per_model
    for path in sorted(results_dir.glob(f"{dataset}_*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload.get("seed", -1))
        if seed not in seeds:
            continue
        if representation and payload.get("representation") != representation:
            continue
        model_id = payload.get("model_id")
        metrics = payload.get("test_at_tau_star", payload.get("test"))
        if not model_id or not metrics:
            print(f"[skip] incomplete result file: {path}")
            continue
        per_model.setdefault(str(model_id), {})[seed] = dict(metrics)
    return per_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build the main comparison table")
    parser.add_argument("--dataset", default="genpept")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--bifan-tag", default="add",
                        help="run tag after the dataset, e.g. add or add_aux0.3")
    parser.add_argument("--bifan-label", default=None,
                        help="display name for the BiFAN reference row")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--ml-dir", default="results/ml")
    parser.add_argument("--deep-dir", default="results/deep")
    parser.add_argument("--representation", default="official421",
                        help="keep classical baselines with this representation")
    parser.add_argument("--out-dir", default="results/summary")
    parser.add_argument("--metric", default="MCC", choices=list(METRIC_KEYS),
                        help="metric used for the paired Wilcoxon test")
    return parser.parse_args()


def make_row(method, display, kind, per_seed, bifan, metric):
    seeds_here = [seed for seed in bifan if seed in per_seed]
    summary = aggregate_metrics([per_seed[seed] for seed in seeds_here], keys=TABLE_KEYS)
    statistic, p_value = paired_wilcoxon(
        [bifan[seed][metric] for seed in seeds_here],
        [per_seed[seed][metric] for seed in seeds_here],
    )
    return {
        "method": method,
        "display": display,
        "kind": kind,
        "seeds": seeds_here,
        "n_seeds": len(seeds_here),
        "summary": summary,
        "p_value": p_value,
        "wilcoxon_statistic": statistic,
        "mean_tau": float(np.mean([per_seed[seed].get("threshold", np.nan)
                                   for seed in seeds_here])),
    }


def main() -> int:
    args = parse_args()
    seeds = list(args.seeds)
    bifan = load_bifan(Path(args.runs_dir), args.dataset, args.bifan_tag, seeds)
    if not bifan:
        raise SystemExit(
            f"no BiFAN run found under {args.runs_dir} for "
            f"dataset={args.dataset!r}, tag={args.bifan_tag!r}"
        )

    classical = load_model_runs(
        Path(args.ml_dir), args.dataset, seeds, representation=args.representation
    )
    deep = load_model_runs(Path(args.deep_dir), args.dataset, seeds)

    reference = make_row(
        "bifan",
        args.bifan_label or default_bifan_label(args.dataset),
        "reference",
        bifan,
        bifan,
        args.metric,
    )
    reference["p_value"] = float("nan")
    rows = [reference]
    for model_id, per_seed in sorted(classical.items()):
        rows.append(make_row(
            model_id,
            MODEL_NAMES.get(model_id, model_id),
            "classical",
            per_seed,
            bifan,
            args.metric,
        ))
    for model_id, per_seed in sorted(deep.items()):
        rows.append(make_row(
            model_id,
            MODEL_NAMES.get(model_id, model_id),
            "deep",
            per_seed,
            bifan,
            args.metric,
        ))
    rows.sort(key=lambda row: -row["summary"]["MCC"]["mean"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"table_comparison_{args.dataset}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "n", *TABLE_KEYS, "p_vs_reference"])
        for row in rows:
            cells = [row["display"], row["n_seeds"]]
            cells.extend(
                format_mean_std(
                    row["summary"][key]["mean"],
                    row["summary"][key]["std"],
                )
                for key in TABLE_KEYS
            )
            p_value = row["p_value"]
            cells.append("reference" if not np.isfinite(p_value) else f"{p_value:.4g}")
            writer.writerow(cells)

    json_path = out_dir / f"table_comparison_{args.dataset}.json"
    payload = {
        "dataset": args.dataset,
        "seeds": seeds,
        "metric_for_test": args.metric,
        "reference_method": args.bifan_label or default_bifan_label(args.dataset),
        "rows": [
            {
                "method": row["method"],
                "display": row["display"],
                "kind": row["kind"],
                "seeds": row["seeds"],
                "n_seeds": row["n_seeds"],
                "mean_tau": row["mean_tau"],
                "p_value_vs_bifan": (
                    None if not np.isfinite(row["p_value"]) else row["p_value"]
                ),
                "wilcoxon_statistic": (
                    None
                    if not np.isfinite(row["wilcoxon_statistic"])
                    else row["wilcoxon_statistic"]
                ),
                "summary": row["summary"],
            }
            for row in rows
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"dataset    : {args.dataset}")
    print(f"BiFAN seeds: {sorted(bifan)}")
    print(f"written    : {csv_path}")
    print(f"written    : {json_path}")
    print()
    print(f"{'method':<46}{'n':>4}{'MCC':>10}{'ACC':>10}{'AUC':>10}")
    for row in rows:
        print(
            f"{row['display']:<46}{row['n_seeds']:>4}"
            f"{row['summary']['MCC']['mean']:>10.4f}"
            f"{row['summary']['ACC']['mean']:>10.4f}"
            f"{row['summary']['AUC']['mean']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

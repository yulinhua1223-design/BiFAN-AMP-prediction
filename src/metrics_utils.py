"""Classification metrics, threshold selection and multi-seed aggregation.

The protocol follows the GenPept-Curated-2025 benchmark:

* the decision threshold is selected on the validation split by maximising MCC
  (BACC and F1 break ties, then closeness to 0.5);
* that threshold is frozen and applied unchanged to the test split;
* every method reports the same metric set so numbers stay comparable.

``compute_classification_metrics`` returns ACC, MCC, AUC, Sn, Sp, F1, Prec,
TP, TN, FP and FN together with AUPRC, BACC, Brier and ECE10.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

#: Metrics reported in the paper tables.
METRIC_KEYS: Tuple[str, ...] = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC")

#: Metrics that are computed by point estimate but averaged across seeds.
THRESHOLD_GRID_POINTS = 181


def _as_arrays(labels, probabilities) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(np.int64).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"labels and probabilities must share shape, got {labels.shape} and {probabilities.shape}"
        )
    if labels.size == 0:
        raise ValueError("empty input")
    return labels, probabilities


def confusion_counts(labels, predictions) -> Tuple[int, int, int, int]:
    """Return (tp, tn, fp, fn) for binary predictions."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn


def compute_binary_ece(labels, probabilities, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins (as used by the benchmark)."""
    labels, probabilities = _as_arrays(labels, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    total = labels.size
    ece = 0.0
    for index in range(n_bins):
        mask = bins == index
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """ACC, MCC, AUC, AUPRC, Sn, Sp, BACC, F1, Prec, Brier, ECE10 plus raw counts.

    If the split contains a single class the AUC and AUPRC are undefined; they
    keep the historical fallback of 0.5 so that per-fold averaging never sees NaN.
    """
    labels, probabilities = _as_arrays(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)

    tp, tn, fp, fn = confusion_counts(labels, predictions)

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sn = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * (precision * sn) / max(precision + sn, 1e-9)
    bacc = 0.5 * (sn + sp)

    denominator = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denominator

    brier = float(np.mean((probabilities - labels) ** 2))

    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    else:
        auc = 0.5
        auprc = float(labels.mean())

    return {
        "ACC": float(acc),
        "MCC": float(mcc),
        "AUC": auc,
        "AUPRC": auprc,
        "Sn": float(sn),
        "Sp": float(sp),
        "BACC": float(bacc),
        "F1": float(f1),
        "Prec": float(precision),
        "Brier": brier,
        "ECE10": compute_binary_ece(labels, probabilities, n_bins=10),
        "threshold": float(threshold),
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """Observed scores, their midpoints, and {0, 0.5, 1} (benchmark rule)."""
    unique = np.unique(np.asarray(probabilities, dtype=np.float64))
    if unique.size >= 2:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
    else:
        midpoints = np.array([], dtype=float)
    candidates = np.unique(np.concatenate([unique, midpoints, np.array([0.0, 0.5, 1.0])]))
    return candidates[(candidates >= 0.0) & (candidates <= 1.0)]


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    return_table: bool = False,
):
    """Pick the validation threshold that maximises MCC (benchmark rule).

    Ties are broken by BACC, then F1, then closeness to 0.5, then the lower tau.
    Returns ``(threshold, metrics_at_threshold)`` and, optionally, the full table.
    """
    labels, probabilities = _as_arrays(labels, probabilities)

    rows: List[Tuple[float, float, float, float, float, Dict[str, float]]] = []
    for tau in threshold_candidates(probabilities):
        metrics = compute_classification_metrics(labels, probabilities, threshold=float(tau))
        rows.append(
            (
                metrics["MCC"],
                metrics["BACC"],
                metrics["F1"],
                abs(float(tau) - 0.5),
                float(tau),
                metrics,
            )
        )
    rows.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))

    best = dict(rows[0][5])
    best["threshold"] = float(best["threshold"])
    if return_table:
        table = [
            {"tau": row[4], "MCC": row[0], "BACC": row[1], "F1": row[2]} for row in rows
        ]
        return best["threshold"], best, table
    return best["threshold"], best


def find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "MCC",
    thresholds: Iterable[float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Legacy single-metric sweep kept for backwards compatibility.

    Prefer :func:`select_threshold`, which implements the benchmark tie-breaking.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, THRESHOLD_GRID_POINTS)

    best_threshold = 0.5
    best_metrics = compute_classification_metrics(labels, probabilities, threshold=best_threshold)
    best_score = best_metrics.get(metric, best_metrics["MCC"])

    for threshold in thresholds:
        current_metrics = compute_classification_metrics(labels, probabilities, threshold=float(threshold))
        current_score = current_metrics.get(metric, current_metrics["MCC"])
        if current_score > best_score:
            best_threshold = float(threshold)
            best_score = current_score
            best_metrics = current_metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def aggregate_metrics(
    per_seed: Sequence[Dict[str, float]],
    keys: Sequence[str] | None = None,
    ddof: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed metric dicts into ``{key: {mean, std, n}}``.

    ``ddof=1`` reports the sample standard deviation.
    """
    if not per_seed:
        raise ValueError("per_seed must contain at least one metrics dict")
    keys = list(keys) if keys is not None else list(METRIC_KEYS)

    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([item[key] for item in per_seed if key in item], dtype=np.float64)
        if values.size == 0:
            continue
        std = float(np.std(values, ddof=ddof)) if values.size > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": std, "n": int(values.size)}
    return summary


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    """Format a paper-ready ``mean ± std`` string."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def paired_wilcoxon(scores_a: Sequence[float], scores_b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired per-seed scores.

    Returns ``(statistic, p_value)``. Requires scipy; if scipy is missing the
    function returns ``(nan, nan)`` instead of raising, so reporting code can
    degrade gracefully.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan"), float("nan")

    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if scores_a.shape != scores_b.shape:
        raise ValueError("paired inputs must have the same shape")
    if scores_a.size < 1 or np.allclose(scores_a, scores_b):
        return float("nan"), float("nan")

    statistic, p_value = wilcoxon(scores_a, scores_b)
    return float(statistic), float(p_value)


def _self_test() -> None:
    """Quick sanity check: ``python metrics_utils.py``."""
    rng = np.random.default_rng(0)
    labels = np.array([0] * 500 + [1] * 500)
    good = np.concatenate([rng.uniform(0.0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    noisy = rng.uniform(0.0, 1.0, 1000)

    print("perfect ranking      :", compute_classification_metrics(labels, (labels + 0.5) / 1.0))
    print("informative model    :", {k: round(v, 4) for k, v in compute_classification_metrics(labels, good).items()})
    print("random model         :", {k: round(v, 4) for k, v in compute_classification_metrics(labels, noisy).items()})
    tau, metrics = select_threshold(labels, good)
    print(f"selected threshold   : {tau:.4f} -> MCC {metrics['MCC']:.4f}, BACC {metrics['BACC']:.4f}")
    assert set(compute_classification_metrics(labels, good)).issuperset(
        {"ACC", "MCC", "AUC", "AUPRC", "Sn", "Sp", "BACC", "F1", "Prec", "Brier", "ECE10"}
    ), "metric keys changed"
    summary = aggregate_metrics([compute_classification_metrics(labels, good, threshold=t) for t in (0.4, 0.5, 0.6)])
    print("aggregated (3 seeds) :", {k: round(v["mean"], 4) for k, v in summary.items()})
    print("wilcoxon placeholder :", paired_wilcoxon([0.9, 0.91, 0.92], [0.8, 0.81, 0.82]))
    print("self test passed")


if __name__ == "__main__":
    _self_test()

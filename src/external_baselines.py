"""Evaluate released Macrel and amPEPpy predictors on the fixed BiFAN splits.

The third-party tools are not retrained. Their released weights are applied
unchanged, and one decision threshold is selected on the local validation
split and then applied once to the held-out test split. These are external
tool comparisons under a local protocol, not independent clinical validation.

Required external tools are not redistributed with BiFAN. Install them from
their upstream sources and pass their local locations explicitly.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from fasta_utils import SequenceRecord, parse_fasta
from metrics_utils import compute_classification_metrics, select_threshold


TOOLS = {
    "macrel": {
        "display": "Macrel v1.6.1 (released weights)",
        "default_threshold": 0.5,
    },
    "ampeppy": {
        "display": "amPEPpy v1.0 (pretrained random forest)",
        "default_threshold": 0.5,
    },
}
PROTOCOL = (
    "released weights; threshold selected on the local validation split; "
    "held-out test split scored once"
)


def _check_probability(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"external predictor returned an invalid probability: {value}")
    return value


def run_macrel(
    fasta: Path,
    workdir: Path,
    threads: int,
    executable: str,
) -> tuple[dict[str, float], dict[str, float]]:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "peptides",
        "-f",
        str(fasta),
        "-o",
        str(workdir),
        "-t",
        str(threads),
        "--keep-negatives",
        "--quiet",
        "--force",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "Macrel failed. Check the executable and installation.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    table = workdir / "macrel.out.prediction.gz"
    if not table.is_file():
        raise SystemExit(f"Macrel produced no prediction table in {workdir}")

    by_access: dict[str, float] = {}
    by_sequence: dict[str, float] = {}
    with gzip.open(table, "rt", encoding="utf-8") as handle:
        header = None
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            record = dict(zip(header, fields))
            probability = _check_probability(record["AMP_probability"])
            by_access[str(record["Access"])] = probability
            by_sequence[str(record["Sequence"]).strip().upper()] = probability
    return by_access, by_sequence


def run_ampeppy(
    fasta: Path,
    workdir: Path,
    threads: int,
    ampeppy_home: str,
) -> tuple[dict[str, float], dict[str, float]]:
    del threads  # amPEPpy inference in this adapter is single-process
    warnings.filterwarnings("ignore")
    home = Path(ampeppy_home).resolve()
    model_path = home / "amPEP.model"
    if not model_path.is_file():
        raise SystemExit(f"amPEPpy model not found: {model_path}")
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    try:
        from amPEPpy.amPEP import score as ampep_score
    except ImportError as error:
        raise SystemExit(
            f"could not import amPEPpy from {home}; pass --ampeppy-home to the checkout"
        ) from error

    with model_path.open("rb") as handle:
        classifier = pickle.load(handle)

    workdir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        with fasta.open("r", encoding="utf-8") as handle:
            features = ampep_score(handle)
    finally:
        os.chdir(previous)

    probabilities = classifier.predict_proba(features)[:, 1]
    by_access = {
        str(key): _check_probability(value)
        for key, value in zip(features.index, probabilities)
    }
    return by_access, {}


def score_split(
    tool: str,
    fasta: Path,
    workdir: Path,
    threads: int,
    macrel_executable: str,
    ampeppy_home: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[SequenceRecord]]:
    records = parse_fasta(fasta)
    if tool == "macrel":
        by_access, by_sequence = run_macrel(
            fasta, workdir, threads, macrel_executable
        )
    elif tool == "ampeppy":
        by_access, by_sequence = run_ampeppy(
            fasta, workdir, threads, ampeppy_home
        )
    else:
        raise ValueError(f"unknown tool {tool!r}")

    labels = np.asarray([record.label for record in records], dtype=np.int64)
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    scored = np.zeros(len(records), dtype=bool)
    matched = {"identifier": 0, "sequence": 0, "unmatched": 0}
    for index, record in enumerate(records):
        if record.identifier in by_access:
            probabilities[index] = by_access[record.identifier]
            matched["identifier"] += 1
            scored[index] = True
        elif record.sequence in by_sequence:
            probabilities[index] = by_sequence[record.sequence]
            matched["sequence"] += 1
            scored[index] = True
        else:
            matched["unmatched"] += 1
    return labels, probabilities, scored, matched, records


def _resolve_macrel(executable: str | None) -> str:
    if executable:
        path = Path(executable)
        if not path.is_file():
            raise SystemExit(f"Macrel executable not found: {path}")
        return str(path)
    resolved = shutil.which("macrel")
    if not resolved:
        raise SystemExit("Macrel is not on PATH; pass --macrel-executable")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate released third-party AMP predictors")
    parser.add_argument("--tools", nargs="+", default=["macrel"], choices=sorted(TOOLS))
    parser.add_argument("--datasets", nargs="+", default=["genpept", "inhouse"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="results/external")
    parser.add_argument("--work-dir", default="results/external/work")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--macrel-executable", default=None)
    parser.add_argument("--ampeppy-home", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    macrel_executable = _resolve_macrel(args.macrel_executable) if "macrel" in args.tools else ""
    if "ampeppy" in args.tools and not args.ampeppy_home:
        raise SystemExit("--ampeppy-home is required when amPEPpy is selected")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for tool in args.tools:
        for dataset in args.datasets:
            split_dir = Path(args.data_root) / dataset
            val_path = split_dir / "val.fasta"
            test_path = split_dir / "test.fasta"
            if not val_path.is_file() or not test_path.is_file():
                print(f"[skip] {dataset}: val.fasta or test.fasta is missing", flush=True)
                continue

            started = time.time()
            val_labels, val_prob, val_scored, val_match, _ = score_split(
                tool,
                val_path,
                Path(args.work_dir) / f"{tool}_{dataset}_val",
                args.threads,
                macrel_executable,
                args.ampeppy_home or "",
            )
            test_labels, test_prob, test_scored, test_match, _ = score_split(
                tool,
                test_path,
                Path(args.work_dir) / f"{tool}_{dataset}_test",
                args.threads,
                macrel_executable,
                args.ampeppy_home or "",
            )
            if not val_scored.any() or not test_scored.any():
                raise SystemExit(f"{tool}/{dataset}: no validation or test records were scored")
            if len(np.unique(val_labels[val_scored])) < 2:
                raise SystemExit(f"{tool}/{dataset}: scored validation split has one class")

            tau, _ = select_threshold(val_labels[val_scored], val_prob[val_scored])
            test_metrics = compute_classification_metrics(
                test_labels[test_scored],
                test_prob[test_scored],
                threshold=float(tau),
            )
            default_tau = TOOLS[tool]["default_threshold"]
            default_metrics = compute_classification_metrics(
                test_labels[test_scored],
                test_prob[test_scored],
                threshold=float(default_tau),
            )
            payload = {
                "dataset": dataset,
                "model_id": tool,
                "display": TOOLS[tool]["display"],
                "protocol": PROTOCOL,
                "tau": float(tau),
                "default_threshold": float(default_tau),
                "n_val": int(val_scored.sum()),
                "n_val_unscored": int((~val_scored).sum()),
                "n_test": int(test_scored.sum()),
                "n_test_unscored": int((~test_scored).sum()),
                "matched_val": val_match,
                "matched_test": test_match,
                "test_at_tau_star": test_metrics,
                "test_at_default_threshold": default_metrics,
                "val_at_tau_star": compute_classification_metrics(
                    val_labels[val_scored],
                    val_prob[val_scored],
                    threshold=float(tau),
                ),
                "seconds": time.time() - started,
            }
            path = out_dir / f"{dataset}_{tool}_external.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary.append(payload)
            print(
                f"{dataset:<12} {tool:<8} tau={tau:.4f} MCC={test_metrics['MCC']:.4f} "
                f"AUC={test_metrics['AUC']:.4f} unscored={int((~test_scored).sum())}",
                flush=True,
            )

    (out_dir / "external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(summary)} result file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

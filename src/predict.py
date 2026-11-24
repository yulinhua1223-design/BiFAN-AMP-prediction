"""Run a trained BiFAN checkpoint on a local FASTA file.

The training entry point saves a pure ``state_dict`` as ``best_model.pth``,
the training-fitted scaler as ``scaler.pkl`` and the run settings as
``run_config.json``. This command loads that local bundle; it never downloads
model weights and never retrains the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from amp_model import AmpPredictor
from data_utils import AmpDataset, DynamicBatchCollator
from fasta_utils import SequenceRecord, parse_fasta
from fusion_variants import FUSION_NAMES
from protein_extract import build_handcrafted_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AMP/non-AMP probabilities with BiFAN")
    parser.add_argument("--checkpoint", required=True, help="path to best_model.pth")
    parser.add_argument("--esm-path", required=True, help="local ESM-2 model directory")
    parser.add_argument("--input", required=True, help="input FASTA")
    parser.add_argument("--output", required=True, help="output CSV")
    parser.add_argument("--scaler", default=None,
                        help="training-fitted scaler.pkl (default: next to checkpoint)")
    parser.add_argument("--config", default=None,
                        help="run_config.json (default: next to checkpoint when present)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="decision threshold (default: tau_star from test_metrics.json, else 0.5)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fusion", default=None, choices=list(FUSION_NAMES))
    parser.add_argument("--branch", default=None, choices=["full", "esm_only", "hand_only"])
    parser.add_argument("--aux-lambda", type=float, default=None)
    parser.add_argument("--cksaagp-gap", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=5,
                        help="reject sequences shorter than this many valid residues")
    parser.add_argument("--max-input-length", type=int, default=200,
                        help="reject sequences longer than this many valid residues")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("--device mps was requested, but MPS is not available")
    return device


def _read_json_if_present(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_dict(checkpoint: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint must contain a model state_dict")
    return payload


def _validate_records(
    records: list[SequenceRecord],
    min_length: int,
    max_length: int,
) -> None:
    if not records:
        raise SystemExit("input FASTA contains no valid records")
    if min_length < 1 or max_length < min_length:
        raise SystemExit("invalid input length bounds")
    invalid = [
        f"{record.identifier} ({len(record.sequence)} aa)"
        for record in records
        if not min_length <= len(record.sequence) <= max_length
    ]
    if invalid:
        examples = ", ".join(invalid[:5])
        raise SystemExit(
            f"{len(invalid)} sequence(s) fall outside {min_length}-{max_length} aa: {examples}"
        )


def _resolve_run_settings(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "run_config.json"
    config = _read_json_if_present(config_path)

    settings = {
        "fusion": args.fusion or config.get("fusion", "add"),
        "branch": args.branch or config.get("branch", "full"),
        "aux_lambda": args.aux_lambda
        if args.aux_lambda is not None
        else float(config.get("aux_lambda", 0.0)),
        "cksaagp_gap": args.cksaagp_gap
        if args.cksaagp_gap is not None
        else int(config.get("cksaagp_gap", 3)),
        "max_length": args.max_length
        if args.max_length is not None
        else int(config.get("max_length", 200)),
        "config_path": config_path,
    }
    return settings


def _resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        metrics_path = Path(args.checkpoint).parent / "test_metrics.json"
        metrics = _read_json_if_present(metrics_path)
        threshold = float(metrics.get("tau_star", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    return threshold


@torch.inference_mode()
def run_inference(
    records: list[SequenceRecord],
    model: AmpPredictor,
    tokenizer,
    scaler,
    cksaagp_gap: int,
    max_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[float]:
    sequences = [record.sequence for record in records]
    features, _ = build_handcrafted_features(
        sequences,
        gap=cksaagp_gap,
        include_length=False,
    )
    expected = int(getattr(scaler, "n_features_in_", -1))
    if features.shape[1] != expected:
        raise SystemExit(
            f"handcrafted feature mismatch: input has {features.shape[1]}, "
            f"scaler expects {expected}"
        )
    features = scaler.transform(features).astype(np.float32)

    dataset = AmpDataset(records, features, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DynamicBatchCollator(tokenizer, max_length=max_length),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    probabilities: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        handcrafted = batch["handcrafted_features"].to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            handcrafted_features=handcrafted,
        )["logits"]
        probabilities.extend(
            torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist()
        )
    return probabilities


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    esm_path = Path(args.esm_path)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not esm_path.is_dir():
        raise SystemExit(
            "ESM-2 must be an existing local directory; automatic downloads are disabled"
        )
    if not input_path.is_file():
        raise SystemExit(f"input FASTA not found: {input_path}")

    settings = _resolve_run_settings(args)
    threshold = _resolve_threshold(args)
    device = _resolve_device(args.device)

    scaler_path = Path(args.scaler) if args.scaler else checkpoint.parent / "scaler.pkl"
    if not scaler_path.is_file():
        raise SystemExit(f"scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, "transform") or not hasattr(scaler, "n_features_in_"):
        raise SystemExit("scaler file does not contain a fitted scikit-learn transformer")

    records = parse_fasta(input_path)
    _validate_records(records, args.min_length, args.max_input_length)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(str(esm_path), local_files_only=True)
    model = AmpPredictor(
        str(esm_path),
        int(scaler.n_features_in_),
        fusion=settings["fusion"],
        branch=settings["branch"],
        aux_lambda=settings["aux_lambda"],
    ).to(device)
    try:
        model.load_state_dict(_load_state_dict(checkpoint, device), strict=True)
    except RuntimeError as error:
        raise SystemExit(
            "checkpoint/model mismatch; supply the run_config.json written by training "
            f"or the matching --fusion/--branch/--aux-lambda/--cksaagp-gap values ({error})"
        ) from error
    model.eval()

    probabilities = run_inference(
        records=records,
        model=model,
        tokenizer=tokenizer,
        scaler=scaler,
        cksaagp_gap=settings["cksaagp_gap"],
        max_length=settings["max_length"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sequence_id", "probability", "prediction"])
        for record, probability in zip(records, probabilities):
            writer.writerow([
                record.identifier,
                f"{float(probability):.6f}",
                int(float(probability) >= threshold),
            ])

    print(
        f"wrote {len(records)} predictions to {output_path} "
        f"(threshold={threshold:.4f}, device={device})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

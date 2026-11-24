#!/usr/bin/env python3
"""Convert the frozen GenPept-Curated-2025 v1.1 CSV into this project's FASTA layout.

The project pipeline (data_utils.parse_fasta) expects headers of the form

    >{identifier}|{label}|{split}          with label in {0, 1}

Outputs (written to --outdir, default: data/genpept)

    train.fasta        7700 records
    val.fasta           990 records
    test.fasta         2310 records
    genpept_all.fasta 11000 records
    labels.csv         sample_id,label,split,length,length_bin   (metadata index)

Only the standard library is used, so this script runs on any machine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

FROZEN_SHA256 = "421e55265e1462052f633c961f8d8cc20ce5e510284ffc36afc6e29a40b79b6c"
# Both verified serializations preserve the same 11,000 IDs, sequences,
# labels and split assignments. See data/README.md for provenance.
VERIFIED_SHA256 = {
    FROZEN_SHA256,
    "bdfe03fbabbd0c7d6ff95ff42905680bb86ad0056e24cf964a33598aefb407a2",
}
EXPECTED_SPLIT_COUNTS = {"train": 7700, "val": 990, "test": 2310}
LABEL_TO_INT = {"AMP": 1, "non-AMP": 0}
SPLIT_ALIASES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None,
                        help="Canonical v1.1 CSV. Default: "
                             "<repo>/data/GenPept-Curated-2025/data/"
                             "GenPept_Curated_2025_primary_split_v1.1.csv")
    parser.add_argument("--outdir", default="data/genpept", help="Directory for the generated files.")
    parser.add_argument("--split-prefix", default="", help="Optional prefix for the FASTA file names.")
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args()


def sha256_of_normalized_bytes(path: Path) -> str:
    """Hash the file with line endings normalized to LF (git on Windows adds CR)."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def verify_frozen_dataset(csv_path: Path) -> None:
    digest = sha256_of_normalized_bytes(csv_path)
    if digest in VERIFIED_SHA256:
        print(f"[ok] known v1.1 serialization verified ({digest})")
        return
    raise SystemExit(
        f"[fatal] checksum mismatch for {csv_path}\n"
        f"        expected one of {', '.join(sorted(VERIFIED_SHA256))}\n"
        f"        actual   {digest}\n"
        f"        Re-download the frozen release (Zenodo 10.5281/zenodo.22994107)."
    )


def read_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "sequence", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"[fatal] CSV is missing required columns: {sorted(missing)}")
        return [row for row in reader]


def check_records(records: list[dict[str, str]]) -> None:
    seen_ids, seen_seq_by_split = set(), {}
    split_counts, label_counts = Counter(), Counter()
    for row in records:
        split = SPLIT_ALIASES.get(str(row["split"]).strip().lower())
        if split is None:
            raise SystemExit(f"[fatal] unexpected split value: {row['split']!r}")
        label = LABEL_TO_INT.get(str(row["label"]).strip())
        if label is None:
            raise SystemExit(f"[fatal] unexpected label value: {row['label']!r}")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise SystemExit("[fatal] a record has an empty sample_id")
        if sample_id in seen_ids:
            raise SystemExit(f"[fatal] duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        split_counts[split] += 1
        label_counts[label] += 1
        sequence = str(row["sequence"]).strip().upper()
        if not 10 <= len(sequence) <= 200:
            raise SystemExit(f"[fatal] {sample_id} length {len(sequence)} outside 10-200")
        if not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
            raise SystemExit(f"[fatal] {sample_id} contains non-standard residues")
        if sequence in seen_seq_by_split and seen_seq_by_split[sequence] != split:
            raise SystemExit(f"[fatal] identical sequence in two splits: {sample_id}")
        seen_seq_by_split[sequence] = split

    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if split_counts[split] != expected:
            raise SystemExit(f"[fatal] {split}: expected {expected} records, found {split_counts[split]}")
    if label_counts[1] != 5500 or label_counts[0] != 5500:
        raise SystemExit(f"[fatal] expected 5500/5500 class balance, found {dict(label_counts)}")
    print(f"[ok] {sum(split_counts.values())} records | "
          f"train/val/test = {split_counts['train']}/{split_counts['val']}/{split_counts['test']} | "
          f"AMP/non-AMP = {label_counts[1]}/{label_counts[0]}")


def write_fasta(rows: list[dict[str, str]], path: Path) -> None:
    lines = []
    for row in rows:
        label = LABEL_TO_INT[str(row["label"]).strip()]
        split = SPLIT_ALIASES[str(row["split"]).strip().lower()]
        lines.append(f">{row['sample_id']}|{label}|{split}")
        lines.append(str(row["sequence"]).strip().upper())
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def write_labels_csv(records: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "label", "split", "length", "length_bin"])
        for row in records:
            label = LABEL_TO_INT[str(row["label"]).strip()]
            split = SPLIT_ALIASES[str(row["split"]).strip().lower()]
            writer.writerow([row["sample_id"], label, split,
                             row.get("length", len(str(row["sequence"]))),
                             str(row.get("length_bin", "")).replace("-", "\u2013")])


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    if repo_root.name == "src":
        repo_root = repo_root.parent
    csv_path = Path(args.csv) if args.csv else (
        repo_root / "data" / "GenPept-Curated-2025" / "data"
        / "GenPept_Curated_2025_primary_split_v1.1.csv")
    if not csv_path.is_file():
        raise SystemExit(f"[fatal] canonical CSV not found: {csv_path}")

    if not args.skip_hash_check:
        verify_frozen_dataset(csv_path)

    records = read_records(csv_path)
    check_records(records)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in records:
        by_split[SPLIT_ALIASES[str(row["split"]).strip().lower()]].append(row)

    for split in ("train", "val", "test"):
        target = outdir / f"{args.split_prefix}{split}.fasta"
        write_fasta(by_split[split], target)
        print(f"[write] {target}  ({len(by_split[split])} records)")

    write_fasta(records, outdir / "genpept_all.fasta")
    print(f"[write] {outdir / 'genpept_all.fasta'}  ({len(records)} records)")

    write_labels_csv(records, outdir / "labels.csv")
    print(f"[write] {outdir / 'labels.csv'}")
    print("\nFASTA headers use >{sample_id}|{label}|{split}; ready for data_utils.parse_fasta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

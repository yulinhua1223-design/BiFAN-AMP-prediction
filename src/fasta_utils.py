"""Small, dependency-light FASTA helpers shared by training and prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from protein_extract import sanitize_sequence


@dataclass
class SequenceRecord:
    identifier: str
    label: int
    split: str
    sequence: str


def parse_fasta(path: str | Path) -> List[SequenceRecord]:
    """Parse ``>{id}|{label}|{split}`` FASTA records.

    The two metadata fields are optional so the same parser can be used for
    inference input. Labels default to 0 and splits default to the file stem.
    """
    path = Path(path)
    records: List[SequenceRecord] = []
    header = None
    chunks: List[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        parts = header.split("|")
        identifier = parts[0].strip()
        label = int(parts[1]) if len(parts) > 1 and parts[1].strip() else 0
        split = parts[2].strip() if len(parts) > 2 and parts[2].strip() else path.stem
        sequence = sanitize_sequence("".join(chunks))
        if identifier and sequence:
            records.append(SequenceRecord(identifier, label, split, sequence))
        header = None
        chunks = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:]
        else:
            chunks.append(line)
    flush()
    return records

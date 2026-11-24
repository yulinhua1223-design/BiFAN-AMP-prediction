from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from fasta_utils import SequenceRecord, parse_fasta
from protein_extract import build_handcrafted_features


class AmpDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SequenceRecord],
        handcrafted_features: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.records = list(records)
        self.handcrafted_features = handcrafted_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "sequence": record.sequence,
            "label": torch.tensor(record.label, dtype=torch.long),
            "handcrafted_features": torch.tensor(self.handcrafted_features[index], dtype=torch.float32),
            "identifier": record.identifier,
        }


class DynamicBatchCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
        sequences = [item["sequence"] for item in batch]
        tokenized = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.stack([item["label"] for item in batch])
        handcrafted = torch.stack([item["handcrafted_features"] for item in batch])
        identifiers = [item["identifier"] for item in batch]
        tokenized["labels"] = labels
        tokenized["handcrafted_features"] = handcrafted
        tokenized["identifiers"] = identifiers
        tokenized["raw_sequences"] = sequences
        return tokenized


def prepare_datasets(
    train_fasta: str | Path,
    test_fasta: str | Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    cksaagp_gap: int,
    valid_size: float,
    seed: int,
    include_length_feature: bool = False,
) -> tuple[AmpDataset, AmpDataset | None, AmpDataset, List[str], StandardScaler]:
    all_train_records = parse_fasta(train_fasta)
    test_records = parse_fasta(test_fasta)

    # valid_size=0 is used when an explicit validation split is supplied.
    if valid_size > 0:
        train_indices, valid_indices = train_test_split(
            np.arange(len(all_train_records)),
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=[record.label for record in all_train_records],
        )
        train_records = [all_train_records[idx] for idx in train_indices]
        valid_records = [all_train_records[idx] for idx in valid_indices]
    else:
        # Keep all training records and create no validation split.
        train_records = all_train_records
        valid_records = []
    # --------------------------------

    # Handcrafted descriptors
    train_features, feature_names = build_handcrafted_features(
        [record.sequence for record in train_records],
        gap=cksaagp_gap,
        include_length=include_length_feature,
    )
    
    # Extract validation descriptors only when a validation split exists.
    if valid_records:
        valid_features, _ = build_handcrafted_features(
            [record.sequence for record in valid_records],
            gap=cksaagp_gap,
            include_length=include_length_feature,
        )
    else:
        valid_features = np.empty((0, train_features.shape[1]))

    test_features, _ = build_handcrafted_features(
        [record.sequence for record in test_records],
        gap=cksaagp_gap,
        include_length=include_length_feature,
    )

    # Fit the scaler on training data only.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    if valid_records:
        valid_features = scaler.transform(valid_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    # Build datasets
    train_dataset = AmpDataset(train_records, train_features, tokenizer, max_length=max_length)
    valid_dataset = AmpDataset(valid_records, valid_features, tokenizer, max_length=max_length) if valid_records else None
    test_dataset = AmpDataset(test_records, test_features, tokenizer, max_length=max_length)
    
    return train_dataset, valid_dataset, test_dataset, feature_names, scaler


def build_dataset_from_fasta(
    fasta: str | Path,
    tokenizer: AutoTokenizer,
    scaler: StandardScaler,
    max_length: int,
    cksaagp_gap: int,
    include_length_feature: bool = False,
) -> AmpDataset:
    """Build the dataset of a single split with an already-fitted scaler.

    Validation and test splits must be standardised with the statistics of the
    training split. Fitting a separate scaler on an evaluation split leaks
    information and is exactly what this helper exists to prevent.
    """
    records = parse_fasta(fasta)
    features, _ = build_handcrafted_features(
        [record.sequence for record in records],
        gap=cksaagp_gap,
        include_length=include_length_feature,
    )
    features = scaler.transform(features).astype(np.float32)
    return AmpDataset(records, features, tokenizer, max_length=max_length)

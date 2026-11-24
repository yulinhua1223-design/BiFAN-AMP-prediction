from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}


def sanitize_sequence(sequence: str) -> str:
    return "".join([aa for aa in sequence.upper() if aa in AA_TO_INDEX])


def generate_group_pairs(group_keys: Iterable[str]) -> Dict[str, int]:
    pairs: Dict[str, int] = {}
    for key1 in group_keys:
        for key2 in group_keys:
            pairs[f"{key1}.{key2}"] = 0
    return pairs


def count_chars(chars: str, sequence: str) -> int:
    return sum(sequence.count(aa) for aa in chars)


def compute_aac(sequence: str) -> np.ndarray:
    sequence = sanitize_sequence(sequence)
    vector = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    if not sequence:
        return vector

    counts = Counter(sequence)
    length = float(len(sequence))
    for aa, idx in AA_TO_INDEX.items():
        vector[idx] = counts.get(aa, 0) / length
    return vector


def compute_dpc(sequence: str) -> np.ndarray:
    sequence = sanitize_sequence(sequence)
    vector = np.zeros(len(AMINO_ACIDS) * len(AMINO_ACIDS), dtype=np.float32)
    if len(sequence) < 2:
        return vector

    total = len(sequence) - 1
    for left, right in zip(sequence[:-1], sequence[1:]):
        index = AA_TO_INDEX[left] * len(AMINO_ACIDS) + AA_TO_INDEX[right]
        vector[index] += 1.0

    vector /= float(total)
    return vector


def compute_cksaagp(sequence: str, gap: int = 5) -> np.ndarray:
    sequence = sanitize_sequence(sequence)
    group = {
        "alphaticr": "GAVLMI",
        "aromatic": "FYW",
        "postivecharger": "KRH",
        "negativecharger": "DE",
        "uncharger": "STCPNQ",
    }

    aa_set = set(AMINO_ACIDS)
    group_keys = list(group.keys())
    index: Dict[str, str] = {}
    for key in group_keys:
        for aa in group[key]:
            index[aa] = key

    pair_index = [f"{key1}.{key2}" for key1 in group_keys for key2 in group_keys]
    features: List[float] = []

    for gap_idx in range(gap + 1):
        group_pair = generate_group_pairs(group_keys)
        total = 0
        for p1 in range(len(sequence)):
            p2 = p1 + gap_idx + 1
            if p2 < len(sequence) and sequence[p1] in aa_set and sequence[p2] in aa_set:
                pair_key = f"{index[sequence[p1]]}.{index[sequence[p2]]}"
                group_pair[pair_key] += 1
                total += 1

        if total == 0:
            features.extend([0.0] * len(pair_index))
        else:
            features.extend([group_pair[key] / total for key in pair_index])

    return np.asarray(features, dtype=np.float32)


def compute_ctdc(sequence: str) -> np.ndarray:
    sequence = sanitize_sequence(sequence)
    group1 = {
        "hydrophobicity_PRAM900101": "RKEDQN",
        "hydrophobicity_ARGP820101": "QSTNGDE",
        "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
        "hydrophobicity_PONP930101": "KPDESNQT",
        "hydrophobicity_CASG920101": "KDEQPSRNTG",
        "hydrophobicity_ENGD860101": "RDKENQHYP",
        "hydrophobicity_FASG890101": "KERSQD",
        "normwaalsvolume": "GASTPDC",
        "polarity": "LIFWCMVY",
        "polarizability": "GASDT",
        "charge": "KR",
        "secondarystruct": "EALMQKRH",
        "solventaccess": "ALFCGIVW",
    }
    group2 = {
        "hydrophobicity_PRAM900101": "GASTPHY",
        "hydrophobicity_ARGP820101": "RAHCKMV",
        "hydrophobicity_ZIMJ680101": "HMCKV",
        "hydrophobicity_PONP930101": "GRHA",
        "hydrophobicity_CASG920101": "AHYMLV",
        "hydrophobicity_ENGD860101": "SGTAW",
        "hydrophobicity_FASG890101": "NTPG",
        "normwaalsvolume": "NVEQIL",
        "polarity": "PATGS",
        "polarizability": "CPNVEQIL",
        "charge": "ANCQGHILMFPSTWYV",
        "secondarystruct": "VIYCWFT",
        "solventaccess": "RKQEND",
    }
    properties = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )

    if not sequence:
        return np.zeros(len(properties) * 3, dtype=np.float32)

    length = float(len(sequence))
    features: List[float] = []
    for prop in properties:
        c1 = count_chars(group1[prop], sequence) / length
        c2 = count_chars(group2[prop], sequence) / length
        c3 = max(0.0, 1.0 - c1 - c2)
        features.extend([c1, c2, c3])
    return np.asarray(features, dtype=np.float32)


def get_feature_names(gap: int = 5) -> List[str]:
    ctdc_properties = (
        "hydrophobicity_PRAM900101",
        "hydrophobicity_ARGP820101",
        "hydrophobicity_ZIMJ680101",
        "hydrophobicity_PONP930101",
        "hydrophobicity_CASG920101",
        "hydrophobicity_ENGD860101",
        "hydrophobicity_FASG890101",
        "normwaalsvolume",
        "polarity",
        "polarizability",
        "charge",
        "secondarystruct",
        "solventaccess",
    )
    ctdc_names = []
    for name in ctdc_properties:
        ctdc_names.extend([f"CTDC_{name}_g1", f"CTDC_{name}_g2", f"CTDC_{name}_g3"])

    cksaagp_groups = ["alphaticr", "aromatic", "postivecharger", "negativecharger", "uncharger"]
    cksaagp_names = []
    for gap_idx in range(gap + 1):
        for left in cksaagp_groups:
            for right in cksaagp_groups:
                cksaagp_names.append(f"CKSAAGP_gap{gap_idx}_{left}_{right}")

    aac_names = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    dpc_names = [f"DPC_{left}{right}" for left in AMINO_ACIDS for right in AMINO_ACIDS]
    return ctdc_names + cksaagp_names + aac_names + dpc_names


def build_handcrafted_features(
    sequences: Sequence[str],
    gap: int = 5,
    include_length: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    feature_names = get_feature_names(gap=gap)
    if include_length:
        feature_names = ["Sequence_Length"] + feature_names

    vectors = []
    for sequence in sequences:
        seq_len = len(sanitize_sequence(sequence))
        parts = [
            compute_ctdc(sequence),
            compute_cksaagp(sequence, gap=gap),
            compute_aac(sequence),
            compute_dpc(sequence),
        ]
        if include_length:
            parts.insert(0, np.array([seq_len], dtype=np.float32))
        vectors.append(np.concatenate(parts, axis=0))
        
    return np.asarray(vectors, dtype=np.float32), feature_names

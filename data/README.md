# Data

This directory contains the fixed train, validation, and test splits used by BiFAN. The training entry point uses the supplied files without random re-splitting.

## Format and counts

FASTA headers use `>{identifier}|{label}|{split}`, where label 1 denotes AMP and label 0 denotes the comparison class. Split values are `train`, `val`, and `test`.

| File | GenPept-Curated-2025 | In-house |
|---|---:|---:|
| `train.fasta` | 7,700 | 13,483 |
| `val.fasta` | 990 | 1,672 |
| `test.fasta` | 2,310 | 1,676 |
| Pooled FASTA | `genpept_all.fasta`: 11,000 | `all.fasta`: 16,831 |
| `test_length_matched.fasta` | 1,126 | 714 |

GenPept has 5,500 records per class and sequence lengths of 10–200 aa. In-house lengths are 5–50 aa. The length-matched files are balanced subsets of the respective test split with identical class-specific length distributions; they are not replacement primary test sets.

Additional files:

- [`genpept/labels.csv`](genpept/labels.csv): 11,000 sample IDs, labels, split assignments, lengths, and length bins.
- [`inhouse/split_report.json`](inhouse/split_report.json): construction parameters, split statistics, and cross-split audit summary.
- [`inhouse/attrition_step0.csv`](inhouse/attrition_step0.csv): preprocessing attrition through exact deduplication, ending at 16,836 records.

Five further records were removed during leakage repair, giving 16,831 final in-house records. The removal count is recorded as `n_sequences_dropped_for_leakage` in the split report.

## GenPept provenance and labels

The primary split is derived from **GenPept-Curated-2025 v1.1**:

- [Upstream repository](https://github.com/biochem-data-sci/GenPept-Curated-2025).
- [Dataset record and citation](https://doi.org/10.5281/zenodo.22994107).
- Canonical filename: `GenPept_Curated_2025_primary_split_v1.1.csv`.
- [Upstream source-and-rights statement](https://github.com/biochem-data-sci/GenPept-Curated-2025/blob/main/docs/SOURCE_AND_RIGHTS.md).

AMP labels reflect upstream annotation rules. The non-AMP class contains annotation-negative/unlabeled comparison records; it is not an experimentally confirmed inactive class.

The packaged FASTA preserves sample IDs, sequences, labels, and primary split assignments. Two byte-level versions of the canonical CSV were checked during release preparation:

| CSV serialization | SHA-256 after CRLF-to-LF normalization |
|---|---|
| Original BiFAN experiment copy | `421e55265e1462052f633c961f8d8cc20ce5e510284ffc36afc6e29a40b79b6c` |
| Upstream GitHub copy checked on 2026-11-24 | `bdfe03fbabbd0c7d6ff95ff42905680bb86ad0056e24cf964a33598aefb407a2` |

All 11,000 sample-ID/sequence/label/split tuples in both copies match the packaged pooled FASTA. Different full-file checksums therefore do not indicate a different BiFAN sequence split. The second checksum also matches the CSV in the Zenodo archive checked on the same date. The converter accepts these two known checksums and validates counts and sequence constraints. It normalizes length-bin dashes in the metadata index to the original BiFAN en-dash convention. It does not automatically approve future upstream changes.

To regenerate the primary files from a downloaded CSV:

```bash
python src/prepare_genpept_data.py --csv /path/to/GenPept_Curated_2025_primary_split_v1.1.csv --outdir outputs/genpept_regenerated
```

Run from the repository root and replace the CSV path. This writes `train.fasta`, `val.fasta`, `test.fasta`, `genpept_all.fasta`, and `labels.csv`; it does not regenerate the length-matched subset. The packaged FASTA files can also be used directly.

## In-house construction and audit

The in-house corpus combines records from APD3, dbAMP 2.0, DRAMP 4.0, and the non-AMP source pool described in the manuscript. The recorded preliminary grouping used **CD-HIT-EST**, followed by MMseqs2 similarity grouping and cross-split auditing. This documents the executable actually used; it is not a recommendation to use a nucleotide clustering program for new protein datasets.

The recorded clustering settings were `-c 0.90 -aS 0.80 -n 5 -l 5 -G 0 -g 1`. The independent MMseqs2 criterion was at least 90% identity and at least 80% coverage of both sequences (`--min-seq-id 0.9 -c 0.8 --cov-mode 0 -s 7.5 --max-seqs 100000`). The split report records:

- 13,889 final homology groups.
- Intact-group 80:10:10 partitioning, stratified by majority label and length bin, with seed 42.
- 11 repair rounds and zero cross-split hits at the stated MMseqs2 criterion.

Zero hits refers to that threshold and search configuration; it does not mean that every possible level of sequence relatedness is absent. The JSON is a saved report, not a script that reruns the audit. This version includes the final splits but not the full source-collection and split-construction workflow.

## Attribution and access

Upstream CC BY 4.0 applies to creator-owned GenPept curation, labels, split assignments, documentation, and compilation, to the extent of the creators' rights. Underlying third-party sequence rights remain with their holders; preserve attribution when reusing the data.

The original in-house source tables and complete per-sequence source mappings are not included. The packaged splits support rerunning training and evaluation; reconstructing source collection requires the original materials. For source-level enquiries, contact the corresponding author, Jie Shen, at [shenjie@hdu.edu.cn](mailto:shenjie@hdu.edu.cn).

The repository's MIT License applies to BiFAN source code and does not relicense third-party sequence records. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). File checksums are listed in [`SHA256SUMS.txt`](../SHA256SUMS.txt).

# BiFAN model card

## Model and availability

| Field | Value |
|---|---|
| Version | 1.0.0 |
| Task | Binary AMP-label classification |
| Sequence encoder | Fine-tuned ESM-2 650M |
| Descriptors | 559 composition and physicochemical features |
| Fusion | Additive fusion of projected branch representations |
| Output | AMP probability and thresholded binary prediction |

This version provides source code, fixed data splits, and aggregate results. It does not include a trained BiFAN checkpoint or a checkpoint download link. Train with `--keep-checkpoint` to obtain a prediction bundle; see [`README.md`](README.md).

## Evaluated configurations

| Dataset | Sequence length | Training configuration |
|---|---|---|
| GenPept-Curated-2025 v1.1 | 10–200 aa | Additive fusion; no auxiliary loss |
| In-house | 5–50 aa | Additive fusion; auxiliary descriptor loss with `lambda = 0.3` |

These configurations are trained separately. Each model must be used with its own fitted scaler, run configuration, and validation-selected threshold. GenPept labels follow annotation-based rules; the non-AMP class does not establish experimental inactivity. Data provenance and counts are documented in [`data/README.md`](data/README.md).

## Intended use

Research on AMP sequence classification, representation complementarity, and computational prioritization of candidates for further study. The binary output does not estimate minimum inhibitory concentration, toxicity, hemolysis, activity spectrum, or clinical efficacy.

## Inputs and preprocessing

Prediction accepts a local FASTA file. Headers may be `>{id}` or `>{id}|{label}|{split}`; labels are optional and are not used to generate probabilities.

The parser uppercases sequences and removes characters outside the 20 canonical residues `ACDEFGHIKLMNPQRSTVWY`. Empty cleaned records are omitted. Default prediction length bounds are 5–200 residues **after** cleaning. These software bounds are broader than the training domain of either individual configuration.

The recorded tokenizer setting is `max_length = 200` tokens, including special tokens. The sequence branch truncates inputs beyond the token budget. Descriptors use the full cleaned sequence. Preserve the recorded setting when reproducing results.

## Prediction artifacts and output

Keep these matching files together:

- `best_model.pth`: trained model state dictionary.
- `scaler.pkl`: scaler fitted on that run's training descriptors.
- `run_config.json`: architecture and feature settings.
- `test_metrics.json`: supplies the validation-selected `tau_star` threshold.

An existing local ESM-2 model/tokenizer directory is also required. The prediction script operates offline. When configuration metadata is absent, the code falls back to default settings; use the matching configuration to avoid architecture mismatches.

Output CSV columns are `sequence_id,probability,prediction`. Prediction is 1 when `probability >= threshold`. The threshold defaults to `tau_star` from the metrics file, or 0.5 if unavailable, and can be overridden with `--threshold`.

## Evaluation

Descriptor scaling uses training data only. Checkpoint selection and early stopping use validation MCC at threshold 0.5. The final decision threshold maximizes validation MCC, with ties resolved by BACC, F1, proximity to 0.5, and lower threshold. Test metrics are then calculated with the fixed threshold.

Main results report mean ± sample standard deviation over six seeds. The length-only control uses its own 90:10 training-pool fit/threshold split. [`results/README.md`](results/README.md) explains the reported tables, statistical comparisons, and error-coverage denominators.

## Applicability

Reported performance applies to the fixed evaluation splits and label definitions. Sequence composition, length, annotation practices, and dataset shift can affect predictions. Candidate activity and safety require separate evaluation; the classifier output alone does not establish either property.

## Attribution

BiFAN source code is MIT-licensed. Dataset and upstream-model terms are described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Software citation metadata is available in [`CITATION.cff`](CITATION.cff).

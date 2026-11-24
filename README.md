# BiFAN

**Additive fusion of protein language representations and physicochemical descriptors for antimicrobial peptide prediction.**

BiFAN is a binary antimicrobial peptide (AMP) classifier combining a fine-tuned ESM-2 encoder with 559 composition and physicochemical descriptors. The branches are projected into a shared representation and combined by additive fusion. Sequence-level error coverage is used to assess their complementarity.

Version `1.0.0` includes model and training code, baseline implementations, fixed data splits, and 18 reported result tables. Trained BiFAN checkpoints are not included or linked in this version. To predict sequences, first train a model with `--keep-checkpoint` as shown below.

## Contents

| Path | Contents |
|---|---|
| [`src/`](src/) | Model, features, training, prediction, baselines, and main comparison aggregation |
| [`data/`](data/README.md) | Fixed splits, length-matched test subsets, and in-house audit report |
| [`results/`](results/README.md) | Reported tables and column definitions |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Recorded environment and training settings |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model configuration, inputs, outputs, and intended use |
| [`CITATION.cff`](CITATION.cff) | Software citation metadata |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Data, model, and external-tool attribution |
| [`LICENSE`](LICENSE) | MIT License for BiFAN source code |
| [`SHA256SUMS.txt`](SHA256SUMS.txt) | Checksums of distributed files |

## Installation

Run commands from the repository root. Examples use Bash syntax. In PowerShell, put each multiline command on one line or replace each trailing backslash with a PowerShell backtick.

Python 3.10 is the recorded training version. The source syntax supports Python 3.10 or newer; the dependency ranges do not constitute a compatibility test of every version combination.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

Install a suitable PyTorch build using the [official selector](https://pytorch.org/get-started/locally/), then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The six-model classical baseline example includes LightGBM and requires:

```bash
python -m pip install -r requirements-optional.txt
```

Full ESM-2 fine-tuning is intended for a GPU; the reported runs used one RTX 4090 with bf16 autocast. CPU prediction is supported but can be slow. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for recorded versions and settings.

## Download ESM-2

Download the pretrained backbone and tokenizer:

```bash
python src/download_esm.py --target models/esm2_t33_650M_UR50D
```

The default model is `facebook/esm2_t33_650M_UR50D`. If a mirror is needed, append `--endpoint https://hf-mirror.com --disable-xet`. The download is several gigabytes and contains the pretrained backbone, not a trained BiFAN classifier.

## Fixed data splits

The packaged FASTA files can be used directly; rebuilding the datasets is not required for training.

| Dataset | Length (aa) | Train | Validation | Test | Length-matched test |
|---|---:|---:|---:|---:|---:|
| GenPept-Curated-2025 v1.1 | 10–200 | 7,700 | 990 | 2,310 | 1,126 |
| In-house | 5–50 | 13,483 | 1,672 | 1,676 | 714 |

The training entry point uses these fixed splits without random re-splitting. The length-matched files are test subsets for a separate control analysis. GenPept labels follow upstream annotation rules; its non-AMP class is an annotation-negative comparison class, not an experimentally confirmed inactive class.

[`data/README.md`](data/README.md) documents provenance, labels, audit thresholds, and verified CSV checksums. To optionally regenerate the primary GenPept files, obtain the canonical CSV from the [upstream dataset](https://doi.org/10.5281/zenodo.22994107) and use its actual local path:

```bash
python src/prepare_genpept_data.py \
  --csv /path/to/GenPept_Curated_2025_primary_split_v1.1.csv \
  --outdir outputs/genpept_regenerated
```

This conversion preserves the primary split and does not generate the separate length-matched subset.

## Train BiFAN

GenPept-Curated-2025:

```bash
python src/main1.py \
  --dataset genpept \
  --seed 100 \
  --esm-path models/esm2_t33_650M_UR50D \
  --fusion add \
  --tag add \
  --cksaagp-gap 3 \
  --max-length 200 \
  --batch-size 16 \
  --precision bf16 \
  --epochs 50 \
  --patience 3 \
  --lr 8e-5 \
  --weight-decay 0.01 \
  --keep-checkpoint
```

For the in-house dataset, use the same command with `--dataset inhouse`, `--tag add_aux0.3`, and the additional argument `--aux-lambda 0.3`.

Repeat each configuration with seeds `100 101 102 103 104 105` for the main six-seed comparison. For devices without bf16 support, use `--precision fp32`; adjust batch size if needed and record the changed configuration. On Windows, `--num-workers 0` can simplify DataLoader execution.

`--max-length 200` is the tokenizer **token limit**, including special tokens. Inputs exceeding that budget are truncated in the sequence branch; descriptors use the full cleaned sequence. Keep the recorded setting when reproducing reported runs.

Training writes to `runs/bifan_<dataset>_<tag>_seed<seed>/`:

| Artifact | Purpose |
|---|---|
| `best_model.pth` | Best-validation weights, retained only with `--keep-checkpoint` |
| `scaler.pkl` | Descriptor scaler fitted on training data |
| `run_config.json` | Run arguments and architecture settings |
| `test_metrics.json` | Validation-selected threshold, metrics, and run details |
| `predictions.npz` | Validation/test IDs, labels, and probabilities |
| `history.json` | Epoch-level training and validation metrics |

## Predict from a trained model

After the GenPept training command completes:

```bash
python src/predict.py \
  --checkpoint runs/bifan_genpept_add_seed100/best_model.pth \
  --esm-path models/esm2_t33_650M_UR50D \
  --input data/genpept/test.fasta \
  --output outputs/predictions.csv
```

Replace `--input` with your own FASTA file for candidate screening. Keep the checkpoint, scaler, configuration, and metrics file together. Prediction loads local artifacts without downloading models or training a classifier. For the in-house model, use its own `add_aux0.3` run directory.

Headers may be `>{id}` or `>{id}|{label}|{split}`; prediction labels are optional. The parser uppercases sequences and removes characters outside `ACDEFGHIKLMNPQRSTVWY`. Length validation is applied **after** cleaning, with default bounds of 5–200 residues. Records with no remaining residues are omitted. Supply canonical sequences to preserve the intended input exactly.

Output columns are `sequence_id,probability,prediction`. The binary prediction is 1 when the AMP probability is at least the threshold. By default, the threshold is validation-selected `tau_star` from the matching `test_metrics.json`; if unavailable, it is 0.5. Use `--threshold` to override it explicitly.

## Baselines and controls

### Classical models

The reported classical baselines use `Length_norm + AAC20 + DPC400` (421 features). Install the optional LightGBM dependency before running all six default models:

```bash
python src/baselines_ml.py \
  --dataset genpept \
  --seeds 100 101 102 103 104 105 \
  --representation official421 \
  --out-dir results/ml
```

Repeat with `--dataset inhouse`. To omit LightGBM, select the other models explicitly with `--models logistic_regression linear_svm_calibrated random_forest extra_trees hist_gradient_boosting`.

### Deep models

These architecture-level baselines use frozen ESM-2 token embeddings: MLP, a CNN identified by CLI name `amplify`, BiLSTM, and a transformer encoder. The CNN is not the original AMPlify attention-BiLSTM architecture; these are not official reproductions of published tools.

```bash
python src/deep_baselines.py \
  --datasets genpept inhouse \
  --models mlp amplify bilstm transformer \
  --seeds 100 101 102 \
  --esm-path models/esm2_t33_650M_UR50D \
  --data-root data \
  --out-dir results/deep
```

### Length-only control

```bash
python src/length_baseline.py \
  --data-dir data/genpept \
  --train genpept_all.fasta \
  --test genpept_all.fasta \
  --out results/length_baseline_genpept.json

python src/length_baseline.py \
  --data-dir data/inhouse \
  --train all.fasta \
  --test all.fasta \
  --out results/length_baseline_inhouse.json
```

Although both arguments name the pooled FASTA, the script selects `train` and `test` records by their header fields. It fits on 90% of the training pool and selects the threshold on the remaining 10%, stratified by label. It does not fit on test records or use the packaged validation split.

### External predictors

Install [Macrel](https://github.com/BigDataBiology/macrel) and [amPEPpy](https://github.com/tlawrence3/amPEPpy) separately. Their released weights are applied unchanged; a threshold is selected on the local validation split before test evaluation.

```bash
python src/external_baselines.py \
  --tools macrel ampeppy \
  --datasets genpept inhouse \
  --macrel-executable /path/to/macrel \
  --ampeppy-home /path/to/amPEPpy
```

Replace both paths with existing installations. amPEPpy and its serialized model must be compatible with the environment running this adapter.

## Aggregate a new main comparison

After completing the relevant BiFAN and baseline runs:

```bash
python src/aggregate_results.py \
  --dataset genpept \
  --seeds 100 101 102 103 104 105 \
  --bifan-tag add \
  --ml-dir results/ml \
  --deep-dir results/deep \
  --representation official421 \
  --out-dir outputs/comparison
```

Repeat with `--dataset inhouse --bifan-tag add_aux0.3`. Each command writes `table_comparison_<dataset>.csv` and `.json`. A method's summary uses seeds shared with available BiFAN runs. `p_vs_reference` tests MCC by a two-sided paired Wilcoxon signed-rank test unless `--metric` is changed.

Use a separate output directory to preserve the distributed reference tables. The aggregator generates the two main comparison tables. Other distributed tables are archived analysis summaries; this version does not include all scripts or original per-seed inputs needed to regenerate them. See [`results/README.md`](results/README.md).

## Reported performance

Values are mean ± sample standard deviation over six seeds on fixed test splits with validation-selected thresholds.

| Dataset | ACC | MCC | AUC | AUPRC |
|---|---:|---:|---:|---:|
| GenPept-Curated-2025 | 0.9531 ± 0.0035 | 0.9067 ± 0.0071 | 0.9815 ± 0.0019 | 0.9758 ± 0.0033 |
| In-house | 0.9435 ± 0.0028 | 0.8873 ± 0.0055 | 0.9770 ± 0.0040 | 0.9793 ± 0.0037 |

The in-house configuration uses auxiliary descriptor supervision (`lambda = 0.3`); the GenPept configuration uses no auxiliary term. These are separately trained models.

## Evaluation protocol

- Fixed train/validation/test files; descriptor scaler fitted on training data only.
- Checkpoint selection and early stopping by validation MCC at threshold 0.5.
- After checkpoint selection, maximize validation MCC to choose the decision threshold; ties use BACC, F1, proximity to 0.5, and then the lower threshold.
- Freeze this threshold for final test metrics. AUC and AUPRC use predicted probabilities.
- Apply the separate training-pool protocol above for the length-only control.

## Scope, attribution, and citation

BiFAN predicts binary AMP labels for research and candidate prioritization. It does not estimate potency, toxicity, organism-specific activity, or clinical efficacy. See the [model card](MODEL_CARD.md) for evaluated sequence domains.

Use [`CITATION.cff`](CITATION.cff) to cite this software version. No BiFAN article DOI or software DOI is assigned in this metadata. Cite original datasets and external tools as applicable. For GenPept-Curated-2025:

> Pham HT, Nguyen-Vo T-H, Huynh B. GenPept-Curated-2025, version 1.1. Zenodo. <https://doi.org/10.5281/zenodo.22994107>

BiFAN source code is distributed under the [MIT License](LICENSE). Third-party data and artifacts retain their source terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Verify file integrity

On systems with `sha256sum`, run this from the repository root:

```bash
sha256sum -c SHA256SUMS.txt
```

The manifest covers every distributed file except itself. It excludes Git metadata and locally generated artifacts. Text files use LF line endings.

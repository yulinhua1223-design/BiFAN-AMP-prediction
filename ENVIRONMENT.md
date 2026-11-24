# Experiment environment

## Recorded versions

The versions below were transcribed from the archived experiment dependency snapshot. They describe the reported runs, rather than a claim that every platform or dependency combination has been tested.

| Component | Recorded version or hardware |
|---|---|
| Python | 3.10 |
| PyTorch | 2.14.0 |
| Transformers | 5.17.0 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |
| SciPy | 1.15.3 |
| Joblib | 1.6.0 |
| Matplotlib | 3.10.9 |
| LightGBM | 4.7.0 |
| GPU | One NVIDIA GeForce RTX 4090 |

The installation ranges in [`requirements.txt`](requirements.txt) are dependency constraints, not a frozen environment. PyTorch should be installed for the local operating system and CUDA driver using its [official installation selector](https://pytorch.org/get-started/locally/). LightGBM is included in [`requirements-optional.txt`](requirements-optional.txt) and is required when running all six default classical models.

## Main training settings

| Setting | Value |
|---|---|
| Backbone | `facebook/esm2_t33_650M_UR50D` |
| Fine-tuning | End-to-end for BiFAN |
| Descriptors | CTDC39 + CKSAAGP100 + AAC20 + DPC400; no explicit length feature |
| CKSAAGP gap | 3 |
| Optimizer | AdamW |
| Learning rate / weight decay | `8e-5` / `0.01` |
| Batch size | 16 |
| Precision | bf16 autocast on CUDA |
| Maximum epochs / early-stopping patience | 50 / 3 |
| Gradient clipping | 1.0 |
| Dropout | 0.1 |
| Tokenizer maximum length | 200 tokens, including special tokens |
| Main seeds | 100, 101, 102, 103, 104, 105 |
| Auxiliary-loss weight | GenPept: 0; In-house: 0.3 |

Checkpoint selection uses validation MCC at threshold 0.5. After checkpoint selection, the decision threshold is optimized on validation data and fixed for test evaluation. The tokenizer can truncate the longest sequences; descriptors are computed from the complete cleaned sequence.

The deep architecture baselines use frozen ESM-2 embeddings and seeds 100–102. Their training defaults differ from BiFAN: see `python src/deep_baselines.py --help` and the recorded output JSON for each new run. The length-only control fits and selects its threshold using a 90:10 stratified subdivision of the training pool.

## External dependencies

- The pretrained ESM-2 files are downloaded separately.
- Macrel and amPEPpy are installed separately and retain their own dependency requirements. The adapter invokes Macrel as a command and loads amPEPpy in the current Python process.
- CD-HIT-EST and MMseqs2 belong to the historical in-house data-construction workflow. They are not needed to train on the packaged fixed FASTA splits. See [`data/README.md`](data/README.md) for the recorded construction and audit scope.

## Record a new environment

Save the actual package versions and hardware alongside each reproduction:

```bash
python --version
python -m pip freeze > environment-freeze.txt
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('CUDA', torch.version.cuda); print('CUDA available', torch.cuda.is_available())"
```

Changing hardware, precision, batch size, or library versions can change numerical results. Use the same data, run configuration, scaler, and validation-selected threshold when comparing runs.

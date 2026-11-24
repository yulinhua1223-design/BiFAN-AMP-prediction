# Reported results

[`summary/`](summary/) contains 18 final CSV tables accompanying BiFAN. These are archived experimental summaries, not the outputs of a fresh run. The original per-seed predictions and complete analysis pipeline are not included in this version.

## Table index

| File | Content |
|---|---|
| [`table_comparison_genpept.csv`](summary/table_comparison_genpept.csv) | BiFAN, six classical models, and four deep architecture baselines on GenPept |
| [`table_comparison_inhouse.csv`](summary/table_comparison_inhouse.csv) | Corresponding in-house comparison |
| [`table_deep_baselines.csv`](summary/table_deep_baselines.csv) | Deep architecture baselines by dataset |
| [`table_external_predictors.csv`](summary/table_external_predictors.csv) | Released Macrel and amPEPpy models with locally selected thresholds |
| [`table_fusion_comparison.csv`](summary/table_fusion_comparison.csv) | Six fusion operators on GenPept |
| [`table_branch_ablation.csv`](summary/table_branch_ablation.csv) | Full, ESM-2-only, and descriptor-only models on GenPept |
| [`table_branch_ablation_inhouse.csv`](summary/table_branch_ablation_inhouse.csv) | Corresponding in-house ablation |
| [`table_aux_supervision.csv`](summary/table_aux_supervision.csv) | Auxiliary-loss weights on GenPept |
| [`table_backbone_scale.csv`](summary/table_backbone_scale.csv) | ESM-2 8M, 35M, and 650M configurations |
| [`table_calibration.csv`](summary/table_calibration.csv) | Brier score and 10-bin expected calibration error |
| [`table_length_only_baseline.csv`](summary/table_length_only_baseline.csv) | Length-only logistic-regression control |
| [`table_length_matched.csv`](summary/table_length_matched.csv) | Evaluation on fixed exact length-matched test subsets |
| [`table_complementarity.csv`](summary/table_complementarity.csv) | Branch error overlap and recovery |
| [`table_imbalance.csv`](summary/table_imbalance.csv) | Evaluation on a low-AMP-prevalence subset |
| [`table_dataset_statistics.csv`](summary/table_dataset_statistics.csv) | Counts, lengths, and length-distribution KS tests |
| [`table_attrition.csv`](summary/table_attrition.csv) | In-house source-pool attrition |
| [`table_params_breakdown.csv`](summary/table_params_breakdown.csv) | Parameter counts by module |
| [`table_training_cost.csv`](summary/table_training_cost.csv) | Runtime, throughput, memory, and parameter summaries |

## Reading the tables

- Values with `±` are mean ± sample standard deviation over the contributing seeds, not confidence intervals. Counts appear as `n`, `Seeds`, `seeds`, or `n_seeds`, depending on the table. These columns give the number of seeds, not their identifiers.
- Main BiFAN and classical-model comparisons use six seeds (100–105); the deep baselines and smaller backbones use three (100–102). Some auxiliary configurations also use three; consult the row count.
- ACC is accuracy; MCC is Matthews correlation coefficient; AUC is ROC area; AUPRC is average precision as implemented in `metrics_utils.py`; Sn and Sp are sensitivity and specificity; BACC is balanced accuracy. Probability-based metrics and ratios are on a 0–1 scale unless the column explicitly states percent.
- `p_vs_reference` compares MCC with the reference configuration using a two-sided paired Wilcoxon signed-rank test over shared seeds. It is one P value for the chosen metric, not for every metric in the row. These tables do not apply a multiple-comparison correction.
- A scalar without `±` should be read according to the table: a deterministic or single evaluation, a count, a threshold, or a runtime summary. It does not by itself indicate the number of runs.
- `In-house (re-drawn)` refers to the fixed in-house dataset distributed here, not a new split drawn on each run. `deployed` in historical row labels means the selected BiFAN configuration, not a hosted prediction service.

## Complementarity definitions

In [`table_complementarity.csv`](summary/table_complementarity.csv), each model's binary decisions use its own validation-selected threshold. For each seed:

| Column | Definition |
|---|---|
| `seq_only_errors` | Number of sequences misclassified by ESM-2 alone |
| `hand_correct_share_pct` | Percentage of those ESM-2 errors classified correctly by the descriptor-only model |
| `fused_recovers_share_pct` | Percentage of those same ESM-2 errors classified correctly by the fused model |
| `hand_only_errors` | Number of descriptor-only errors |
| `seq_correct_share_pct` | Percentage of descriptor-only errors classified correctly by ESM-2 alone |
| `oracle_union_ACC` | Fraction correct under an oracle that chooses a correct branch decision whenever either branch is correct |
| `fused_ACC` | Accuracy of the actual fused model |

Both descriptor coverage and fused recovery use **all ESM-2-only errors as their denominator**. Fused recovery is not a percentage of the descriptor-covered subset. The oracle uses true labels to choose between decisions; it is a descriptive overlap measure, not a deployable prediction rule.

## Other conventions

- Classical models use `Length_norm + AAC20 + DPC400` (421 features), including normalized length. BiFAN uses `CTDC39 + CKSAAGP100 + AAC20 + DPC400` (559 features), without an explicit length input.
- Deep baselines use frozen ESM-2 embeddings. The historical `AMPlify` row is the CNN selected by CLI name `amplify`, not the original AMPlify attention-BiLSTM implementation. The transformer row is also an architecture-level baseline, not an official AMP-BERT reproduction.
- Macrel and amPEPpy use released weights without retraining. Their thresholds are selected on the local validation split. `n_unscored` reports unavailable scores; the saved threshold can have more decimal places than the rounded metrics.
- The length-only model uses a 90:10 fit/threshold subdivision of the training pool. The primary BiFAN models use the packaged validation split.
- `Prevalence` in the imbalance table is the AMP fraction in the evaluated subset. It is not an estimate of natural AMP prevalence.
- Zero or very small KS P values in dataset statistics are floating-point/formatting outputs, not a claim of an exact mathematical probability of zero.
- Runtime columns ending in `_median` are medians over contributing runs; `peak_gpu_gib_max` is a maximum in GiB. They are hardware-dependent. Parameter counts are counts, not performance estimates.

## Generating new comparisons

[`src/aggregate_results.py`](../src/aggregate_results.py) generates the two main comparison tables from new run JSON files. It does not regenerate all 18 archived tables. Run it with `--out-dir outputs/comparison` to keep the reference tables unchanged; commands are in the [main README](../README.md).

New training runs save `test_metrics.json`, `run_config.json`, and `predictions.npz`. Retain these outputs if you need to recompute seed statistics, calibration, or sequence-level error overlap. Rounded aggregate cells alone are insufficient to reconstruct the original paired tests or predictions.

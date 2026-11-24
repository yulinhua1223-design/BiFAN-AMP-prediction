# Attribution and third-party materials

## BiFAN code

Original BiFAN source code is distributed under the [MIT License](LICENSE). That license does not relicense third-party datasets, sequence records, pretrained weights, or external tools. Each upstream artifact retains its applicable terms.

## GenPept-Curated-2025 v1.1

Files in `data/genpept/` serialize the primary split as FASTA and provide labels and a derived length-matched test subset.

- [Upstream repository](https://github.com/biochem-data-sci/GenPept-Curated-2025).
- [Dataset DOI and creator list](https://doi.org/10.5281/zenodo.22994107).
- [Source-and-rights statement](https://github.com/biochem-data-sci/GenPept-Curated-2025/blob/main/docs/SOURCE_AND_RIGHTS.md).
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) applies to creator-owned curation, operational labels, split assignments, documentation, and compilation, to the extent the creators hold the relevant rights. The upstream statement does not claim ownership of third-party sequence records.

The FASTA conversion preserves sample IDs, sequences, labels, and primary split assignments. The BiFAN length-matched file is a derived test subset. Cite the dataset authors and version when reusing these records. Exact CSV checksums and conversion details are in [`data/README.md`](data/README.md).

## In-house data

The in-house corpus combines public AMP resources, including APD3, dbAMP 2.0, and DRAMP 4.0, with the non-AMP pool described in the manuscript. Final splits and an audit summary are distributed for reproducibility. Original source sequences remain subject to the source providers' terms; no new ownership of those sequences is asserted.

The release does not contain complete per-sequence source mappings. Contact [Jie Shen](mailto:shenjie@hdu.edu.cn) for source-level enquiries. Source-provider terms, rather than the repository's software license, govern third-party data reuse.

## ESM-2

The pretrained backbone is obtained separately from [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D). The repository does not distribute ESM-2 weights. Consult the upstream model card and license when obtaining or redistributing model artifacts.

## External prediction tools

[Macrel](https://github.com/BigDataBiology/macrel) and [amPEPpy](https://github.com/tlawrence3/amPEPpy) are not redistributed. The adapter uses local installations and released model artifacts supplied by the user. Preserve upstream citations and comply with their licenses.

The deep baselines in this repository are architecture-level implementations. Their historical labels do not imply that original AMPlify or AMP-BERT code or pretrained models are included.

## Dependencies

PyTorch, Transformers, NumPy, SciPy, scikit-learn, Joblib, Matplotlib, Hugging Face Hub, and optional LightGBM retain their respective licenses. Installing these dependencies does not transfer their rights into BiFAN's MIT License.

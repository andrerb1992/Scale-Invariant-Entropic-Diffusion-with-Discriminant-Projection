# SIEDP: Scale-Invariant Entropic Diffusion with Discriminant Projection

In the code, the final method is the variant named `SEI-NCA-S`. The ablations are `SEI-NCA` (fixed gamma), `SEI-S` (no projection), `SEI0-NCA` (eta = 0), `ISOMAP-NCA` and `PCA-NCA`.

## Installation
```bash
pip install -r requirements.txt
```
Python 3.10+ is recommended. `umap-learn` (supervised UMAP baseline) and `metric-learn` (LMNN, LFDA) are optional: those baselines are skipped when the package is missing.

## Data
The benchmark uses 34 of the 36 datasets of the Supervised K-ISOMAP article (Mendes and Levada, *Pattern Recognition* 183, 115087). They are downloaded automatically from [OpenML](https://www.openml.org) with `sklearn.datasets.fetch_openml` (names are listed in `benchmark36.py`). No manual download is needed, but the first run requires an internet connection. `CIFAR_10_small` and `GesturePhaseSegmentation` are not included.

## Reproducing the experiments
Protocol used in the article: stratified 50/50 train/test splits (seeds 42-46), at most 1000 samples per dataset, `d = 2`, eight classifiers, balanced accuracy and macro-F1.
```bash
python benchmark36.py --selftest                 # quick check with sklearn datasets (no internet)
python benchmark36.py --run --quick              # 3 datasets x 2 splits
python benchmark36.py --run --splits 5 --max-n 1000   # full benchmark (resumable)
python benchmark36.py --summary                  # tables, Friedman and Wilcoxon (Holm)
```
Runs are resumable: results are appended to `bench36_runs.csv` and finished combinations are skipped. Output goes to the script folder; set `SEI_OUT_DIR` to change it. `limpar_csv.py` removes duplicated or stale rows from the CSV files, and `diag_sei.py` runs diagnostics on SEI-ISOMAP.

**Note on sample sizes.** Rows with different `n_used` for the same dataset are inconsistent. The analysis keeps only the rows where `n_used` equals the dataset minimum, so rerun with `--max-n 1000` to get uniform samples.

## Figures and statistics
The scripts in `figures/` produce the figures and the statistical analysis of the article from the CSV files (Wilcoxon with Holm correction, Friedman and Nemenyi, bootstrap confidence intervals, Spearman correlations with dataset characteristics):
`analysis.py`, `figs2.py`, `build.py`, `finalize.py`, `sens.py` (sensitivity), `scale.py` (scale invariance) and `cases.py` (Iris, digits and vehicle embeddings).

## Main results (balanced accuracy, d = 2, 34 datasets)
| Method | Balanced accuracy |
|---|---|
| RAW features (reference) | 0.679 |
| Supervised UMAP | 0.607 |
| **SIEDP** | **0.599** |
| SIEDP, fixed gamma | 0.595 |
| PCA + same projection | 0.591 |
| NCA | 0.582 |
| Supervised K-ISOMAP | 0.566 |
| SIEDP without projection | 0.515 |
| SEI-ISOMAP | 0.512 |

SIEDP improves on the unprojected embedding it builds on (+0.084, 31 wins in 34 datasets, Holm-corrected p < 1e-5), but it is statistically indistinguishable from Supervised K-ISOMAP, NCA, supervised UMAP and PCA followed by the same projection. See the article for the complete analysis and the limitations.

## Repository content
| File | Purpose |
|---|---|
| `sei_isomap.py` | SEI-ISOMAP / SIEDP core, data loading and metrics |
| `benchmark36.py` | leakage-free benchmark on 34 datasets, summary and tests |
| `limpar_csv.py`, `diag_sei.py` | CSV cleaning and diagnostics |
| `figures/` | figure and analysis scripts |
| `bench36_runs.csv`, `bench36_summary.csv`, `bench36_stats.csv` | result files used in the article |
| `requirements.txt` | dependencies |

## Citation
If you use this code, please cite the article above and, for the base method, the SEI-ISOMAP article and the Zenodo record.

```bibtex
@misc{brito2026siedp,
  author = {de Brito, Andr{\'e} Ribeiro},
  title  = {Scale-Invariant Entropic Diffusion with Discriminant Projection for Supervised Dimensionality Reduction},
  year   = {2026},
  note   = {Code: REPOSITORY URL, DOI: ZENODO DOI}
}
```

## License
Add a license file (for example MIT) before publishing the repository.

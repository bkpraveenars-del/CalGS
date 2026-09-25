# CalGS — calibrated genomic selection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22962590.svg)](https://doi.org/10.5281/zenodo.22962590)

Code and simulated data for:

> Kumar P. *Calibrated genomic selection: distribution-free prediction intervals turn deep genomic prediction into a decision tool.* Manuscript submitted to *Genetics Selection Evolution*.

## Zenodo

Concept DOI: [10.5281/zenodo.22962590](https://doi.org/10.5281/zenodo.22962590)  
Published version: [10.5281/zenodo.22962591](https://doi.org/10.5281/zenodo.22962591)

The pipeline simulates a sesame-like (*Sesamum indicum* L.) diversity panel with known true breeding values. It then compares GBLUP with three deep architectures (a multilayer perceptron, a one-dimensional CNN and a patch-based genomic transformer) and asks three questions: whether each model's own uncertainty statement is calibrated, whether split-conformal calibration repairs it, and whether acting on calibrated uncertainty changes selection decisions. **All data are simulated.** No accession, marker or allele corresponds to a real one.

## Reproduce everything

```bash
python -m pip install -r requirements.txt     # Python 3.11
bash run_all.sh                               # about 3 h on 2 CPU cores
```

| Step | Script | Output |
|---|---|---|
| 1 | `simulate_genome.py` | `data/panel.npz`, `results/panel_summary.json` |
| 2 | `run_experiment.py` | `results/tuning.csv`, `results/best_hp.json`, `results/cv_main.csv`, `results/attention.npz` |
| 3 | `run_supp.py` | `results/supp_trainsize.csv`, `results/supp_loeo.csv` |
| 4 | `analyse.py` | `results/tables/*.csv`, `results/numbers.json` |
| 5 | `fig.py` | `figures/Figure1–6.png` (600 dpi) and `.pdf` |
| 6 | `make_additional_files.py` | `submission/Additional_file_1–7.xlsx` (journal supplementary files) |

All randomness derives from `MASTER_SEED = 20260902` in `config.py`. Every number in the manuscript is written by `analyse.py` to `results/numbers.json` or to a table in `results/tables/`; none is typed by hand.

## Files

| File | Content |
|---|---|
| `config.py` | Every design constant: genome, founders, traits, environments, CV design, seed |
| `simulate_genome.py` | Genome, founder-mosaic accessions, quality control, traits (pleiotropy, epistasis, G×E, observation-scale links) |
| `models_stat.py` | VanRaden G, GBLUP with REML (spectral likelihood), PEV, split-conformal and Gaussian interval constructions |
| `models_dl.py` | MLP, 1D-CNN, genomic transformer; training with early stopping on validation correlation; MC dropout; class-token attention |
| `run_experiment.py` | Hyperparameter tuning on two tuning traits; repeated stratified 5-fold CV |
| `run_supp.py` | Training-population size and leave-one-environment-out experiments |
| `analyse.py` | Tables 1–6, S2–S5 and all quoted numbers |
| `fig.py` | Figures 1–6 |
| `make_additional_files.py` | Additional files 1–7 for the journal (README sheet, fold-level data, formula-linked summaries) |

### `data/panel.npz`

| Key | Shape | Meaning |
|---|---|---|
| `X` | 430 × m | QC-passed genotypes, 0/2 (fully homozygous) |
| `chrom`, `pos` | m | chromosome and physical position (bp) of the QC markers |
| `group` | 430 | subpopulation (0, 1, 2) |
| `Y` | 430 × 5 | line-mean phenotypes over six environments (observed scale) |
| `TBV` | 430 × 5 | true additive breeding values (latent scale) |
| `<TRAIT>_obs` | 430 × 6 | observed phenotype in each environment |
| `<TRAIT>_latent`, `<TRAIT>_G` | | latent phenotype; total genotypic value |
| `<TRAIT>_qtl`, `<TRAIT>_pairs`, `<TRAIT>_gxe_loci` | | indices (into `chrom_all`/`pos_all`) of additive QTL, epistatic pairs, G×E loci |
| `chrom_all`, `pos_all`, `keep` | | all simulated markers and the QC mask |

Traits: `SYPP` seed yield per plant, `OIL` total oil content, `PH` plant height, `SCC` seed coat colour (L\*), `DTI` drought tolerance index.
Environments: well-watered, moderate and severe terminal drought, each in two years.

## Key design choices (see Methods)

- **Conformal calibration set.** 20% of each training partition is held out and is **never** used for fitting. The models are not refitted on training plus calibration data, because refitting would void the finite-sample coverage guarantee.
- **Deep-model input.** 4,096 markers thinned uniformly along the genome, independent of the phenotype.
- **Early stopping.** Uses validation Pearson correlation, not MSE.
- **Hyperparameters.** Chosen once per architecture by internal-validation correlation on seed coat colour and the drought index (the highest and lowest target heritability), then fixed for all traits. Test folds are never used for tuning.

## Licence and citation

MIT licence. Please cite the article and the archived release (see `CITATION.cff`).

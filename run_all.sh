#!/usr/bin/env bash
# Regenerates every number, table and figure in the manuscript (about 3 h on 2 CPU cores).
set -euo pipefail
python simulate_genome.py        # data/panel.npz
python run_experiment.py         # tuning + main cross-validation
python run_supp.py               # training-size and leave-one-environment-out experiments
python analyse.py                # results/tables/*.csv, results/numbers.json
python fig.py                    # figures/Figure1-6 (.png, .pdf)
python make_additional_files.py   # submission/Additional_file_1-7.xlsx

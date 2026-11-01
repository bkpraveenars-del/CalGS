"""Global configuration for the CalGS pipeline.

Every stochastic step derives its random stream from MASTER_SEED, so the
whole study (panel, phenotypes, folds, networks) is regenerated exactly.
"""
from pathlib import Path
import os

MASTER_SEED = int(os.environ.get("CALGS_SEED", 20260902))

ROOT = Path(__file__).resolve().parent
_TAG = os.environ.get("CALGS_TAG", "")
DATA = ROOT / ("data" + _TAG)
RESULTS = ROOT / ("results" + _TAG)
FIGURES = ROOT / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

# ---------------- genome ----------------
CHR_LEN_MB = [25.1, 23.0, 22.4, 21.2, 20.6, 20.0, 18.3, 17.5, 16.9, 16.2, 15.3, 14.0, 13.6]  # 244.1 Mb
MARKERS_PER_MB = 110
FST = 0.06
GROUP_SIZES = [int(round(g * float(os.environ.get("CALGS_SCALE", 1)))) for g in (178, 145, 107)]          # 430 accessions
N_FOUNDERS = 46                        # founder haplotypes per chromosome
FOUNDERS_PER_GROUP = [19, 16, 11]
LD_HALF_LIFE_KB = 340.0                # latent AR(1) correlation halves every 340 kb
SEGMENT_MEAN_MB = 7.0                  # mean ancestry-segment length in the mosaic
W_OWN, W_OTHER = 1.0, 0.28             # founder sampling weights (own vs other subpopulation)
MISSING_RATE = 0.03
MIN_CALL_RATE = 0.90
MIN_MAF = 0.05

# ---------------- traits ----------------
TRAITS = ["SYPP", "OIL", "PH", "SCC", "DTI"]
TRAIT_NAMES = {
    "SYPP": "Seed yield per plant (g)",
    "OIL": "Total oil content (%)",
    "PH": "Plant height (cm)",
    "SCC": "Seed coat colour (L*)",
    "DTI": "Drought tolerance index",
}
TRAIT_SPEC = {
    #        n additive QTL, effect dist, epistatic pairs, epistatic share, GxE ratio, target line-mean h2, link
    "SYPP": dict(n_qtl=420, dist="gamma", n_epi=90, epi=0.22, gxe=0.55, h2=0.45, link="lognormal"),
    "OIL": dict(n_qtl=140, dist="normal", n_epi=18, epi=0.10, gxe=0.14, h2=0.82, link="identity"),
    "PH": dict(n_qtl=180, dist="normal", n_epi=25, epi=0.10, gxe=0.28, h2=0.68, link="identity"),
    "SCC": dict(n_qtl=22, dist="oligo", n_epi=4, epi=0.10, gxe=0.08, h2=0.86, link="identity"),
    "DTI": dict(n_qtl=520, dist="gamma", n_epi=120, epi=0.22, gxe=0.72, h2=0.38, link="logistic"),
}
GAMMA_SHAPE = 0.4
N_SHARED_FACTORS = 3
LOCI_PER_FACTOR = 260
# Target genetic correlations (order = TRAITS). Only the four relationships reported
# repeatedly in sesame are specified; all other targets are zero.
TARGET_RG = {("SYPP", "OIL"): -0.25, ("SYPP", "PH"): 0.35, ("OIL", "SCC"): 0.50, ("SYPP", "DTI"): 0.30}
STRESS_TRAITS = ["SYPP", "PH", "DTI"]
# Environments: 3 water regimes x 2 years
ENVS = [("WW", 1), ("WW", 2), ("MOD", 1), ("MOD", 2), ("SEV", 1), ("SEV", 2)]
DROUGHT_SHIFT = {"WW": 0.0, "MOD": -0.5, "SEV": -1.0}    # latent SD units, stress traits only
YEAR_SHIFT = {1: 0.1, 2: -0.1}

# ---------------- evaluation ----------------
N_FOLDS = 5
N_REPEATS = 2
CAL_FRAC = 0.20          # of each training partition, held out for conformal calibration
VAL_FRAC = 0.20          # of the remainder, internal validation for early stopping
LEVELS = [0.80, 0.90, 0.95]
SELECTION_FRACTION = 0.10
N_THIN = 4096            # markers given to the deep models
MC_PASSES = 25
ENSEMBLE_SIZE = 3
TUNING_TRAITS = ["SCC", "DTI"]   # highest and lowest target heritability

"""Build the journal Additional files 1-7 (.xlsx) from the raw outputs.

Each workbook has a README sheet (title, description, column dictionary, provenance),
the raw per-fold data, and, where the manuscript quotes aggregates, a Summary sheet
whose cells are AVERAGEIFS formulas over the raw sheet so every summary value can be
traced to the fold-level data.
Usage:  python make_additional_files.py      -> submission/Additional_file_{1..7}.xlsx
"""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
import config as C

OUT = C.ROOT / "submission"
OUT.mkdir(exist_ok=True)
R, T = C.RESULTS, C.RESULTS / "tables"
F = Font(name="Arial", size=10)
FB = Font(name="Arial", size=10, bold=True)
FT = Font(name="Arial", size=12, bold=True)
HEAD = PatternFill("solid", fgColor="DDEBF7")
CITATION = ("Kumar P. Calibrated genomic selection: distribution-free prediction intervals turn deep "
            "genomic prediction into a decision tool. Genetics Selection Evolution (submitted).")


def readme(wb, title, description, columns, provenance):
    ws = wb.active
    ws.title = "README"
    rows = [(title, FT), ("", F), (CITATION, F), ("", F), ("Description", FB), (description, F), ("", F),
            ("Provenance", FB), (provenance, F), ("", F), ("Column dictionary", FB)]
    for i, (v, f) in enumerate(rows, 1):
        c = ws.cell(i, 1, v); c.font = f; c.alignment = Alignment(wrap_text=True, vertical="top")
    r = len(rows) + 1
    ws.cell(r, 1, "Sheet / column").font = FB; ws.cell(r, 2, "Meaning").font = FB
    for k, v in columns:
        r += 1
        ws.cell(r, 1, k).font = F
        c = ws.cell(r, 2, v); c.font = F; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 95
    for i in range(1, len(rows) + 1):
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=2)
    return ws


def data_sheet(wb, name, df, decimals=4):
    ws = wb.create_sheet(name)
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(1, j, str(col)); c.font = FB; c.fill = HEAD
        ws.column_dimensions[get_column_letter(j)].width = max(10, min(28, len(str(col)) + 2))
    for i, row in enumerate(df.itertuples(index=False), 2):
        for j, v in enumerate(row, 1):
            if isinstance(v, (float, np.floating)):
                v = None if np.isnan(v) else round(float(v), decimals)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            ws.cell(i, j, v).font = F
    ws.freeze_panes = "A2"
    return ws


def summary_sheet(wb, name, raw_name, raw_df, keys, values, note):
    """Summary of AVERAGEIFS formulas over raw_name, grouped by the columns in `keys`."""
    ws = wb.create_sheet(name)
    col_of = {c: get_column_letter(j + 1) for j, c in enumerate(raw_df.columns)}
    n = len(raw_df) + 1
    groups = raw_df[keys].drop_duplicates().values.tolist()
    hdr = keys + [f"mean {v}" for v in values] + ["n folds"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(1, j, h); c.font = FB; c.fill = HEAD
        ws.column_dimensions[get_column_letter(j)].width = max(12, len(h) + 2)
    for i, g in enumerate(groups, 2):
        crit = []
        for j, (k, gv) in enumerate(zip(keys, g), 1):
            ws.cell(i, j, gv).font = F
            crit.append(f"'{raw_name}'!${col_of[k]}$2:${col_of[k]}${n},${get_column_letter(j)}{i}")
        for j, v in enumerate(values, len(keys) + 1):
            f = f"=AVERAGEIFS('{raw_name}'!${col_of[v]}$2:${col_of[v]}${n},{','.join(crit)})"
            c = ws.cell(i, j, f); c.font = F; c.number_format = "0.000"
        c = ws.cell(i, len(keys) + len(values) + 1, f"=COUNTIFS({','.join(crit)})"); c.font = F
    r = len(groups) + 3
    c = ws.cell(r, 1, note); c.font = Font(name="Arial", size=9, italic=True)
    ws.freeze_panes = "A2"


CV_COLS = [
    ("trait", "SYPP seed yield per plant; OIL total oil content; PH plant height; SCC seed coat colour (L*); DTI drought tolerance index"),
    ("rep, fold", "Cross-validation repeat (0-1) and fold (0-4); stratified by subpopulation"),
    ("model, source", "GBLUP-PEV; MLP / CNN (1D-CNN) / TRF (transformer) with MC (Monte-Carlo dropout) or ENS (deep ensemble)"),
    ("h2_reml", "GBLUP only: REML heritability 1/(1+lambda) on the fitting set"),
    ("best_epoch", "Deep models: epoch with best validation correlation (mean over ensemble members for ENS)"),
    ("pa", "Predictive ability: Pearson r(prediction, observed line-mean phenotype) in the test fold"),
    ("ta", "True accuracy: Pearson r(prediction, true breeding value) in the test fold"),
    ("cov_<c>_<L>", "Empirical coverage of the nominal L% interval of construction c (native, plugin, epi_ale, conformal, adaptive)"),
    ("width_<c>_<L>", "Mean interval width in panel phenotypic standard deviations"),
    ("qa90", "Normalised conformal quantile at 90% (multiplier of the native SD in the adaptive interval)"),
    ("gain_point, gain_lb", "Realised genetic gain (TBV SD units) selecting the top 10% on the point prediction / lower adaptive-conformal bound"),
    ("coin_point, coin_lb", "Coincidence index: fraction of the true top 10% recovered"),
    ("oracle", "Gain from selecting on the true breeding values"),
    ("n_test", "Number of accessions in the test fold"),
]
PROV = ("Generated by the CalGS pipeline (seed 20260902): run_experiment.py / run_supp.py write the raw data; "
        "analyse.py writes the tables quoted in the article; this workbook is built by make_additional_files.py. "
        "All data are simulated. Code and data: https://github.com/bkpraveenars-del/CalGS")


def main():
    cv = pd.read_csv(R / "cv_main.csv")
    lead = ["trait", "rep", "fold", "model", "source"]
    cv = cv[lead + [c for c in cv.columns if c not in lead]]

    # 1 --------------------------------------------------------------
    wb = Workbook()
    readme(wb, "Additional file 1: Table S1. Full cross-validation output",
           "One row per trait, repeat, fold, model and uncertainty source (5 traits x 10 folds x 7 model-source "
           "combinations = 350 rows): accuracy, coverage and width of every interval construction at nominal 80%, "
           "90% and 95%, and selection outcomes. Tables 3-6 and Figures 3-5 of the article are aggregates of this sheet.",
           [("Sheet 'Table S1'", "Raw fold-level results")] + CV_COLS, PROV)
    data_sheet(wb, "Table S1", cv)
    wb.save(OUT / "Additional_file_1.xlsx")

    # 2 --------------------------------------------------------------
    wb = Workbook()
    s2 = pd.read_csv(T / "table_S2_h2.csv")
    readme(wb, "Additional file 2: Table S2. Realised heritabilities",
           "Target and realised narrow-sense heritability on a line-mean basis over six environments, and realised "
           "single-environment heritability (squared correlation between true breeding value and the single-plot "
           "phenotype) in each environment.",
           [("trait", CV_COLS[0][1]), ("target_linemean", "Target line-mean heritability used to set the residual variance"),
            ("realised_linemean", "Squared correlation between TBV and the observed six-environment line mean"),
            ("WW/MOD/SEV-Y1/Y2", "Single-environment realised heritability: well-watered, moderate or severe terminal drought, year 1 or 2")],
           PROV)
    data_sheet(wb, "Table S2", s2, 3)
    wb.save(OUT / "Additional_file_2.xlsx")

    # 3 --------------------------------------------------------------
    wb = Workbook()
    s3 = pd.read_csv(R / "supp_trainsize.csv")
    readme(wb, "Additional file 3: Table S3. Training-population size experiment",
           "GBLUP and the MLP with Monte-Carlo dropout fitted with the proper training set subsampled to 80, 120, 160 "
           "and 219 lines (validation and calibration sets fixed; first repeat, five folds, five traits).",
           [("Sheet 'Raw'", "One row per trait, fold, training size and model"),
            ("n_train / n_fit", "Proper training lines / lines used for fitting (proper training + validation)"),
            ("pa, ta", "Predictive ability and true accuracy"),
            ("cov_native_90, cov_conformal_90", "Coverage of the nominal 90% native and split-conformal intervals"),
            ("width_native_90, width_conformal_90", "Mean width in phenotypic SD"),
            ("Sheet 'Summary'", "AVERAGEIFS over the raw sheet by model and training size (values quoted in the Results)")],
           PROV)
    data_sheet(wb, "Raw", s3)
    summary_sheet(wb, "Summary", "Raw", s3, ["model", "n_train"],
                  ["pa", "ta", "cov_native_90", "cov_conformal_90", "width_native_90", "width_conformal_90"],
                  "Means over 5 traits x 5 folds; formulas reference sheet 'Raw'.")
    wb.save(OUT / "Additional_file_3.xlsx")

    # 4 --------------------------------------------------------------
    wb = Workbook()
    s4 = pd.read_csv(R / "supp_loeo.csv")
    readme(wb, "Additional file 4: Table S4. Leave-one-environment-out experiment",
           "GBLUP fitted and calibrated on the mean of five environments and tested on single-plot phenotypes in the "
           "sixth. Regimes: unseen (calibration on five-environment means); centred (as unseen, predictions shifted by "
           "the target environment main effect); recal (calibration lines phenotyped in the target environment).",
           [("Sheet 'Raw'", "One row per trait, fold, held-out environment and regime"),
            ("env", "Held-out environment: WW/MOD/SEV x year"),
            ("cov_conformal_90", "Coverage of the nominal 90% split-conformal interval for target-environment phenotypes"),
            ("width_conformal_90", "Mean width in SD units of the target-environment phenotype"),
            ("Sheet 'Summary'", "AVERAGEIFS by regime and trait")], PROV)
    data_sheet(wb, "Raw", s4)
    summary_sheet(wb, "Summary", "Raw", s4, ["regime", "trait"], ["cov_conformal_90", "width_conformal_90"],
                  "Means over 5 folds x 6 held-out environments; formulas reference sheet 'Raw'.")
    wb.save(OUT / "Additional_file_4.xlsx")

    # 5 --------------------------------------------------------------
    wb = Workbook()
    s5 = pd.read_csv(T / "table_S5_attention.csv")
    readme(wb, "Additional file 5: Table S5. Transformer attention versus true QTL",
           "Spearman correlation between class-token attention (last block, heads averaged, averaged over test "
           "accessions and folds) and the number of the trait's additive QTL within each of the 64 genomic patches; "
           "one-sided Mann-Whitney test of QTL enrichment in the top attention quintile. The per-patch profiles are "
           "given in sheet 'Profiles'.",
           [("spearman_rho, p", "Spearman correlation and two-sided P"),
            ("qtl_top_quintile, qtl_other", "Mean true QTL per patch in the top attention quintile / remaining patches"),
            ("mannwhitney_p_greater", "One-sided Mann-Whitney P (top quintile > others)"),
            ("Sheet 'Profiles'", "patch index (0-63), mean attention and true QTL count per patch, for each trait")], PROV)
    data_sheet(wb, "Table S5", s5)
    prof = []
    for t in C.TRAITS:
        A, dens = np.load(R / f"attention_profile_{t}.npy")
        prof.append(pd.DataFrame(dict(trait=t, patch=np.arange(len(A)), attention=A, true_qtl=dens.astype(int))))
    data_sheet(wb, "Profiles", pd.concat(prof), 6)
    wb.save(OUT / "Additional_file_5.xlsx")

    # 6 --------------------------------------------------------------
    wb = Workbook()
    s6 = pd.read_csv(R / "supp_stopping.csv")
    readme(wb, "Additional file 6: Table S6. Early stopping on validation correlation versus validation MSE",
           "MLP and transformer trained with identical seeds, stopping on validation Pearson correlation (r, the "
           "criterion used throughout) or on validation mean squared error (mse); first repeat, five folds, five traits.",
           [("Sheet 'Raw'", "One row per trait, fold, architecture and criterion"),
            ("best_epoch", "Epoch at which the restored weights were obtained"),
            ("pa", "Test-fold predictive ability"),
            ("Sheet 'Summary'", "AVERAGEIFS by architecture and criterion")], PROV)
    data_sheet(wb, "Raw", s6)
    summary_sheet(wb, "Summary", "Raw", s6, ["arch", "criterion"], ["best_epoch", "pa"],
                  "Means over 5 traits x 5 folds; paired t-tests are reported in the article.")
    wb.save(OUT / "Additional_file_6.xlsx")

    # 7 --------------------------------------------------------------
    wb = Workbook()
    s7 = pd.read_csv(R / "tuning.csv")
    readme(wb, "Additional file 7: Table S7. Hyperparameter search",
           "Four configurations per architecture evaluated by internal-validation correlation on the two tuning traits "
           "(SCC, DTI; first repeat, five folds). The configuration with the highest mean was selected and held fixed "
           "for all traits. Test folds were not used.",
           [("Sheet 'Raw'", "One row per architecture, grid point, tuning trait and fold"),
            ("arch, grid", "Architecture and grid index (0-3)"),
            ("val_r, best_epoch", "Best internal-validation correlation and its epoch"),
            ("hidden / channels / d / depth / dropout / lr / wd", "Hyperparameters (empty where not applicable)"),
            ("Sheet 'Summary'", "AVERAGEIFS by architecture and grid point")], PROV)
    data_sheet(wb, "Raw", s7)
    summary_sheet(wb, "Summary", "Raw", s7, ["arch", "grid"], ["val_r", "best_epoch"],
                  "Mean over 2 tuning traits x 5 folds; selected configurations are in results/best_hp.json.")
    wb.save(OUT / "Additional_file_7.xlsx")
    print("written:", sorted(p.name for p in OUT.glob("Additional_file_*.xlsx")))


if __name__ == "__main__":
    main()

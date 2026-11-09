"""Supplementary Tables S12 (revised) and S13 for the second revision, from the outputs of analyse_r2.py.
Usage: python make_supp_r2.py  (writes results/tables/Supplementary_Table_S12.xlsx and _S13.xlsx)"""
import pandas as pd
import config as C

T = C.ROOT / "results" / "tables"
DEF = [
    ("rule", "conformal: Benjamini-Hochberg (BH) on conformal p-values (Eq. 15); native / plugin: BH on Gaussian "
             "p-values from the native or plug-in SD; native_meannull / plugin_meannull: largest set, in order of increasing "
             "Gaussian null probability, whose mean null probability is <= q (mean-probability rule); top10: top 10% on the "
             "point prediction"),
    ("threshold", "q50, q75, q90: median, 75th and 90th percentile of the standardised phenotype in the fitting set"),
    ("q", "target FDR 0.20"),
    ("frac_sel", "share of the test fold advanced"),
    ("any_sel", "1 if at least one line was advanced (summaries: share of trait x fold units)"),
    ("fdp / fdr", "false discovery proportion against the phenotype (0 when nothing advanced); its mean is the FDR"),
    ("fdp_cond", "FDP over the units in which at least one line was advanced (FDP if any)"),
    ("fdp_tbvq / fdr_tbvq", "FDP / FDR against the TBV: false discovery if the TBV does not exceed the same percentile of "
                            "the TBV in the fitting set (reported in the article as FDR (TBV))"),
    ("fdp_tbv / fdr_tbv", "alternative TBV definition: TBV projected to the phenotype scale (Eq. 14) does not exceed c"),
    ("power", "share of test lines with phenotype above the threshold that were advanced"),
    ("summaries", "five-panel summaries are means over panel means; n_sel_nonempty, fdp_cond are means over units "
                  "with any advancement"),
]

with pd.ExcelWriter(T / "Supplementary_Table_S12.xlsx") as w:
    pd.DataFrame(DEF, columns=["column", "definition"]).to_excel(w, sheet_name="README", index=False)
    pd.read_csv(T / "rev2_fdr_summary.csv").to_excel(w, sheet_name="five_panels_summary", index=False)
    pd.read_csv(T / "rev2_fdr_by_trait.csv").to_excel(w, sheet_name="five_panels_by_trait_q50", index=False)
    pd.read_csv(T / "rev2_fdr_foldlevel.csv").to_excel(w, sheet_name="five_panels_foldlevel", index=False)
    pd.read_csv(T / "rev2_fdr_big.csv").to_excel(w, sheet_name="enlarged_GBLUP_summary_q50", index=False)
    pd.read_csv(T / "rev2_fdr_big_foldlevel.csv").to_excel(w, sheet_name="enlarged_GBLUP_foldlevel", index=False)

rv = pd.read_csv(T / "rev_S8_replicates_foldlevel.csv")
rv["entry"] = rv.model + "-" + rv.source
cols = [c for c in rv.columns if c.startswith(("cov_", "covg_", "width_")) and c.endswith(("_80", "_90"))]
bt = rv.groupby(["entry", "trait", "panel"])[cols].mean().groupby(["entry", "trait"]).mean().reset_index()
pm = rv.groupby(["entry", "trait", "panel"])[cols].mean().reset_index()
DEF13 = [("cov_<construction>_<level>", "coverage of the observed line mean by the nominal 80% or 90% interval"),
         ("covg_<construction>_<level>", "coverage of the TBV expressed on the standardised phenotype scale (Eq. 14)"),
         ("width_<construction>_<level>", "mean interval width in phenotypic SD"),
         ("constructions", "native, plugin, epi_ale (epistemic + aleatoric), conformal (split), adaptive (conformal), "
                           "pev_only (GBLUP, PEV without residual variance)"),
         ("by_trait", "means over the five panel means (5 test folds per panel)"),
         ("by_trait_panel", "panel-level means")]
with pd.ExcelWriter(T / "Supplementary_Table_S13.xlsx") as w:
    pd.DataFrame(DEF13, columns=["column", "definition"]).to_excel(w, sheet_name="README", index=False)
    bt.to_excel(w, sheet_name="by_trait", index=False)
    pm.to_excel(w, sheet_name="by_trait_panel", index=False)
print("written")

"""Summaries for the revision. Reads the revision outputs of all panels and writes
results/rev_numbers.json and results/tables/rev_*.csv. Every number quoted in the revised
manuscript for the new experiments comes from here.  Usage: python analyse_rev.py"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import config as C

R = C.ROOT
TAGS = ["", "_r1", "_r2", "_r3", "_r4"]          # panel 1 .. panel 5
OUT = R / "results" / "tables"
OUT.mkdir(exist_ok=True)
N = {}


def jf(x):
    return float(np.round(x, 6))


def panel_t(a, b=None):
    """Paired (or one-sample) t-test over panel-level means."""
    d = np.asarray(a) - (0 if b is None else np.asarray(b))
    t = stats.ttest_1samp(d, 0.0)
    return dict(diff=jf(d.mean()), se=jf(d.std(ddof=1) / np.sqrt(len(d))), t=jf(t.statistic), p=jf(t.pvalue), n=len(d))


def corrected_t(d, n_te_over_n_tr):
    """Nadeau and Bengio (2003) corrected resampled t-test."""
    d = np.asarray(d, float)
    J = len(d)
    v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te_over_n_tr) * v) if v > 0 else 0.0
    return dict(diff=jf(d.mean()), t=jf(t), p=jf(2 * stats.t.sf(abs(t), J - 1)), J=J)


# --------------------------------------------------------------- panel 1, original design
cm = pd.read_csv(R / "results" / "cv_main.csv")
RATIO = 86 / 344
pa_tests = {}
for t in C.TRAITS:
    g = cm[(cm.trait == t) & (cm.model == "GBLUP")].sort_values(["rep", "fold"])
    m = cm[(cm.trait == t) & (cm.model == "MLP") & (cm.source == "ENS")].sort_values(["rep", "fold"])
    pa_tests[t] = corrected_t(m.pa.values - g.pa.values, RATIO)
N["p1_pa_corrected"] = pa_tests
lb_tests = {}
for mod, src in (("GBLUP", "PEV"), ("MLP", "ENS"), ("CNN", "ENS"), ("TRF", "ENS")):
    x = cm[(cm.model == mod) & (cm.source == src)]
    per_trait = {t: corrected_t((x[x.trait == t].gain_lb - x[x.trait == t].gain_point).values, RATIO)
                 for t in C.TRAITS}
    lb_tests[mod] = per_trait
N["p1_lb_corrected"] = lb_tests
u = cm.groupby(["trait", "rep", "fold"])[["cov_conformal_80", "cov_conformal_90"]].mean()
N["p1_conf_cov"] = [jf(cm[c].mean()) for c in u.columns]
N["p1_conf_cov_se_rows350"] = [jf(cm[c].std(ddof=1) / np.sqrt(len(cm))) for c in u.columns]
N["p1_conf_cov_se_units50"] = [jf(u[c].std(ddof=1) / np.sqrt(len(u))) for c in u.columns]

# --------------------------------------------------------------- replicate panels
frames = []
for i, tag in enumerate(TAGS):
    fs = sorted((R / f"results{tag}").glob("rev_cv*.csv"))
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["trait", "fold", "model", "source"])
        d["panel"] = i + 1
        frames.append(d)
rv = pd.concat(frames, ignore_index=True)
rv["entry"] = rv.model + "-" + rv.source
N["n_panels"] = int(rv.panel.nunique())
N["panels"] = sorted(int(p) for p in rv.panel.unique())
rv.to_csv(OUT / "rev_S8_replicates_foldlevel.csv", index=False)
PM = rv.groupby(["panel", "entry"]).mean(numeric_only=True).reset_index()   # panel-level means

ENTRIES = ["GBLUP-PEV", "GBLUP-thin-PEV", "MLP-MC", "MLP-ENS", "CNN-MC", "CNN-ENS", "TRF-MC", "TRF-ENS"]


def summ(col, entry):
    v = PM[PM.entry == entry][col].values
    return dict(mean=jf(v.mean()), se=jf(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                min=jf(v.min()), max=jf(v.max()))


# accuracy (point entries: GBLUP, GBLUP-thin, deep ENS)
acc = []
for e in ["GBLUP-PEV", "GBLUP-thin-PEV", "MLP-ENS", "CNN-ENS", "TRF-ENS"]:
    for t in C.TRAITS:
        x = rv[(rv.entry == e) & (rv.trait == t)].groupby("panel")[["pa", "ta"]].mean()
        acc.append(dict(entry=e, trait=t, pa=x.pa.mean(), pa_se=x.pa.std(ddof=1) / np.sqrt(len(x)),
                        ta=x.ta.mean(), ta_se=x.ta.std(ddof=1) / np.sqrt(len(x))))
acc = pd.DataFrame(acc)
acc.to_csv(OUT / "rev_accuracy_by_trait.csv", index=False)
N["rep_pa"] = {e: summ("pa", e) for e in ENTRIES}
N["rep_ta"] = {e: summ("ta", e) for e in ENTRIES}
gp = PM[PM.entry == "GBLUP-PEV"].set_index("panel")
N["rep_pa_test"] = {e: panel_t(PM[PM.entry == e].set_index("panel").pa, gp.pa)
                    for e in ["GBLUP-thin-PEV", "MLP-ENS", "CNN-ENS", "TRF-ENS"]}
# per trait: was any deep model better than GBLUP in any panel x trait?
pt = rv[rv.source.isin(["PEV", "ENS"])].groupby(["panel", "trait", "model"]).pa.mean().unstack("model")
N["rep_cells"] = int(len(pt))
N["rep_cells_deep_beats_gblup"] = int((pt[["MLP", "CNN", "TRF"]].max(axis=1) > pt["GBLUP"]).sum())
N["rep_cells_mlp_beats_gblup"] = int((pt["MLP"] > pt["GBLUP"]).sum())
N["rep_cells_trf_beats_mlp"] = int((pt["TRF"] > pt["MLP"]).sum())

# coverage of phenotype and TBV
cov = []
for e in ENTRIES:
    row = dict(entry=e)
    for c in ["native", "plugin", "epi_ale", "conformal", "adaptive", "pev_only"]:
        for lev in (80, 90):
            col = f"cov_{c}_{lev}"
            if col in PM and PM[PM.entry == e][col].notna().any():
                s = summ(col, e); row[f"y_{c}_{lev}"] = s["mean"]; row[f"y_{c}_{lev}_se"] = s["se"]
                row[f"y_{c}_{lev}_min"] = s["min"]; row[f"y_{c}_{lev}_max"] = s["max"]
                s = summ(f"covg_{c}_{lev}", e); row[f"g_{c}_{lev}"] = s["mean"]; row[f"g_{c}_{lev}_se"] = s["se"]
                row[f"w_{c}_{lev}"] = summ(f"width_{c}_{lev}", e)["mean"]
    cov.append(row)
cov = pd.DataFrame(cov)
cov.to_csv(OUT / "rev_coverage.csv", index=False)
N["rep_cov"] = {r["entry"]: {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "entry"}
                for r in cov.to_dict("records")}
deep = [e for e in ENTRIES if not e.startswith("GBLUP")]
N["rep_native_y90_deep_range"] = [jf(cov.set_index("entry").loc[deep, "y_native_90"].min()),
                                  jf(cov.set_index("entry").loc[deep, "y_native_90"].max())]
N["rep_native_g90_deep_range"] = [jf(cov.set_index("entry").loc[deep, "g_native_90"].min()),
                                  jf(cov.set_index("entry").loc[deep, "g_native_90"].max())]
pc = PM.groupby("panel")[["cov_conformal_80", "cov_conformal_90"]].mean()
N["rep_conf_pooled"] = {c: dict(mean=jf(pc[c].mean()), se=jf(pc[c].std(ddof=1) / np.sqrt(len(pc))),
                                min=jf(pc[c].min()), max=jf(pc[c].max())) for c in pc.columns}
pp = PM.groupby("panel")[["cov_plugin_80", "cov_plugin_90"]].mean()
N["rep_plugin_pooled"] = {c: dict(mean=jf(pp[c].mean()), se=jf(pp[c].std(ddof=1) / np.sqrt(len(pp)))) for c in pp.columns}
cellmin = rv.groupby(["panel", "trait", "entry"]).cov_conformal_90.mean()
N["rep_conf90_cell_range"] = [jf(cellmin.min()), jf(cellmin.max())]
N["rep_conf90_cells_below_085"] = int((cellmin < 0.85).sum())
N["rep_conf90_cells"] = int(len(cellmin))

# selection
sel = []
for e in ["GBLUP-PEV", "GBLUP-thin-PEV", "MLP-ENS", "CNN-ENS", "TRF-ENS"]:
    x = PM[PM.entry == e].set_index("panel")
    sel.append(dict(entry=e, gain_point=x.gain_point.mean(), gain_lb=x.gain_lb.mean(), oracle=x.oracle.mean(),
                    **{f"lb_{k}": v for k, v in panel_t(x.gain_lb, x.gain_point).items()}))
sel = pd.DataFrame(sel)
sel.to_csv(OUT / "rev_selection.csv", index=False)
N["rep_sel"] = sel.set_index("entry").to_dict("index")

# advancement with FDR control (decision.py)
import decision as DEC
dec = DEC.run(TAGS)
dec.to_csv(OUT / "rev_S12_fdr_selection_foldlevel.csv", index=False)
dpm = dec.groupby(["panel", "entry", "threshold", "rule"])[["frac_sel", "fdp", "power", "gain"]].mean().reset_index()
ds = dpm.groupby(["entry", "threshold", "rule"]).agg(frac_sel=("frac_sel", "mean"), fdr=("fdp", "mean"),
                                                     fdr_se=("fdp", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                                     power=("power", "mean"), gain=("gain", "mean")).reset_index()
ds.to_csv(OUT / "rev_fdr_selection.csv", index=False)
N["fdr"] = {f"{r['entry']}|{r['threshold']}|{r['rule']}": {k: (None if pd.isna(v) else jf(v)) for k, v in r.items()
                                                          if k not in ("entry", "threshold", "rule")}
            for r in ds.to_dict("records")}
big = DEC.run_big()
big.to_csv(OUT / "rev_fdr_selection_big_foldlevel.csv", index=False)
bs = big.groupby(["n_train", "threshold", "rule"])[["frac_sel", "fdp", "power", "gain"]].mean().reset_index()
bs.to_csv(OUT / "rev_fdr_selection_big.csv", index=False)
N["fdr_big"] = {f"{int(r['n_train'])}|{r['threshold']}|{r['rule']}": {k: (None if pd.isna(v) else jf(v)) for k, v in r.items()
                                                                     if k not in ("n_train", "threshold", "rule")}
                for r in bs.to_dict("records")}

# --------------------------------------------------------------- LOSO
lo = []
for i, tag in enumerate(TAGS):
    f = R / f"results{tag}" / "rev_loso.csv"
    if f.exists():
        d = pd.read_csv(f); d["panel"] = i + 1; lo.append(d)
if lo:
    lo = pd.concat(lo, ignore_index=True)
    lo["entry"] = lo.model + "-" + lo.source
    lo.to_csv(OUT / "rev_S9_loso_foldlevel.csv", index=False)
    pm = lo.groupby(["panel", "entry", "regime"]).mean(numeric_only=True).reset_index()
    s = pm.groupby(["entry", "regime"]).agg(cov_conf=("cov_conformal_90", "mean"),
                                            cov_conf_se=("cov_conformal_90", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                            cov_native=("cov_native_90", "mean"), width_conf=("width_conformal_90", "mean"),
                                            pa=("pa", "mean")).reset_index()
    s.to_csv(OUT / "rev_loso.csv", index=False)
    N["loso"] = {f"{r['entry']}|{r['regime']}": {k: jf(v) for k, v in r.items() if k not in ("entry", "regime")}
                 for r in s.to_dict("records")}
    cell = lo[lo.regime == "training-pop calibration"].groupby(["panel", "held_out", "entry"]).cov_conformal_90.mean()
    N["loso_trainpop_cell_range"] = [jf(cell.min()), jf(cell.max())]
    cell = lo[lo.regime == "target-pop calibration"].groupby(["panel", "held_out", "entry"]).cov_conformal_90.mean()
    N["loso_target_cell_range"] = [jf(cell.min()), jf(cell.max())]
    byh = lo[lo.regime == "training-pop calibration"].groupby("held_out").cov_conformal_90.mean()
    N["loso_trainpop_by_heldout"] = {int(k): jf(v) for k, v in byh.items()}
    N["loso_pa_within_vs"] = None

# --------------------------------------------------------------- LOEO MLP
f = R / "results" / "rev_loeo_mlp.csv"
if f.exists():
    le = pd.read_csv(f)
    le.to_csv(OUT / "rev_S10_loeo_mlp_foldlevel.csv", index=False)
    N["loeo_mlp"] = {r: dict(cov_conf=jf(x.cov_conformal_90.mean()), cov_native=jf(x.cov_native_90.mean()),
                             width=jf(x.width_conformal_90.mean()))
                     for r, x in le.groupby("regime")}
    bt = le[le.regime == "unseen"].groupby("trait").cov_conformal_90.mean()
    N["loeo_mlp_unseen_by_trait"] = {k: jf(v) for k, v in bt.items()}
    bt = le[le.regime == "recal"].groupby("trait").cov_conformal_90.mean()
    N["loeo_mlp_recal_by_trait"] = {k: jf(v) for k, v in bt.items()}

# --------------------------------------------------------------- large training populations
f = R / "results_big" / "rev_bigtp.csv"
if f.exists():
    bg = pd.read_csv(f)
    bg.to_csv(OUT / "rev_S11_bigtp_foldlevel.csv", index=False)
    s = bg.groupby(["model", "n_train"]).agg(pa=("pa", "mean"), ta=("ta", "mean"), cov_native=("cov_native_90", "mean"),
                                             cov_conf=("cov_conformal_90", "mean"), w_native=("width_native_90", "mean"),
                                             w_conf=("width_conformal_90", "mean"), sec=("sec", "mean")).reset_index()
    s.to_csv(OUT / "rev_bigtp.csv", index=False)
    N["bigtp"] = {f"{r['model']}|{int(r['n_train'])}": {k: jf(v) for k, v in r.items() if k not in ("model", "n_train")}
                  for r in s.to_dict("records")}
    d8 = bg[bg.n_train == 800].pivot_table(index=["split", "trait"], columns="model", values="pa")
    N["bigtp_800_mlp_minus_gblup"] = jf((d8["MLP-ENS"] - d8["GBLUP"]).mean())
    N["bigtp_800_cells_mlp_beats"] = int((d8["MLP-ENS"] > d8["GBLUP"]).sum())
    N["bigtp_800_cells"] = int(len(d8))

# --------------------------------------------------------------- tuning
f = R / "results" / "tuning_all.csv"
if f.exists():
    ta = pd.read_csv(f)
    g = ta.groupby(["arch", "grid"]).val_r.mean().reset_index()
    g.to_csv(OUT / "rev_S7_tuning_summary.csv", index=False)
    N["tuning"] = {a: dict(best_grid=int(x.loc[x.val_r.idxmax(), "grid"]), best_r=jf(x.val_r.max()),
                           orig_best_r=jf(x[x.grid < 4].val_r.max()), n_configs=int(len(x)))
                   for a, x in g.groupby("arch")}
    N["best_hp_rev"] = json.load(open(R / "best_hp_rev.json"))

json.dump(N, open(R / "results" / "rev_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: N[k] for k in N if k in ("n_panels", "rep_pa_test", "rep_conf_pooled", "tuning")}, indent=1, default=float))

# stopping criterion: fold-level means over traits, corrected resampled t-test (J = 5)
st = pd.read_csv(R / "results" / "supp_stopping.csv")
N["stopping_corrected"] = {}
for a in ("MLP", "TRF"):
    x = st[st.arch == a].pivot_table(index=["trait", "fold"], columns="criterion", values="pa")
    f = (x["r"] - x["mse"]).groupby("fold").mean().values
    N["stopping_corrected"][a] = corrected_t(f, RATIO)
json.dump(N, open(R / "results" / "rev_numbers.json", "w"), indent=1, default=float)

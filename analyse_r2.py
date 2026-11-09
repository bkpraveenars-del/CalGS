"""Second-round revision analyses (Reviewer 2, round 2). Reads only saved outputs (CSV files and saved
predictions); no new model fits except the 20-split GBLUP enlarged-panel predictions (bigtp_gblup_preds.py with
CALGS_NSPLIT=20). Writes results/rev2_numbers.json and results/tables/rev2_*.csv.  Usage: python analyse_r2.py"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import config as C
import decision as DEC

R = C.ROOT
TAGS = ["", "_r1", "_r2", "_r3", "_r4"]
OUT = R / "results" / "tables"
OUT.mkdir(exist_ok=True)
N = {}
jf = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else float(np.round(x, 4))


def panel_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_1samp(d, 0.0)
    return dict(diff=jf(d.mean()), se=jf(d.std(ddof=1) / np.sqrt(len(d))), p=jf(t.pvalue))


# ------------------------------------------------------------------ fold-level CV results, 5 panels
rv = []
for i, tag in enumerate(TAGS):
    fs = sorted((R / f"results{tag}").glob("rev_cv*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["trait", "fold", "model", "source"])
    d["panel"] = i + 1
    rv.append(d)
rv = pd.concat(rv, ignore_index=True)
rv["entry"] = rv.model + "-" + rv.source

# 2.2 coverage by trait (means over the five panel means)
cols = ["cov_native_90", "cov_conformal_90", "covg_native_90", "covg_pev_only_90", "covg_conformal_90",
        "cov_pev_only_90"]
bt = rv.groupby(["entry", "trait", "panel"])[cols].mean().groupby(["entry", "trait"]).mean().reset_index()
bt.to_csv(OUT / "rev2_coverage_by_trait.csv", index=False)
N["cov_by_trait"] = {f"{r['entry']}|{r['trait']}": {c: jf(r[c]) for c in cols} for r in bt.to_dict("records")}
conf_tbv = bt.groupby("trait").covg_conformal_90.agg(["min", "max"])
N["conf_tbv_range_by_trait"] = {t: [jf(r["min"]), jf(r["max"])] for t, r in conf_tbv.iterrows()}
N["conf_tbv_range_all"] = [jf(bt.covg_conformal_90.min()), jf(bt.covg_conformal_90.max())]

# 2.7 marker number: GBLUP-thin vs MLP ensemble, five panel means
PM = rv.groupby(["panel", "entry"])[["pa", "ta"]].mean().reset_index()
g = lambda e: PM[PM.entry == e].set_index("panel")
N["thin_vs_mlp"] = {k: panel_t(g("GBLUP-thin-PEV")[k], g("MLP-ENS")[k]) for k in ("pa", "ta")}
N["full_vs_mlp"] = {k: panel_t(g("GBLUP-PEV")[k], g("MLP-ENS")[k]) for k in ("pa", "ta")}
N["full_vs_thin"] = {k: panel_t(g("GBLUP-PEV")[k], g("GBLUP-thin-PEV")[k]) for k in ("pa", "ta")}
for e in ("CNN-ENS", "TRF-ENS"):
    N[f"thin_vs_{e}"] = {k: panel_t(g("GBLUP-thin-PEV")[k], g(e)[k]) for k in ("pa", "ta")}

# 2.6 panel 1 test accuracy, original (2 repeats) vs extended search (1 repeat)
cm = pd.read_csv(R / "results" / "cv_main.csv")
orig = cm[cm.source.isin(["PEV", "ENS"])].groupby("model").pa.mean()
ext = rv[(rv.panel == 1) & rv.source.isin(["PEV", "ENS"])].groupby("model").pa.mean()
N["p1_orig_vs_ext"] = {m: dict(original=jf(orig.get(m, np.nan)), extended=jf(ext.get(m, np.nan)))
                       for m in ["GBLUP", "MLP", "CNN", "TRF"]}

# 2.5 Holm over five traits for the within-panel GBLUP-MLP tests (original configuration)
rn = json.load(open(R / "results" / "rev_numbers.json"))
ps = {t: v["p"] for t, v in rn["p1_pa_corrected"].items()}
order = sorted(ps, key=ps.get)
holm, run = {}, 0.0
for k, t in enumerate(order):
    run = max(run, min(1.0, (len(ps) - k) * ps[t]))
    holm[t] = jf(run)
N["p1_pa_holm"] = holm


# ------------------------------------------------------------------ 2.3 decision experiment, extended
def mean_null(pnull, q):
    """Largest set, taken in order of increasing null probability, whose mean null probability is <= q."""
    o = np.argsort(pnull)
    cm_ = np.cumsum(pnull[o]) / np.arange(1, len(pnull) + 1)
    ok = np.where(cm_ <= q)[0]
    sel = np.zeros(len(pnull), bool)
    if len(ok):
        sel[o[: ok.max() + 1]] = True
    return sel


def evaluate(path, panel, tbv_sd, extra=None, entries=DEC.ENTRIES):
    P = np.load(path)
    cal, te, y, tbv = P["cal"], P["te"], P["y"], P["tbv"]
    fit = np.setdiff1d(np.arange(len(y)), np.r_[cal, te])
    # TBV on the standardised phenotype scale (Eq. 14), projection estimated in the fitting set
    b, a = np.polyfit(tbv[fit], y[fit], 1)
    gsc = a + b * tbv
    t, k = path.stem.split("_")[:2]
    rows = []
    for e in entries:
        if f"{e}_yt" not in P:
            continue
        yc, yt, st = P[f"{e}_yc"], P[f"{e}_yt"], P[f"{e}_st"]
        sd_res = np.sqrt(np.mean((y[cal] - yc) ** 2))
        for cn, pc in DEC.PCTS.items():
            c = float(np.quantile(P["y_fit"], pc))
            c_tbv = float(np.quantile(tbv[fit], pc))
            elite = y[te] > c
            elite_g = gsc[te] > c                 # TBV projected to the phenotype scale exceeds c
            elite_q = tbv[te] > c_tbv             # TBV exceeds the same percentile of the fitting-set TBV
            p_nat = norm.cdf((c - yt) / np.maximum(st, 1e-12))
            p_plg = norm.cdf((c - yt) / sd_res)
            q = 0.20
            rules = {"conformal": DEC.bh(DEC.conformal_p(yc, y[cal], yt, c), q),
                     "native": DEC.bh(p_nat, q), "plugin": DEC.bh(p_plg, q),
                     "native_meannull": mean_null(p_nat, q), "plugin_meannull": mean_null(p_plg, q)}
            s = np.zeros(len(te), bool)
            s[np.argsort(-yt)[: int(np.ceil(0.10 * len(te)))]] = True
            rules["top10"] = s
            for rn_, sel in rules.items():
                n = int(sel.sum())
                rows.append(dict(panel=panel, trait=t, fold=int(k), entry=e, threshold=cn, rule=rn_, **(extra or {}),
                                 n_te=len(te), n_cal=len(cal), n_sel=n, frac_sel=n / len(te), any_sel=n > 0,
                                 fdp=float((~elite[sel]).mean()) if n else 0.0,
                                 fdp_cond=float((~elite[sel]).mean()) if n else np.nan,
                                 fdp_tbv=float((~elite_g[sel]).mean()) if n else 0.0,
                                 fdp_tbvq=float((~elite_q[sel]).mean()) if n else 0.0,
                                 power=float(sel[elite].mean()) if elite.any() else np.nan))
    return rows


rows = []
for i, tag in enumerate(TAGS):
    d = np.load(R / f"data{tag}" / "panel.npz")
    for f in sorted((R / f"results{tag}" / "preds").glob("*.npz")):
        t = f.stem.rsplit("_", 1)[0]
        rows += evaluate(f, i + 1, d["TBV"][:, C.TRAITS.index(t)].std())
dec = pd.DataFrame(rows)
dec.to_csv(OUT / "rev2_fdr_foldlevel.csv", index=False)


def summarise(df, by):
    pm = df.groupby(["panel"] + by).agg(frac_sel=("frac_sel", "mean"), any_sel=("any_sel", "mean"),
                                       fdr=("fdp", "mean"), fdr_tbv=("fdp_tbv", "mean"), fdr_tbvq=("fdp_tbvq", "mean"),
                                       power=("power", "mean")).reset_index()
    s = pm.groupby(by).mean(numeric_only=True).drop(columns="panel").reset_index()
    nz = df[df.any_sel].groupby(by).agg(n_sel_nonempty=("n_sel", "mean"), n_sel_min=("n_sel", "min"),
                                        n_sel_max=("n_sel", "max"), fdp_cond=("fdp_cond", "mean"),
                                        fdp_tbv_cond=("fdp_tbv", "mean")).reset_index()
    cnt = df.groupby(by).agg(units=("any_sel", "size"), units_sel=("any_sel", "sum")).reset_index()
    return s.merge(cnt, on=by).merge(nz, on=by, how="left")


S = summarise(dec, ["entry", "threshold", "rule"])
S.to_csv(OUT / "rev2_fdr_summary.csv", index=False)
N["fdr"] = {f"{r['entry']}|{r['threshold']}|{r['rule']}": {k: jf(v) for k, v in r.items()
                                                          if k not in ("entry", "threshold", "rule")}
            for r in S.to_dict("records")}
T = summarise(dec[dec.threshold == "q50"], ["entry", "trait", "rule"])
T.to_csv(OUT / "rev2_fdr_by_trait.csv", index=False)
N["fdr_by_trait_q50"] = {f"{r['entry']}|{r['trait']}|{r['rule']}": {k: jf(v) for k, v in r.items()
                                                                    if k not in ("entry", "trait", "rule")}
                         for r in T.to_dict("records")}
N["n_cal_range"] = [int(dec.n_cal.min()), int(dec.n_cal.max())]
N["n_te_range"] = [int(dec.n_te.min()), int(dec.n_te.max())]
# smallest non-empty BH set with conformal p-values: p >= 1/(n_cal+1) so k >= m / (q (n_cal + 1))
N["bh_min_k"] = {f"{nc}|{m}": int(np.ceil(m / (0.20 * (nc + 1)))) for nc in sorted(dec.n_cal.unique())
                 for m in sorted(dec.n_te.unique())}

# enlarged panel, GBLUP, all available splits
d = np.load(R / "data_big" / "panel.npz")
rows = []
for f in sorted((R / "results_big" / "preds_big").glob("*.npz")):
    t, s, n = f.stem.split("_")
    rows += [dict(r, split=int(s)) for r in evaluate(f, 0, d["TBV"][:, C.TRAITS.index(t)].std(), extra=dict(n_train=int(n)))]
big = pd.DataFrame(rows)
big.to_csv(OUT / "rev2_fdr_big_foldlevel.csv", index=False)
N["big_n_splits"] = int(big.split.nunique())
bq = big[(big.threshold == "q50")]
bs = bq.groupby(["n_train", "rule"]).agg(frac_sel=("frac_sel", "mean"), any_sel=("any_sel", "mean"),
                                         fdr=("fdp", "mean"),
                                         fdr_se=("fdp", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                         fdr_tbv=("fdp_tbv", "mean"), fdr_tbvq=("fdp_tbvq", "mean"),
                                         power=("power", "mean"),
                                         units=("fdp", "size")).reset_index()
nz = bq[bq.any_sel].groupby(["n_train", "rule"]).fdp_cond.mean().rename("fdp_cond").reset_index()
bs = bs.merge(nz, on=["n_train", "rule"], how="left")
bs.to_csv(OUT / "rev2_fdr_big.csv", index=False)
N["fdr_big_q50"] = {f"{int(r['n_train'])}|{r['rule']}": {k: jf(v) for k, v in r.items() if k not in ("n_train", "rule")}
                    for r in bs.to_dict("records")}

# ------------------------------------------------------------------ 2.4 LOSO and LOEO details
r2, ncal = [], []
for i, tag in enumerate(TAGS):
    d = np.load(R / f"data{tag}" / "panel.npz")
    grp = d["group"]
    for ti, t in enumerate(C.TRAITS):
        y = d["Y"][:, ti]
        fit = pd.Series(y).groupby(grp).transform("mean").values
        r2.append(dict(panel=i + 1, trait=t, r2=1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    sizes = np.bincount(grp)
    ncal.append(dict(panel=i + 1, sizes=sizes.tolist(), n_cal_target=[int(round(0.2 * s)) for s in sizes]))
r2 = pd.DataFrame(r2)
N["subpop_r2_mean"] = jf(r2.r2.mean())
N["subpop_r2_range"] = [jf(r2.r2.min()), jf(r2.r2.max())]
N["loso_sizes"] = ncal
for f, key in ((R / "results" / "supp_loeo.csv", "loeo_gblup"), (R / "results" / "rev_loeo_mlp.csv", "loeo_mlp")):
    x = pd.read_csv(f)
    N[key] = {r: dict(cov=jf(v.cov_conformal_90.mean()), width=jf(v.width_conformal_90.mean()))
              for r, v in x.groupby("regime")}

json.dump(N, open(R / "results" / "rev2_numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in N.items() if k not in ("fdr", "fdr_by_trait_q50", "cov_by_trait", "bh_min_k")},
                 indent=1))

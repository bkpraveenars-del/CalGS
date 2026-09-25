"""Build every manuscript table and headline number from the raw outputs.

Writes results/tables/table{1..6}.csv, table_S2_h2.csv, table_S5_attention.csv
and results/numbers.json (every number quoted in the text).
Usage:  python analyse.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import config as C

T = C.RESULTS / "tables"
T.mkdir(exist_ok=True)
MODELS = ["GBLUP", "MLP", "CNN", "TRF"]
LABEL = {"GBLUP": "GBLUP", "MLP": "MLP", "CNN": "1D-CNN", "TRF": "Transformer"}
CONS = ["native", "plugin", "epi_ale", "conformal", "adaptive"]


def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    d = np.load(C.DATA / "panel.npz")
    Y, TBV = d["Y"], d["TBV"]
    cv = pd.read_csv(C.RESULTS / "cv_main.csv")
    N = {}
    h2r = {t: float(np.corrcoef(TBV[:, i], Y[:, i])[0, 1] ** 2) for i, t in enumerate(C.TRAITS)}

    # ---------- panel diagnostics (chromosome 1 LD, PCA)
    X = d["X"].astype(float)
    c1 = d["chrom"] == 1
    R = np.corrcoef(X[:, c1].T) ** 2
    p1 = d["pos"][c1]
    iu = np.triu_indices(c1.sum(), 1)
    dist, r2 = np.abs(p1[iu[0]] - p1[iu[1]]), R[iu]
    edges = np.logspace(3, 7, 41)
    mid = np.sqrt(edges[:-1] * edges[1:])
    mr = np.array([r2[(dist >= a) & (dist < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    ok = ~np.isnan(mr)
    mid, mr = mid[ok], np.minimum.accumulate(mr[ok])     # monotone envelope for interpolation
    short = r2[dist < 5e3].mean()
    below = lambda thr: float(np.exp(np.interp(-thr, -mr, np.log(mid))) / 1e3)
    N["ld_short_r2"] = float(short)
    N["ld_half_kb"] = below(short / 2)
    N["ld_r2_005_kb"] = below(0.05)
    Z = X - X.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    N["pc_var_pct"] = (s[:3] ** 2 / np.sum(s ** 2) * 100).tolist()
    N["n_markers_qc"] = int(X.shape[1])
    N["n_markers_sim"] = int(len(d["pos_all"]))
    N["spacing_kb"] = float(sum(C.CHR_LEN_MB) * 1e3 / X.shape[1])
    N["thin_spacing_kb"] = float(sum(C.CHR_LEN_MB) * 1e3 / C.N_THIN)
    del X, Z, R

    # ---------- Table 1
    rows = []
    for i, t in enumerate(C.TRAITS):
        s = C.TRAIT_SPEC[t]
        e1 = d[f"{t}_obs"][:, 0]
        G = d[f"{t}_G"]
        lat = d[f"{t}_latent"]
        rows.append(dict(Trait=C.TRAIT_NAMES[t], Min=e1.min(), Max=e1.max(), Mean=e1.mean(), SD=e1.std(ddof=1),
                         Target_h2=s["h2"], Realised_h2=h2r[t], Additive_QTL=s["n_qtl"], Epistatic_pairs=s["n_epi"],
                         GxE_ratio=s["gxe"]))
    pd.DataFrame(rows).round(3).to_csv(T / "table1.csv", index=False)
    N["realised_h2"] = h2r

    # ---------- Table S2: per-environment observed-scale h2 (squared corr TBV, single-env phenotype)
    s2 = []
    for i, t in enumerate(C.TRAITS):
        r = {"trait": t, "target_linemean": C.TRAIT_SPEC[t]["h2"], "realised_linemean": h2r[t]}
        for e, (w, yr) in enumerate(C.ENVS):
            r[f"{w}-Y{yr}"] = float(np.corrcoef(TBV[:, i], d[f"{t}_obs"][:, e])[0, 1] ** 2)
        s2.append(r)
    pd.DataFrame(s2).round(3).to_csv(T / "table_S2_h2.csv", index=False)

    # ---------- Table 2
    rg = np.corrcoef(TBV.T)
    pd.DataFrame(rg, index=C.TRAITS, columns=C.TRAITS).round(2).to_csv(T / "table2.csv")
    N["rg"] = {f"{a}-{b}": float(rg[i, j]) for i, a in enumerate(C.TRAITS) for j, b in enumerate(C.TRAITS) if i < j}

    # ---------- Table 3 (deep models: ensemble mean as point prediction)
    pt = cv[(cv.model == "GBLUP") | (cv.source == "ENS")]
    rows, tests = [], {}
    for t in C.TRAITS:
        for m in MODELS:
            x = pt[(pt.trait == t) & (pt.model == m)]
            rows.append(dict(Trait=C.TRAIT_NAMES[t], Model=LABEL[m], PA=x.pa.mean(), PA_SE=se(x.pa),
                             TA=x.ta.mean(), TA_SE=se(x.ta), r_over_h=x.pa.mean() / np.sqrt(h2r[t])))
        g = pt[(pt.trait == t) & (pt.model == "GBLUP")].sort_values(["rep", "fold"])
        deep = {m: pt[(pt.trait == t) & (pt.model == m)].sort_values(["rep", "fold"]) for m in MODELS[1:]}
        bm = max(deep, key=lambda m: deep[m].pa.mean())
        diff = deep[bm].pa.values - g.pa.values
        tests[t] = dict(gblup=g.pa.mean(), gblup_se=se(g.pa), best=bm, diff=diff.mean(),
                        p=float(stats.ttest_rel(deep[bm].pa.values, g.pa.values).pvalue))
    t3 = pd.DataFrame(rows)
    t3.round(3).to_csv(T / "table3.csv", index=False)
    N["pa_tests"] = tests
    N["n_folds_total"] = int(C.N_FOLDS * C.N_REPEATS)
    N["r_over_h"] = dict(min=t3.r_over_h.min(), max=t3.r_over_h.max(), median=t3.r_over_h.median(),
                         argmax=t3.loc[t3.r_over_h.idxmax(), ["Trait", "Model"]].tolist())
    N["pa_mean"], N["ta_mean"] = t3.PA.mean(), t3.TA.mean()
    N["ta_pa_ratio"] = t3.TA.mean() / t3.PA.mean()
    N["mean_inv_h"] = float(np.mean([1 / np.sqrt(h2r[t]) for t in C.TRAITS]))
    N["frac_folds_ta_gt_pa"] = float((pt.ta > pt.pa).mean())

    # ---------- calibration summaries
    cv["entry"] = cv.model + "-" + cv.source
    lev = [80, 90, 95]
    cal = {}
    for c in CONS:
        for sub, mask in {"all": cv.index == cv.index, "deep": cv.model != "GBLUP", "gblup": cv.model == "GBLUP"}.items():
            cal[f"{c}_{sub}"] = [float(cv.loc[mask, f"cov_{c}_{L}"].mean()) for L in lev]
    for k, v in list(cal.items()):
        cal[k + "_mace"] = float(np.mean(np.abs(np.array(v) - np.array([0.8, 0.9, 0.95]))))
    N["coverage"] = cal
    n_cal = int(round(C.CAL_FRAC * 344))
    N["n_cal"] = n_cal
    N["conformal_upper_bound"] = [L / 100 + 1 / (n_cal + 1) for L in lev]
    # standard error of pooled conformal coverage (fold x trait units, models averaged)
    u = cv.groupby(["trait", "rep", "fold"])[[f"cov_conformal_{L}" for L in lev]].mean()
    N["conformal_cov_se_units"] = [float(se(u[c])) for c in u.columns]
    nat = cv[cv.model != "GBLUP"].groupby("entry").cov_native_90.mean()
    N["deep_native_90_range"] = [float(nat.min()), float(nat.max()), nat.idxmin()]

    # ---------- Table 4
    order = ["GBLUP-PEV", "MLP-MC", "MLP-ENS", "CNN-MC", "CNN-ENS", "TRF-MC", "TRF-ENS"]
    t4 = cv.groupby("entry")[[f"cov_{c}_90" for c in CONS]].mean().loc[order]
    t4.round(3).to_csv(T / "table4.csv")

    # ---------- Table 5 (widths, 90%, deep = ensemble source)
    rows = []
    for t in C.TRAITS:
        for m in MODELS:
            x = pt[(pt.trait == t) & (pt.model == m)]
            rows.append(dict(Trait=C.TRAIT_NAMES[t], Model=LABEL[m], **{c: x[f"width_{c}_90"].mean() for c in CONS}))
    t5 = pd.DataFrame(rows)
    t5.round(2).to_csv(T / "table5.csv", index=False)
    dw = cv[cv.model != "GBLUP"]
    N["width90"] = {
        "deep_native_all": float(dw.width_native_90.mean()), "deep_conformal_all": float(dw.width_conformal_90.mean()),
        "deep_adaptive_all": float(dw.width_adaptive_90.mean()),
        "deep_native_ens": float(t5[t5.Model != "GBLUP"].native.mean()),
        "deep_conformal_ens": float(t5[t5.Model != "GBLUP"].conformal.mean()),
        "deep_adaptive_range_ens": [float(t5[t5.Model != "GBLUP"].adaptive.min()), float(t5[t5.Model != "GBLUP"].adaptive.max())],
        "deep_conformal_range_ens": [float(t5[t5.Model != "GBLUP"].conformal.min()), float(t5[t5.Model != "GBLUP"].conformal.max())],
        "gblup_native": float(t5[t5.Model == "GBLUP"].native.mean()),
        "gblup_conformal": float(t5[t5.Model == "GBLUP"].conformal.mean()),
        "gblup_adaptive": float(t5[t5.Model == "GBLUP"].adaptive.mean()),
    }

    # ---------- Table 6 and decision tests
    rows, dec = [], {}
    for t in C.TRAITS:
        for m in MODELS:
            x = pt[(pt.trait == t) & (pt.model == m)]
            rows.append(dict(Trait=C.TRAIT_NAMES[t], Model=LABEL[m], Gain_point=x.gain_point.mean(),
                             Gain_point_SE=se(x.gain_point), Gain_lb=x.gain_lb.mean(), Gain_lb_SE=se(x.gain_lb),
                             Coincidence_point=x.coin_point.mean(), Coincidence_lb=x.coin_lb.mean(), Oracle=x.oracle.mean()))
    t6 = pd.DataFrame(rows)
    t6.round(3).to_csv(T / "table6.csv", index=False)
    for m in MODELS:
        x = pt[pt.model == m]
        dec[m] = dict(diff=float((x.gain_lb - x.gain_point).mean()),
                      p=float(stats.ttest_rel(x.gain_lb, x.gain_point).pvalue), n=len(x))
    N["decision"] = dec
    N["gain_point_mean"] = float(t6.Gain_point.mean())
    N["oracle_mean"] = float(t6.Oracle.mean())
    N["gain_frac"] = N["gain_point_mean"] / N["oracle_mean"]
    # dispersion of per-candidate uncertainty relative to its mean (why LB ranking hurts/does not)
    N["qa90_mean"] = cv.groupby("entry").qa90.mean().to_dict()

    # ---------- attention (Table S5)
    att = np.load(C.RESULTS / "attention.npz")
    X_n = d["X"].shape[1]
    thin = np.round(np.linspace(0, X_n - 1, C.N_THIN)).astype(int)
    chrom, pos = d["chrom"][thin], d["pos"][thin]
    chrom_all, pos_all = d["chrom_all"], d["pos_all"]
    patch = 64
    rows = []
    for t in C.TRAITS:
        A = np.mean([att[k] for k in att.files if k.startswith(t + "_")], 0)
        q = d[f"{t}_qtl"]
        qc, qp = chrom_all[q], pos_all[q]
        dens = np.zeros(len(A))
        for j in range(len(A)):
            a, b = j * patch, (j + 1) * patch - 1
            # a patch spans from its first to its last thinned marker (possibly across a chromosome boundary)
            key_a = chrom[a] * 1e9 + pos[a]
            key_b = chrom[b] * 1e9 + pos[b]
            kq = qc * 1e9 + qp
            dens[j] = np.sum((kq >= key_a) & (kq <= key_b))
        rho, p = stats.spearmanr(A, dens)
        top = A >= np.quantile(A, 0.8)
        mw = stats.mannwhitneyu(dens[top], dens[~top], alternative="greater").pvalue
        rows.append(dict(trait=t, spearman_rho=rho, p=p, qtl_top_quintile=dens[top].mean(),
                         qtl_other=dens[~top].mean(), mannwhitney_p_greater=mw))
        np.save(C.RESULTS / f"attention_profile_{t}.npy", np.vstack([A, dens]))
    ta = pd.DataFrame(rows)
    ta.round(4).to_csv(T / "table_S5_attention.csv", index=False)
    N["attention"] = ta.set_index("trait").to_dict("index")

    # ---------- supplementary experiments
    if (C.RESULTS / "supp_trainsize.csv").exists():
        s3 = pd.read_csv(C.RESULTS / "supp_trainsize.csv")
        g = s3.groupby(["model", "n_train"])[["pa", "cov_native_90", "cov_conformal_90", "width_native_90", "width_conformal_90"]].mean()
        g.round(3).to_csv(T / "table_S3_trainsize.csv")
        N["trainsize"] = {f"{m}_{n}": r.to_dict() for (m, n), r in g.iterrows()}
    if (C.RESULTS / "supp_loeo.csv").exists():
        s4 = pd.read_csv(C.RESULTS / "supp_loeo.csv")
        g = s4.groupby(["regime", "trait"])[["cov_conformal_90", "width_conformal_90"]].mean()
        g.round(3).to_csv(T / "table_S4_loeo.csv")
        N["loeo"] = s4.groupby("regime")[["cov_conformal_90", "width_conformal_90"]].mean().to_dict("index")
        N["loeo_range_unseen"] = [float(g.loc["unseen"].cov_conformal_90.min()), float(g.loc["unseen"].cov_conformal_90.max())]
        N["loeo_range_recal"] = [float(g.loc["recal"].cov_conformal_90.min()), float(g.loc["recal"].cov_conformal_90.max())]
    if (C.RESULTS / "supp_stopping.csv").exists():
        s6 = pd.read_csv(C.RESULTS / "supp_stopping.csv")
        g = s6.groupby(["arch", "criterion"])[["best_epoch", "pa"]].mean()
        g.round(3).to_csv(T / "table_S6_stopping.csv")
        N["stopping"] = {f"{a}_{c}": r.to_dict() for (a, c), r in g.iterrows()}
        for a in ("MLP", "TRF"):
            x = s6[s6.arch == a].pivot_table(index=["trait", "fold"], columns="criterion", values="pa")
            N["stopping"][f"{a}_p"] = float(stats.ttest_rel(x["r"], x["mse"]).pvalue)
    tun = pd.read_csv(C.RESULTS / "tuning.csv")
    tun.groupby(["arch", "grid"]).agg(val_r=("val_r", "mean"), best_epoch=("best_epoch", "mean")).round(3).to_csv(T / "table_S7_tuning.csv")
    N["best_hp"] = json.load(open(C.RESULTS / "best_hp.json"))
    N["best_epoch_cv"] = cv[cv.model != "GBLUP"].groupby("model").best_epoch.mean().to_dict()
    json.dump(N, open(C.RESULTS / "numbers.json", "w"), indent=2, default=float)
    print(json.dumps(N, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

"""Revision experiments (Frontiers in Plant Science, manuscript 2021142).

Stages (run per simulated panel; the panel is selected with CALGS_SEED / CALGS_TAG):
  cv       one repeat of stratified 5-fold CV with all models, adding
           - coverage of the true breeding value (TBV) as well as of the phenotype
           - a PEV-only GBLUP interval (targets the breeding value, not the phenotype)
           - GBLUP fitted on the same 4,096 thinned markers given to the deep models
           - a three-way triage decision (advance / discard / phenotype again)
           -> results<TAG>/rev_cv.csv
  loso     leave-one-subpopulation-out, GBLUP and MLP, calibration drawn from the
           training subpopulations or from the target subpopulation -> rev_loso.csv
  loeo     leave-one-environment-out for the MLP (MC dropout)         -> rev_loeo_mlp.csv
  bigtp    training-population size up to 800 on an enlarged panel   -> rev_bigtp.csv
Hyperparameters are read from best_hp_rev.json (expanded search, tune_ext.py).
Usage: python run_revision.py cv loso
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
import config as C
import models_dl as D
from models_stat import GBLUP, vanraden_G, intervals
from run_experiment import load, folds, inner_split, thin_index, standardise_X

torch.set_num_threads(int(os.environ.get("CALGS_THREADS", 1)))
HP = json.load(open(os.environ.get("CALGS_HP", C.ROOT / "best_hp_rev.json")))
CONSTR = ["native", "plugin", "epi_ale", "conformal", "adaptive"]


def tbv_on_y_scale(y, tbv, fit):
    """Linear projection of the standardised phenotype on the TBV, estimated in the fitting set:
    the additive genetic value expressed on the scale on which every interval is built."""
    b, a = np.polyfit(tbv[fit], y[fit], 1)
    return a + b * tbv


def triage(lo, hi, y_te, g_te, thr_y):
    """Advance if the lower bound clears the threshold, discard if the upper bound falls
    below it, otherwise send the line to another round of phenotyping."""
    adv, dis = lo > thr_y, hi < thr_y
    top_y = y_te > thr_y
    k = int(np.ceil(0.10 * len(g_te)))
    top_g = np.zeros(len(g_te), bool)
    top_g[np.argsort(-g_te)[:k]] = True
    return dict(frac_adv=adv.mean(), frac_dis=dis.mean(),
                lost_top_y=dis[top_y].mean() if top_y.any() else np.nan,
                lost_top_g=dis[top_g].mean(),
                false_adv=(~top_y[adv]).mean() if adv.any() else np.nan)


def entry(rows, base, y_cal, yh_cal, sd_cal, y_te, yh_te, sd_te, g_te, tbv_te, scale, tbv_sd, thr_y,
          pev_te=None):
    r = dict(base)
    r["pa"] = float(np.corrcoef(yh_te, y_te)[0, 1])
    r["ta"] = float(np.corrcoef(yh_te, tbv_te)[0, 1])
    for lev in (0.80, 0.90):
        ints, _ = intervals(yh_cal, y_cal, sd_cal, yh_te, sd_te, lev)
        if pev_te is not None:
            h = norm.ppf(0.5 + lev / 2) * np.sqrt(np.maximum(pev_te, 0))
            ints["pev_only"] = (yh_te - h, yh_te + h)
        for c, (lo, hi) in ints.items():
            tag = f"{c}_{int(lev * 100)}"
            r[f"cov_{tag}"] = float(np.mean((y_te >= lo) & (y_te <= hi)))
            r[f"covg_{tag}"] = float(np.mean((g_te >= lo) & (g_te <= hi)))
            r[f"width_{tag}"] = float(np.mean(hi - lo) * scale)
            if lev == 0.80:
                for kk, v in triage(lo, hi, y_te, g_te, thr_y).items():
                    r[f"{kk}_{c}"] = float(v)
        if lev == 0.90:
            lb = ints["adaptive"][0]
    n = len(y_te)
    k = int(np.ceil(C.SELECTION_FRACTION * n))
    top = set(np.argsort(-tbv_te)[:k])
    for name, crit in (("point", yh_te), ("lb", lb)):
        sel = np.argsort(-crit)[:k]
        r[f"gain_{name}"] = float((tbv_te[sel].mean() - tbv_te.mean()) / tbv_sd)
        r[f"coin_{name}"] = len(top & set(sel)) / k
    r["oracle"] = float((np.sort(tbv_te)[::-1][:k].mean() - tbv_te.mean()) / tbv_sd)
    # point-rank rule discarding the same number of lines as conformal triage
    nd = int(round(r["frac_dis_conformal"] * n))
    dis = np.zeros(n, bool)
    dis[np.argsort(yh_te)[:nd]] = True
    top_y = y_te > thr_y
    r["lost_top_y_rank"] = float(dis[top_y].mean()) if top_y.any() else np.nan
    r["n_test"] = n
    rows.append(r)


def deep_fit(arch, Xs, y, tr, va, seed):
    nets, eps = [], []
    for e in range(C.ENSEMBLE_SIZE):
        net, _, ep = D.train_net(arch, HP[arch], Xs[tr], y[tr], Xs[va], y[va], seed=seed + e)
        nets.append(net); eps.append(ep)
    return nets, eps


def deep_preds(nets, Xs, idx, mc_seed):
    p = np.stack([D.predict(n_, Xs[idx]) for n_ in nets])
    mc = D.mc_dropout(nets[0], Xs[idx], C.MC_PASSES, seed=mc_seed)
    return (p.mean(0), p.std(0, ddof=1)), mc


# ------------------------------------------------------------------ stage: cv
def cv():
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    ti_ = thin_index(X.shape[1])
    Xt_all = X[:, ti_]
    fsel = os.environ.get("CALGS_FOLDS")
    out = C.RESULTS / ("rev_cv.csv" if not fsel else f"rev_cv_f{fsel.replace(',', '')}.csv")
    fsel = [int(v) for v in fsel.split(",")] if fsel else list(range(C.N_FOLDS))
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["trait"], r["fold"]) for r in rows}
    rep = 0
    f = folds(group, rep)
    for k in fsel:
        te = np.where(f == k)[0]
        tr, va, cal = inner_split(np.where(f != k)[0], rep, k)
        fit = np.r_[tr, va]
        G = vanraden_G(X, X[fit].mean(0) / 2)
        Gt = vanraden_G(Xt_all, Xt_all[fit].mean(0) / 2)
        Xs = standardise_X(Xt_all, fit)
        for ti, t in enumerate(C.TRAITS):
            if (t, k) in done:
                continue
            t0 = time.time()
            mu, sd = Y[fit, ti].mean(), Y[fit, ti].std()
            y = (Y[:, ti] - mu) / sd
            scale = sd / Y[:, ti].std()
            tbv, tbv_sd = TBV[:, ti], TBV[:, ti].std()
            g = tbv_on_y_scale(y, tbv, fit)
            thr = float(np.quantile(y[fit], 0.90))
            base = dict(seed=C.MASTER_SEED, trait=t, rep=rep, fold=k)
            a = (tbv[te], scale, tbv_sd, thr)
            P = dict(cal=cal, te=te, y=y, g=g, tbv=tbv, thr=thr, y_fit=y[fit])
            for name, GG in (("GBLUP", G), ("GBLUP-thin", Gt)):
                m = GBLUP().fit(GG[np.ix_(fit, fit)], y[fit])
                yc, _, sc = m.predict(GG[np.ix_(cal, fit)], np.diag(GG)[cal])
                yt, pt, st = m.predict(GG[np.ix_(te, fit)], np.diag(GG)[te])
                entry(rows, dict(base, model=name, source="PEV", h2_reml=1 / (1 + m.lam)),
                      y[cal], yc, sc, y[te], yt, st, g[te], *a, pev_te=pt)
                P.update({f"{name}-PEV_yc": yc, f"{name}-PEV_sc": sc, f"{name}-PEV_yt": yt, f"{name}-PEV_st": st,
                          f"{name}-PEV_pev": pt})
            for arch in ("MLP", "CNN", "TRF"):
                nets, eps = deep_fit(arch, Xs, y, tr, va, seed=int(1e5 + 1e3 * k + 10 * ti))
                (ec, esc), (mc_, msc) = deep_preds(nets, Xs, cal, 7 + k)
                (et, est), (mt, mst) = deep_preds(nets, Xs, te, 8 + k)
                entry(rows, dict(base, model=arch, source="ENS", best_epoch=float(np.mean(eps))),
                      y[cal], ec, esc, y[te], et, est, g[te], *a)
                entry(rows, dict(base, model=arch, source="MC", best_epoch=float(eps[0])),
                      y[cal], mc_, msc, y[te], mt, mst, g[te], *a)
                P.update({f"{arch}-ENS_yc": ec, f"{arch}-ENS_sc": esc, f"{arch}-ENS_yt": et, f"{arch}-ENS_st": est,
                          f"{arch}-MC_yc": mc_, f"{arch}-MC_sc": msc, f"{arch}-MC_yt": mt, f"{arch}-MC_st": mst})
            (C.RESULTS / "preds").mkdir(exist_ok=True)
            np.savez_compressed(C.RESULTS / "preds" / f"{t}_{k}.npz", **P)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"[{C.MASTER_SEED}] cv fold {k} {t} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ stage: loso
def loso():
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    Xt_all = X[:, thin_index(X.shape[1])]
    out = C.RESULTS / "rev_loso.csv"
    if out.exists() and len(pd.read_csv(out)) >= 90:
        print("loso already complete", flush=True)
        return
    rows = []
    for h in np.unique(group):
        rng = np.random.default_rng([C.MASTER_SEED, 700, int(h)])
        idx = rng.permutation(np.where(group != h)[0])
        n_cal = int(round(C.CAL_FRAC * len(idx)))
        cal, rest = idx[:n_cal], idx[n_cal:]
        n_val = int(round(C.VAL_FRAC * len(rest)))
        va, tr = rest[:n_val], rest[n_val:]
        fit = np.r_[tr, va]
        tgt = rng.permutation(np.where(group == h)[0])
        n_tc = int(round(C.CAL_FRAC * len(tgt)))
        cal_t, te = tgt[:n_tc], tgt[n_tc:]          # target-population calibration lines, test lines
        G = vanraden_G(X, X[fit].mean(0) / 2)
        Xs = standardise_X(Xt_all, fit)
        for ti, t in enumerate(C.TRAITS):
            mu, sd = Y[fit, ti].mean(), Y[fit, ti].std()
            y = (Y[:, ti] - mu) / sd
            scale = sd / Y[:, ti].std()
            m = GBLUP().fit(G[np.ix_(fit, fit)], y[fit])
            preds = {}
            for nm, ii in (("cal", cal), ("calt", cal_t), ("te", te)):
                yh, _, s = m.predict(G[np.ix_(ii, fit)], np.diag(G)[ii])
                preds.setdefault(("GBLUP", "PEV"), {})[nm] = (yh, s)
            nets, _ = deep_fit("MLP", Xs, y, tr, va, seed=int(7e5 + 1e3 * h + 10 * ti))
            for nm, ii, sd_ in (("cal", cal, 31), ("calt", cal_t, 32), ("te", te, 33)):
                (e, es), (mm, ms) = deep_preds(nets, Xs, ii, sd_)
                preds.setdefault(("MLP", "ENS"), {})[nm] = (e, es)
                preds.setdefault(("MLP", "MC"), {})[nm] = (mm, ms)
            for (model, src), p in preds.items():
                yt, st = p["te"]
                for regime, (yc_obs, (yc, sc)) in (("training-pop calibration", (y[cal], p["cal"])),
                                                    ("target-pop calibration", (y[cal_t], p["calt"]))):
                    ints, _ = intervals(yc, yc_obs, sc, yt, st, 0.90)
                    rr = dict(seed=C.MASTER_SEED, held_out=int(h), trait=t, model=model, source=src, regime=regime,
                              pa=float(np.corrcoef(yt, y[te])[0, 1]), n_test=len(te))
                    for c in ("native", "conformal"):
                        lo, hi = ints[c]
                        rr[f"cov_{c}_90"] = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
                        rr[f"width_{c}_90"] = float(np.mean(hi - lo) * scale)
                    rows.append(rr)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"[{C.MASTER_SEED}] loso held-out {h} {t}", flush=True)


# ------------------------------------------------------------------ stage: loeo (MLP)
def loeo():
    d = load()
    X, group = d["X"].astype(float), d["group"]
    Xt_all = X[:, thin_index(X.shape[1])]
    f = folds(group, 0)
    rows = []
    for k in range(C.N_FOLDS):
        te = np.where(f == k)[0]
        tr, va, cal = inner_split(np.where(f != k)[0], 0, k)
        fit = np.r_[tr, va]
        Xs = standardise_X(Xt_all, fit)
        for ti, t in enumerate(C.TRAITS):
            obs = d[f"{t}_obs"]
            for e in range(len(C.ENVS)):
                others = [j for j in range(len(C.ENVS)) if j != e]
                ybar = obs[:, others].mean(1)
                mu, sd = ybar[fit].mean(), ybar[fit].std()
                y5, ye = (ybar - mu) / sd, (obs[:, e] - mu) / sd
                net, _, _ = D.train_net("MLP", HP["MLP"], Xs[tr], y5[tr], Xs[va], y5[va],
                                        seed=int(8e5 + 1e3 * k + 10 * ti + e))
                yc, sc = D.mc_dropout(net, Xs[cal], C.MC_PASSES, seed=41)
                yt, st = D.mc_dropout(net, Xs[te], C.MC_PASSES, seed=42)
                shift = ye[fit].mean() - y5[fit].mean()
                for name, (yco, ych, yth) in {"unseen": (y5[cal], yc, yt), "centred": (y5[cal], yc, yt + shift),
                                              "recal": (ye[cal], yc + shift, yt + shift)}.items():
                    ints, _ = intervals(ych, yco, sc, yth, st, 0.90)
                    rr = dict(trait=t, fold=k, env=f"{C.ENVS[e][0]}-Y{C.ENVS[e][1]}", regime=name, model="MLP-MC")
                    for c in ("native", "conformal"):
                        lo, hi = ints[c]
                        rr[f"cov_{c}_90"] = float(np.mean((ye[te] >= lo) & (ye[te] <= hi)))
                        rr[f"width_{c}_90"] = float(np.mean(hi - lo) * sd / obs[:, e].std())
                    rows.append(rr)
            print("loeo-mlp fold", k, t, flush=True)
        pd.DataFrame(rows).to_csv(C.RESULTS / "rev_loeo_mlp.csv", index=False)


# ------------------------------------------------------------------ stage: bigtp
SIZES = [100, 200, 400, 600, 800]


def bigtp():
    """Enlarged panel (CALGS_SCALE=3, 1,290 accessions). Fixed test (200), calibration (150)
    and validation (100) sets per split; proper training set subsampled to SIZES."""
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    Xt_all = X[:, thin_index(X.shape[1])]
    rows = []
    for s in range(2):
        rng = np.random.default_rng([C.MASTER_SEED, 900, s])
        idx = rng.permutation(len(group))
        te, cal, va, pool = idx[:200], idx[200:350], idx[350:450], idx[450:]
        for n_tr in SIZES:
            tr = pool[:n_tr]
            fit = np.r_[tr, va]
            t0 = time.time()
            G = vanraden_G(X, X[fit].mean(0) / 2)
            t_g = time.time() - t0
            Xs = standardise_X(Xt_all, fit)
            for ti, t in enumerate(C.TRAITS):
                mu, sd = Y[fit, ti].mean(), Y[fit, ti].std()
                y = (Y[:, ti] - mu) / sd
                scale = sd / Y[:, ti].std()
                t0 = time.time()
                m = GBLUP().fit(G[np.ix_(fit, fit)], y[fit])
                yc, _, sc = m.predict(G[np.ix_(cal, fit)], np.diag(G)[cal])
                yt, _, st = m.predict(G[np.ix_(te, fit)], np.diag(G)[te])
                sec_g = time.time() - t0 + t_g
                t0 = time.time()
                nets, _ = deep_fit("MLP", Xs, y, tr, va, seed=int(9e5 + 1e3 * s + 10 * ti + n_tr))
                (ec, esc), (mc_, msc) = deep_preds(nets, Xs, cal, 51)
                (et, est), (mt, mst) = deep_preds(nets, Xs, te, 52)
                sec_m = time.time() - t0
                for model, (a, b_, c_, d_), sec in (("GBLUP", (yc, sc, yt, st), sec_g),
                                                    ("MLP-ENS", (ec, esc, et, est), sec_m),
                                                    ("MLP-MC", (mc_, msc, mt, mst), sec_m)):
                    ints, _ = intervals(a, y[cal], b_, c_, d_, 0.90)
                    rr = dict(split=s, trait=t, n_train=n_tr, n_fit=len(fit), model=model, sec=sec,
                              pa=float(np.corrcoef(c_, y[te])[0, 1]),
                              ta=float(np.corrcoef(c_, TBV[te, ti])[0, 1]))
                    for c in ("native", "conformal"):
                        lo, hi = ints[c]
                        rr[f"cov_{c}_90"] = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
                        rr[f"width_{c}_90"] = float(np.mean(hi - lo) * scale)
                    rows.append(rr)
            pd.DataFrame(rows).to_csv(C.RESULTS / "rev_bigtp.csv", index=False)
            print("bigtp split", s, "n", n_tr, flush=True)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        {"cv": cv, "loso": loso, "loeo": loeo, "bigtp": bigtp}[st]()

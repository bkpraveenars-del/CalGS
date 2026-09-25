"""Supplementary experiments (repeat 0 folds only).

1. Training-population size: GBLUP and MLP (MC dropout) with the proper training
   set subsampled; validation and calibration sets fixed.   -> results/supp_trainsize.csv
2. Leave-one-environment-out (GBLUP): models fitted and calibrated on the mean of
   five environments, tested on single-environment phenotypes of the sixth.
   Three calibration regimes:
     unseen      calibration on the 5-environment means (exchangeability broken)
     centred     as unseen, predictions shifted by the target-environment main effect
     recal       calibration lines phenotyped in the target environment
                                                             -> results/supp_loeo.csv
Usage:  python run_supp.py
"""
import json
import numpy as np
import pandas as pd
import config as C
from models_stat import GBLUP, vanraden_G, intervals
import models_dl as D
from run_experiment import load, folds, inner_split, thin_index, standardise_X

SIZES = [80, 120, 160, 219]


def trainsize():
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    best = json.load(open(C.RESULTS / "best_hp.json"))
    Xt_all = X[:, thin_index(X.shape[1])]
    f = folds(group, 0)
    rows = []
    for k in range(C.N_FOLDS):
        te = np.where(f == k)[0]
        tr_full, va, cal = inner_split(np.where(f != k)[0], 0, k)
        for n_tr in SIZES:
            tr = tr_full[:n_tr]
            fit = np.r_[tr, va]
            G = vanraden_G(X, X[fit].mean(0) / 2)
            Xs = standardise_X(Xt_all, fit)
            for ti, t in enumerate(C.TRAITS):
                mu, sd = Y[fit, ti].mean(), Y[fit, ti].std()
                y = (Y[:, ti] - mu) / sd
                scale = sd / Y[:, ti].std()
                m = GBLUP().fit(G[np.ix_(fit, fit)], y[fit])
                yc, _, sc = m.predict(G[np.ix_(cal, fit)], np.diag(G)[cal])
                yt, _, st = m.predict(G[np.ix_(te, fit)], np.diag(G)[te])
                net, _, _ = D.train_net("MLP", best["MLP"], Xs[tr], y[tr], Xs[va], y[va], seed=5000 + 10 * k + ti)
                mc_c = D.mc_dropout(net, Xs[cal], C.MC_PASSES, seed=11)
                mc_t = D.mc_dropout(net, Xs[te], C.MC_PASSES, seed=12)
                for model, (ycal, sdcal, yte, sdte) in {"GBLUP": (yc, sc, yt, st), "MLP-MC": (*mc_c, *mc_t)}.items():
                    ints, _ = intervals(ycal, y[cal], sdcal, yte, sdte, 0.90)
                    r = dict(trait=t, fold=k, n_train=n_tr, n_fit=len(fit), model=model,
                             pa=float(np.corrcoef(yte, y[te])[0, 1]), ta=float(np.corrcoef(yte, TBV[te, ti])[0, 1]))
                    for c in ("native", "conformal"):
                        lo, hi = ints[c]
                        r[f"cov_{c}_90"] = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
                        r[f"width_{c}_90"] = float(np.mean(hi - lo) * scale)
                    rows.append(r)
            print("trainsize fold", k, "n", n_tr, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "supp_trainsize.csv", index=False)


def loeo():
    d = load()
    X, group = d["X"].astype(float), d["group"]
    f = folds(group, 0)
    rows = []
    for k in range(C.N_FOLDS):
        te = np.where(f == k)[0]
        tr, va, cal = inner_split(np.where(f != k)[0], 0, k)
        fit = np.r_[tr, va]
        G = vanraden_G(X, X[fit].mean(0) / 2)
        for t in C.TRAITS:
            obs = d[f"{t}_obs"]                          # accessions x 6 environments
            for e in range(len(C.ENVS)):
                others = [j for j in range(len(C.ENVS)) if j != e]
                ybar = obs[:, others].mean(1)
                mu, sd = ybar[fit].mean(), ybar[fit].std()
                y5 = (ybar - mu) / sd                    # training scale
                ye = (obs[:, e] - mu) / sd               # target-environment phenotype, same scale
                m = GBLUP().fit(G[np.ix_(fit, fit)], y5[fit])
                yc, _, sc = m.predict(G[np.ix_(cal, fit)], np.diag(G)[cal])
                yt, _, st = m.predict(G[np.ix_(te, fit)], np.diag(G)[te])
                shift = ye[fit].mean() - y5[fit].mean()  # environment main effect (known after harvest)
                regimes = {
                    "unseen": (y5[cal], yc, yt),
                    "centred": (y5[cal], yc, yt + shift),
                    "recal": (ye[cal], yc + shift, yt + shift),
                }
                for name, (ycal_obs, ycal_hat, yte_hat) in regimes.items():
                    ints, _ = intervals(ycal_hat, ycal_obs, sc, yte_hat, st, 0.90)
                    lo, hi = ints["conformal"]
                    rows.append(dict(trait=t, fold=k, env=f"{C.ENVS[e][0]}-Y{C.ENVS[e][1]}", regime=name,
                                     cov_conformal_90=float(np.mean((ye[te] >= lo) & (ye[te] <= hi))),
                                     width_conformal_90=float(np.mean(hi - lo) * sd / obs[:, e].std())))
        print("loeo fold", k, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "supp_loeo.csv", index=False)


def stopping():
    """Early stopping on validation correlation (default) versus validation MSE."""
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    best = json.load(open(C.RESULTS / "best_hp.json"))
    Xt_all = X[:, thin_index(X.shape[1])]
    f = folds(group, 0)
    rows = []
    for k in range(C.N_FOLDS):
        te = np.where(f == k)[0]
        tr, va, cal = inner_split(np.where(f != k)[0], 0, k)
        fit = np.r_[tr, va]
        Xs = standardise_X(Xt_all, fit)
        for ti, t in enumerate(C.TRAITS):
            y = (Y[:, ti] - Y[fit, ti].mean()) / Y[fit, ti].std()
            for arch in ("MLP", "TRF"):
                for crit in ("r", "mse"):
                    net, _, ep = D.train_net(arch, best[arch], Xs[tr], y[tr], Xs[va], y[va],
                                             seed=9000 + 10 * k + ti, criterion=crit)
                    p = D.predict(net, Xs[te])
                    rows.append(dict(trait=t, fold=k, arch=arch, criterion=crit, best_epoch=ep,
                                     pa=float(np.corrcoef(p, y[te])[0, 1])))
        print("stopping fold", k, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "supp_stopping.csv", index=False)


if __name__ == "__main__":
    import sys
    stages = sys.argv[1:] or ["loeo", "trainsize", "stopping"]
    for s_ in stages:
        {"loeo": loeo, "trainsize": trainsize, "stopping": stopping}[s_]()

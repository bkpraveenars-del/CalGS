"""Hyperparameter tuning (once per architecture, on two tuning traits) and the
main repeated stratified cross-validation experiment.

Outputs
  results/tuning.csv          validation correlation for every grid point
  results/best_hp.json        selected hyperparameters per architecture
  results/cv_main.csv         one row per trait x repeat x fold x model x uncertainty source
  results/attention.npz       class-token attention per trait/repeat/fold
Usage:  python run_experiment.py [--stage tune|cv|all]
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import config as C
from models_stat import GBLUP, vanraden_G, intervals
import models_dl as D


def load():
    d = np.load(C.DATA / "panel.npz")
    return d


def folds(group, rep):
    """Stratified (by subpopulation) k-fold assignment for one repeat."""
    rng = np.random.default_rng([C.MASTER_SEED, 100 + rep])
    f = np.empty(len(group), int)
    for g in np.unique(group):
        idx = rng.permutation(np.where(group == g)[0])
        f[idx] = np.arange(len(idx)) % C.N_FOLDS
    return f


def inner_split(train_idx, rep, fold):
    """Training partition -> proper-train, validation (early stopping), calibration (held out)."""
    rng = np.random.default_rng([C.MASTER_SEED, 200 + rep, fold])
    idx = rng.permutation(train_idx)
    n_cal = int(round(C.CAL_FRAC * len(idx)))
    cal, rest = idx[:n_cal], idx[n_cal:]
    n_val = int(round(C.VAL_FRAC * len(rest)))
    return rest[n_val:], rest[:n_val], cal


def thin_index(n_markers):
    return np.round(np.linspace(0, n_markers - 1, C.N_THIN)).astype(int)


def standardise_X(X, ref):
    m, s = X[ref].mean(0), X[ref].std(0)
    s[s == 0] = 1
    return ((X - m) / s).astype(np.float32)


def tune():
    d = load()
    X, Y, group = d["X"].astype(float), d["Y"], d["group"]
    Xt_all = X[:, thin_index(X.shape[1])]
    f = folds(group, 0)
    rows = []
    for arch, grid in D.GRID.items():
        for gi, hp in enumerate(grid):
            for t in C.TUNING_TRAITS:
                ti = C.TRAITS.index(t)
                for k in range(C.N_FOLDS):
                    tr, va, _ = inner_split(np.where(f != k)[0], 0, k)
                    Xs = standardise_X(Xt_all, np.r_[tr, va])
                    y = (Y[:, ti] - Y[np.r_[tr, va], ti].mean()) / Y[np.r_[tr, va], ti].std()
                    _, r, ep = D.train_net(arch, hp, Xs[tr], y[tr], Xs[va], y[va], seed=1000 + k)
                    rows.append(dict(arch=arch, grid=gi, trait=t, fold=k, val_r=r, best_epoch=ep, **hp))
                    print(arch, gi, t, k, round(r, 3), ep, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "tuning.csv", index=False)
    best = {}
    for arch in D.GRID:
        m = df[df.arch == arch].groupby("grid").val_r.mean()
        best[arch] = D.GRID[arch][int(m.idxmax())]
    json.dump(best, open(C.RESULTS / "best_hp.json", "w"), indent=2)
    print(best)


def evaluate_entry(rows, base, y_cal, yh_cal, sd_cal, y_te, yh_te, sd_te, tbv_te, scale, tbv_sd):
    r = dict(base)
    r["pa"] = float(np.corrcoef(yh_te, y_te)[0, 1])
    r["ta"] = float(np.corrcoef(yh_te, tbv_te)[0, 1])
    for lev in C.LEVELS:
        ints, qa = intervals(yh_cal, y_cal, sd_cal, yh_te, sd_te, lev)
        for k, (lo, hi) in ints.items():
            r[f"cov_{k}_{int(lev*100)}"] = float(np.mean((y_te >= lo) & (y_te <= hi)))
            r[f"width_{k}_{int(lev*100)}"] = float(np.mean(hi - lo) * scale)
        if lev == 0.90:
            lb = ints["adaptive"][0]
            r["qa90"] = qa
    n = len(y_te)
    k = int(np.ceil(C.SELECTION_FRACTION * n))
    top_true = set(np.argsort(-tbv_te)[:k])
    for name, crit in (("point", yh_te), ("lb", lb)):
        sel = np.argsort(-crit)[:k]
        r[f"gain_{name}"] = float((tbv_te[sel].mean() - tbv_te.mean()) / tbv_sd)
        r[f"coin_{name}"] = len(top_true & set(sel)) / k
    r["oracle"] = float((np.sort(tbv_te)[::-1][:k].mean() - tbv_te.mean()) / tbv_sd)
    r["n_test"] = n
    rows.append(r)


def cv():
    d = load()
    X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
    best = json.load(open(C.RESULTS / "best_hp.json"))
    Xt_all = X[:, thin_index(X.shape[1])]
    out_csv = C.RESULTS / "cv_main.csv"
    done = set()
    rows = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.trait, prev.rep, prev.fold))
    att_path = C.RESULTS / "attention.npz"
    att = dict(np.load(att_path)) if att_path.exists() else {}
    for rep in range(C.N_REPEATS):
        f = folds(group, rep)
        for k in range(C.N_FOLDS):
            te = np.where(f == k)[0]
            tr, va, cal = inner_split(np.where(f != k)[0], rep, k)
            fit = np.r_[tr, va]
            p_ref = X[fit].mean(0) / 2
            G = vanraden_G(X, p_ref)
            Xs = standardise_X(Xt_all, fit)
            for ti, t in enumerate(C.TRAITS):
                if (t, rep, k) in done:
                    continue
                t0 = time.time()
                mu, sd = Y[fit, ti].mean(), Y[fit, ti].std()
                y = (Y[:, ti] - mu) / sd
                scale = sd / Y[:, ti].std()          # standardised units -> panel phenotypic SD
                tbv, tbv_sd = TBV[:, ti], TBV[:, ti].std()
                base = dict(trait=t, rep=rep, fold=k)
                # ---- GBLUP
                m = GBLUP().fit(G[np.ix_(fit, fit)], y[fit])
                yc, _, sc = m.predict(G[np.ix_(cal, fit)], np.diag(G)[cal])
                yt, _, st = m.predict(G[np.ix_(te, fit)], np.diag(G)[te])
                evaluate_entry(rows, dict(base, model="GBLUP", source="PEV", h2_reml=1 / (1 + m.lam)),
                               y[cal], yc, sc, y[te], yt, st, tbv[te], scale, tbv_sd)
                # ---- deep models
                for arch in ("MLP", "CNN", "TRF"):
                    hp = best[arch]
                    nets, eps = [], []
                    for e in range(C.ENSEMBLE_SIZE):
                        net, vr, ep = D.train_net(arch, hp, Xs[tr], y[tr], Xs[va], y[va],
                                                  seed=int(1e5 * (rep + 1) + 1e3 * k + 10 * ti + e))
                        nets.append(net); eps.append(ep)
                    pc = np.stack([D.predict(n_, Xs[cal]) for n_ in nets])
                    pt = np.stack([D.predict(n_, Xs[te]) for n_ in nets])
                    evaluate_entry(rows, dict(base, model=arch, source="ENS", best_epoch=float(np.mean(eps))),
                                   y[cal], pc.mean(0), pc.std(0, ddof=1), y[te], pt.mean(0), pt.std(0, ddof=1),
                                   tbv[te], scale, tbv_sd)
                    mc_c = D.mc_dropout(nets[0], Xs[cal], C.MC_PASSES, seed=7 + k)
                    mc_t = D.mc_dropout(nets[0], Xs[te], C.MC_PASSES, seed=8 + k)
                    evaluate_entry(rows, dict(base, model=arch, source="MC", best_epoch=float(eps[0])),
                                   y[cal], mc_c[0], mc_c[1], y[te], mc_t[0], mc_t[1], tbv[te], scale, tbv_sd)
                    if arch == "TRF":
                        att[f"{t}_{rep}_{k}"] = nets[0].cls_attention(Xs[te]).mean(0)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                np.savez(att_path, **att)
                print(f"rep {rep} fold {k} {t} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if a.stage in ("tune", "all"):
        tune()
    if a.stage in ("cv", "all"):
        cv()

"""GBLUP predictions in the enlarged panel (same splits as run_revision.bigtp), saved per split, size and
trait for the FDR-controlled selection experiment.  Usage: CALGS_SEED=20261110 CALGS_SCALE=3 CALGS_TAG=_big python bigtp_gblup_preds.py"""
import numpy as np
import config as C
from models_stat import GBLUP, vanraden_G
from run_experiment import load
from run_revision import SIZES
d = load()
X, Y, TBV, group = d["X"].astype(float), d["Y"], d["TBV"], d["group"]
out = C.RESULTS / "preds_big"
out.mkdir(exist_ok=True)
for s in range(2):
    rng = np.random.default_rng([C.MASTER_SEED, 900, s])
    idx = rng.permutation(len(group))
    te, cal, va, pool = idx[:200], idx[200:350], idx[350:450], idx[450:]
    for n_tr in SIZES:
        fit = np.r_[pool[:n_tr], va]
        G = vanraden_G(X, X[fit].mean(0) / 2)
        for ti, t in enumerate(C.TRAITS):
            y = (Y[:, ti] - Y[fit, ti].mean()) / Y[fit, ti].std()
            m = GBLUP().fit(G[np.ix_(fit, fit)], y[fit])
            yc, _, sc = m.predict(G[np.ix_(cal, fit)], np.diag(G)[cal])
            yt, _, st = m.predict(G[np.ix_(te, fit)], np.diag(G)[te])
            np.savez_compressed(out / f"{t}_{s}_{n_tr}.npz", cal=cal, te=te, y=y, tbv=TBV[:, ti], y_fit=y[fit],
                                **{"GBLUP-PEV_yc": yc, "GBLUP-PEV_sc": sc, "GBLUP-PEV_yt": yt, "GBLUP-PEV_st": st})
        print("split", s, "n", n_tr, flush=True)

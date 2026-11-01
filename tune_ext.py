"""Expanded hyperparameter search (revision, Reviewer 2 point 6).
Eight further configurations per architecture, including deeper/wider transformers and
different patch sizes, evaluated exactly as in run_experiment.tune() (repeat-0 folds,
tuning traits SCC and DTI, validation Pearson r). Combined with the original four
configurations, the best of 12 is selected.  Usage: python tune_ext.py MLP CNN | TRF"""
import sys, json, time
import numpy as np, pandas as pd, torch
import config as C, models_dl as D
from run_experiment import load, folds, inner_split, thin_index, standardise_X
torch.set_num_threads(1)
EXT = {
 "MLP": [dict(hidden=1024, dropout=0.5, lr=5e-4, wd=1e-1), dict(hidden=512, dropout=0.3, lr=1e-3, wd=1e-1),
         dict(hidden=256, dropout=0.5, lr=3e-4, wd=5e-2), dict(hidden=128, dropout=0.6, lr=1e-3, wd=3e-1),
         dict(hidden=256, dropout=0.2, lr=1e-4, wd=1e-2), dict(hidden=64, dropout=0.5, lr=3e-3, wd=1e-1),
         dict(hidden=1024, dropout=0.7, lr=1e-4, wd=1e-1), dict(hidden=128, dropout=0.4, lr=5e-4, wd=1.0)],
 "CNN": [dict(channels=24, dropout=0.5, lr=5e-4, wd=1e-2), dict(channels=32, dropout=0.3, lr=1e-3, wd=1e-1),
         dict(channels=32, dropout=0.6, lr=3e-4, wd=5e-2), dict(channels=16, dropout=0.4, lr=3e-3, wd=1e-2),
         dict(channels=24, dropout=0.6, lr=3e-4, wd=1e-1), dict(channels=8, dropout=0.5, lr=5e-4, wd=1e-2),
         dict(channels=32, dropout=0.5, lr=1e-4, wd=1e-2), dict(channels=16, dropout=0.2, lr=5e-4, wd=3e-1)],
 "TRF": [dict(d=64, depth=3, dropout=0.2, lr=3e-4, wd=5e-2, patch=64), dict(d=128, depth=2, dropout=0.2, lr=3e-4, wd=5e-2, patch=64),
         dict(d=64, depth=4, dropout=0.1, lr=3e-4, wd=1e-2, patch=64), dict(d=64, depth=2, dropout=0.3, lr=5e-4, wd=1e-1, patch=32),
         dict(d=64, depth=2, dropout=0.2, lr=5e-4, wd=5e-2, patch=128), dict(d=128, depth=3, dropout=0.3, lr=1e-4, wd=1e-1, patch=64),
         dict(d=32, depth=4, dropout=0.1, lr=1e-3, wd=1e-2, patch=32), dict(d=128, depth=4, dropout=0.3, lr=1e-4, wd=5e-2, patch=128)],
}
if __name__ == "__main__":
    d = load(); X, Y, group = d["X"].astype(float), d["Y"], d["group"]
    Xt_all = X[:, thin_index(X.shape[1])]; f = folds(group, 0)
    for arch in sys.argv[1:]:
        out = C.RESULTS / f"tuning_ext_{arch}.csv"
        rows = pd.read_csv(out).to_dict("records") if out.exists() else []
        done = {(r["grid"], r["trait"], r["fold"]) for r in rows}
        for gi, hp in enumerate(EXT[arch], start=4):
            for t in C.TUNING_TRAITS:
                ti = C.TRAITS.index(t)
                for k in range(C.N_FOLDS):
                    if (gi, t, k) in done: continue
                    tr, va, _ = inner_split(np.where(f != k)[0], 0, k)
                    Xs = standardise_X(Xt_all, np.r_[tr, va])
                    y = (Y[:, ti] - Y[np.r_[tr, va], ti].mean()) / Y[np.r_[tr, va], ti].std()
                    t0 = time.time()
                    _, r, ep = D.train_net(arch, hp, Xs[tr], y[tr], Xs[va], y[va], seed=1000 + k)
                    rows.append(dict(arch=arch, grid=gi, trait=t, fold=k, val_r=r, best_epoch=ep, sec=time.time() - t0, **hp))
                    pd.DataFrame(rows).to_csv(out, index=False)
                    print(arch, gi, t, k, round(r, 3), ep, round(time.time() - t0), flush=True)

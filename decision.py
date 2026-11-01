"""Decision experiment on saved predictions (results<TAG>/preds/*.npz): advance lines whose phenotype is
predicted to exceed a threshold c, controlling the false discovery rate (FDR) at level q.

  conformal : conformal p-values for H0_i: y_i <= c (Jin and Candes, 2023),
              p_i = (1 + #{j in cal: y_j <= c and yhat_j > yhat_i}) / (n_cal + 1), then Benjamini-Hochberg
  native    : p_i = Phi((c - yhat_i) / sd_i) from the model's own uncertainty, then Benjamini-Hochberg
  plugin    : p_i = Phi((c - yhat_i) / sd_res), sd_res = RMS calibration residual, then Benjamini-Hochberg
  top10     : the 10% of candidates with the highest point prediction (no error control)
Metrics per test fold: share selected, false discovery proportion (FDP; 0 if nothing selected),
power (share of lines with y > c that were selected), and gain (mean TBV of selected minus fold mean,
in TBV SD of the panel)."""
import numpy as np
import pandas as pd
from scipy.stats import norm
import config as C

ENTRIES = ["GBLUP-PEV", "GBLUP-thin-PEV", "MLP-MC", "MLP-ENS", "CNN-MC", "CNN-ENS", "TRF-MC", "TRF-ENS"]
PCTS = {"q50": 0.50, "q75": 0.75, "q90": 0.90}
QS = [0.10, 0.20]


def bh(p, q):
    m = len(p)
    o = np.argsort(p)
    ok = np.where(p[o] <= q * np.arange(1, m + 1) / m)[0]
    sel = np.zeros(m, bool)
    if len(ok):
        sel[o[: ok.max() + 1]] = True
    return sel


def conformal_p(yc, ycal_obs, yt, c):
    null = yc[ycal_obs <= c]
    return (1 + (null[None, :] > yt[:, None]).sum(1)) / (len(yc) + 1)


def evaluate(path, panel, tbv_sd, extra=None):
    P = np.load(path)
    cal, te, y, tbv = P["cal"], P["te"], P["y"], P["tbv"]
    t, k = path.stem.split("_")[:2]
    rows = []
    for e in ENTRIES:
        if f"{e}_yt" not in P:
            continue
        yc, sc, yt, st = P[f"{e}_yc"], P[f"{e}_sc"], P[f"{e}_yt"], P[f"{e}_st"]
        sd_res = np.sqrt(np.mean((y[cal] - yc) ** 2))
        for cn, pc in PCTS.items():
            c = float(np.quantile(P["y_fit"], pc))
            elite = y[te] > c
            rules = {}
            for q in QS:
                rules[f"conformal_{q}"] = bh(conformal_p(yc, y[cal], yt, c), q)
                rules[f"native_{q}"] = bh(norm.cdf((c - yt) / np.maximum(st, 1e-12)), q)
                rules[f"plugin_{q}"] = bh(norm.cdf((c - yt) / sd_res), q)
            s = np.zeros(len(te), bool)
            s[np.argsort(-yt)[: int(np.ceil(0.10 * len(te)))]] = True
            rules["top10"] = s
            for rn, sel in rules.items():
                n_sel = int(sel.sum())
                rows.append(dict(panel=panel, trait=t, fold=int(k), entry=e, threshold=cn, rule=rn, **(extra or {}),
                                 frac_sel=n_sel / len(te), n_sel=n_sel,
                                 fdp=float((~elite[sel]).mean()) if n_sel else 0.0,
                                 power=float(sel[elite].mean()) if elite.any() else np.nan,
                                 gain=float((tbv[te][sel].mean() - tbv[te].mean()) / tbv_sd) if n_sel else np.nan))
    return rows


def run(tags):
    rows = []
    for i, tag in enumerate(tags):
        d = np.load(C.ROOT / f"data{tag}" / "panel.npz")
        for f in sorted((C.ROOT / f"results{tag}" / "preds").glob("*.npz")):
            t = f.stem.rsplit("_", 1)[0]
            rows += evaluate(f, i + 1, d["TBV"][:, C.TRAITS.index(t)].std())
    return pd.DataFrame(rows)


def run_big():
    d = np.load(C.ROOT / "data_big" / "panel.npz")
    rows = []
    for f in sorted((C.ROOT / "results_big" / "preds_big").glob("*.npz")):
        t, s, n = f.stem.split("_")
        rows += evaluate(f, 0, d["TBV"][:, C.TRAITS.index(t)].std(), extra=dict(n_train=int(n)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run(["", "_r1"])
    print(df.groupby(["threshold", "rule", "entry"])[["frac_sel", "fdp", "power", "gain"]].mean().round(3).to_string())

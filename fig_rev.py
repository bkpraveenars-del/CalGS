"""Figures 7 and 8 of the revised manuscript (revision experiments).
Usage: python fig_rev.py   (after analyse_rev.py)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from fig import OI, save, se

T = C.ROOT / "results" / "tables"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})
ENT = ["GBLUP-PEV", "MLP-MC", "MLP-ENS", "CNN-MC", "CNN-ENS", "TRF-MC", "TRF-ENS"]
ELAB = {"GBLUP-PEV": "GBLUP", "MLP-MC": "MLP MC", "MLP-ENS": "MLP ENS", "CNN-MC": "1D-CNN MC",
        "CNN-ENS": "1D-CNN ENS", "TRF-MC": "Transf. MC", "TRF-ENS": "Transf. ENS", "GBLUP-thin-PEV": "GBLUP-thin"}


def fig7():
    rv = pd.read_csv(T / "rev_S8_replicates_foldlevel.csv")
    rv["entry"] = rv.model + "-" + rv.source
    pm = rv.groupby(["panel", "entry"]).mean(numeric_only=True).reset_index()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    # a: phenotype coverage, native vs conformal, one point per panel
    ax = axs[0, 0]
    for k, (c, col, lab) in enumerate((("native", OI[5], "native"), ("plugin", OI[4], "Gaussian plug-in"),
                                       ("conformal", OI[2], "split conformal"))):
        for i, e in enumerate(ENT):
            v = pm[pm.entry == e][f"cov_{c}_90"].values
            ax.scatter(np.full(len(v), i + (k - 1) * 0.25), v, s=9, color=col, lw=0, label=lab if i == 0 else None)
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(ENT))); ax.set_xticklabels([ELAB[e] for e in ENT], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Coverage of phenotype (nominal 0.90)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="lower right"); ax.set_title("a  Five panels: phenotype coverage", loc="left")
    # b: TBV coverage
    ax = axs[0, 1]
    for k, (c, col, lab) in enumerate((("native", OI[5], "native"), ("pev_only", OI[0], "PEV only (GBLUP)"),
                                       ("conformal", OI[2], "split conformal"))):
        for i, e in enumerate(ENT):
            col_ = f"covg_{c}_90"
            v = pm[pm.entry == e][col_].dropna().values if col_ in pm else []
            if len(v):
                ax.scatter(np.full(len(v), i + (k - 1) * 0.25), v, s=9, color=col, lw=0, label=lab if lab and i == 0 else None)
                lab = None if i == 0 else lab
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(ENT))); ax.set_xticklabels([ELAB[e] for e in ENT], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Coverage of TBV (nominal 0.90)"); ax.set_ylim(0, 1.02)
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, fontsize=6, loc="lower right")
    ax.set_title("b  Five panels: breeding-value coverage", loc="left")
    # c: advancement with FDR control (threshold = fitting-set median, q = 0.20)
    ax = axs[1, 0]
    ds = pd.read_csv(T / "rev_fdr_selection.csv")
    ds = ds[ds.threshold == "q50"]
    mk = {"native_0.2": ("o", "native Gaussian + BH"), "plugin_0.2": ("^", "Gaussian plug-in + BH"),
          "conformal_0.2": ("s", "conformal p-values + BH"), "top10": ("x", "top 10% on point prediction")}
    for rn, (m_, lab) in mk.items():
        for _, r in ds[ds.rule == rn].iterrows():
            if r.entry == "GBLUP-thin-PEV":
                continue
            col = OI[0] if r.entry.startswith("GBLUP") else (OI[1] if "MLP" in r.entry else OI[2] if "CNN" in r.entry else OI[3])
            ax.scatter(r.frac_sel, r.fdr, marker=m_, s=22, color=col, lw=0.9 if m_ == "x" else 0)
        ax.scatter([], [], marker=m_, s=22, color="grey", label=lab)
    ax.axhline(0.20, color="k", ls=":", lw=0.8)
    for col, lab in ((OI[0], "GBLUP"), (OI[1], "MLP"), (OI[2], "1D-CNN"), (OI[3], "Transformer")):
        ax.scatter([], [], color=col, s=10, label=lab)
    ax.set_xlabel("Share of candidates advanced"); ax.set_ylabel("False discovery rate")
    ax.legend(fontsize=5.4, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax.set_title("c  Advance lines above the median, q = 0.20", loc="left")
    # d: accuracy across panels, GBLUP vs GBLUP-thin vs deep (ENS)
    ax = axs[1, 1]
    pe = ["GBLUP-PEV", "GBLUP-thin-PEV", "MLP-ENS", "CNN-ENS", "TRF-ENS"]
    pcol = [OI[0], OI[4], OI[1], OI[2], OI[3]]
    for i, e in enumerate(pe):
        v = pm[pm.entry == e].pa.values
        ax.scatter(np.full(len(v), i), v, s=10, color=pcol[i], lw=0)
        ax.errorbar(i + 0.25, v.mean(), yerr=se(v), fmt="o", color=pcol[i], ms=4, lw=0.8)
    ax.set_xticks(range(len(pe))); ax.set_xticklabels([ELAB[e] for e in pe], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Predictive ability (panel mean)"); ax.set_title("d  Accuracy across five panels", loc="left")
    fig.tight_layout()
    save(fig, "Figure7")


def fig8():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    bg = pd.read_csv(T / "rev_bigtp.csv")
    for m, col in (("GBLUP", OI[0]), ("MLP-ENS", OI[1]), ("MLP-MC", OI[3])):
        x = bg[bg.model == m].sort_values("n_train")
        if m != "MLP-MC":
            axs[0].plot(x.n_train, x.pa, "-o", color=col, ms=3, label=m.replace("-ENS", ""))
        axs[1].plot(x.n_train, x.cov_native, "--o", color=col, ms=3, mfc="none", label=f"{m} native")
        axs[1].plot(x.n_train, x.cov_conf, "-o", color=col, ms=3, label=f"{m} conformal")
    axs[0].set_xlabel("Training lines"); axs[0].set_ylabel("Predictive ability"); axs[0].legend(fontsize=6)
    axs[0].set_title("a  Accuracy, 100–800 lines", loc="left")
    axs[1].axhline(0.9, color="k", ls=":", lw=0.8); axs[1].set_ylim(0, 1.02)
    axs[1].set_xlabel("Training lines"); axs[1].set_ylabel("Coverage (nominal 0.90)"); axs[1].legend(fontsize=5.2, ncol=1)
    axs[1].set_title("b  Coverage vs size", loc="left")
    lo = pd.read_csv(T / "rev_loso.csv")
    ents = ["GBLUP-PEV", "MLP-MC", "MLP-ENS"]
    for k, (rg, col) in enumerate((("training-pop calibration", OI[5]), ("target-pop calibration", OI[2]))):
        x = lo[lo.regime == rg].set_index("entry").loc[ents]
        axs[2].errorbar(np.arange(3) + (k - 0.5) * 0.3, x.cov_conf, yerr=x.cov_conf_se, fmt="o", color=col, ms=5, lw=0.9,
                        capsize=2, label=rg.replace(" calibration", "").replace("-pop", " subpopulations" if k == 0 else " subpopulation"))
    axs[2].axhline(0.9, color="k", ls=":", lw=0.8); axs[2].set_ylim(0.80, 1.0); axs[2].set_xlim(-0.6, 2.6)
    axs[2].set_xticks(range(3)); axs[2].set_xticklabels([ELAB[e] for e in ents], fontsize=6.5)
    axs[2].set_ylabel("Conformal coverage, unseen subpop."); axs[2].legend(fontsize=6, title="calibrated on", title_fontsize=6, loc="upper center", ncol=1)
    axs[2].set_title("c  Leave one subpopulation out", loc="left")
    fig.tight_layout()
    save(fig, "Figure8")


if __name__ == "__main__":
    fig7()
    fig8()

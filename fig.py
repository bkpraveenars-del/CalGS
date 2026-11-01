"""Draw Figures 1-6 from the raw outputs (600 dpi PNG + PDF in figures/).
Usage:  python fig.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import config as C

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False})
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MODELS = ["GBLUP", "MLP", "CNN", "TRF"]
LABEL = {"GBLUP": "GBLUP", "MLP": "MLP", "CNN": "1D-CNN", "TRF": "Transformer"}
MCOL = dict(zip(MODELS, OI[:4]))
CONS = ["native", "plugin", "epi_ale", "conformal", "adaptive"]
CLAB = {"native": "model's own", "plugin": "Gaussian plug-in", "epi_ale": "epistemic + aleatoric",
        "conformal": "split conformal", "adaptive": "adaptive conformal"}
CCOL = dict(zip(CONS, [OI[5], OI[4], OI[3], OI[2], OI[0]]))
SHORT = {"SYPP": "Yield", "OIL": "Oil", "PH": "Height", "SCC": "Coat\ncolour", "DTI": "Drought\nindex"}


def save(fig, name):
    fig.savefig(C.FIGURES / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(C.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def fig1(d, N):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, txt, fc, bold=False, fs=6.6, color="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, c="#666666"):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))

    heads = [(1, "Simulated panel", "#56B4E9"), (26, "Prediction models", "#0072B2"),
             (51, "Uncertainty + calibration", "#009E73"), (77, "Selection", "#E69F00")]
    for x, t, c in heads:
        box(x, 54, 22, 4, t, c, bold=True, color="white", fs=7)
    h2 = [C.TRAIT_SPEC[t]["h2"] for t in C.TRAITS]
    box(1, 43, 22, 8, f"{d['X'].shape[0]} inbred accessions\n{d['X'].shape[1]:,} SNPs, 13 chromosomes\nfounder-mosaic relatedness", "#DDEEF8")
    box(1, 32, 22, 8, f"5 traits, line-mean h² = {min(h2):.2f}–{max(h2):.2f}\nadditive + epistasis\n+ pleiotropy + G×E (6 env.)", "#DDEEF8")
    box(1, 21, 22, 7, "TRUE breeding values\nknown by construction", "#B7DDF4", bold=True)
    box(26, 44, 22, 6, "GBLUP (all SNPs; 4,096-SNP\ncontrol) PEV + σ²e interval", "#D2E4EF", fs=6.2)
    for i, m in enumerate(["MLP", "1D-CNN", "Genomic transformer"]):
        box(26, 35 - i * 7, 22, 5, m, "#D2E4EF")
    ax.text(30, 17.6, f"deep models: {C.N_THIN:,} uniformly thinned SNPs; MC dropout (25 passes); deep ensemble ({C.ENSEMBLE_SIZE} nets)",
            ha="center", va="center", fontsize=6.5, style="italic", color="#444444")
    box(51, 44, 22, 6, "Native intervals\n(assume the model is correct)", "#F6E0D3")
    box(51, 31, 22, 9, "SPLIT-CONFORMAL LAYER\n20% of training held out\n"
        "q̂ = ⌈(n+1)(1−α)⌉-th |residual|", "#BDE5DA")
    box(51, 20, 22, 7, "coverage ≥ 1−α in finite samples,\nany model, no distributional\nassumption",
        "#86D1BA", bold=True, fs=5.8)
    box(77, 44, 22, 6, "Rank on point GEBV\n(standard practice)", "#E3E3E3")
    box(77, 35, 22, 6, "Rank on LOWER\nconformal bound (risk-averse)", "#F5D58F", bold=True, fs=6.2)
    box(77, 26, 22, 6, "Three-way TRIAGE: advance /\ndiscard / re-phenotype", "#F5D58F", bold=True, fs=6.2)
    box(77, 18.5, 22, 4.5, "Gain, top lines lost vs\nthe ORACLE (TBV)", "#FBEBCF", fs=6.2)
    arrow(24.2, 47, 25.6, 47); arrow(24.2, 36, 25.6, 37.5); arrow(24.2, 36, 25.6, 30.5); arrow(24.2, 36, 25.6, 23.5)
    arrow(48.6, 47, 50.4, 47); arrow(48.6, 37.5, 50.4, 36); arrow(48.6, 30.5, 50.4, 35); arrow(48.6, 23.5, 50.4, 34)
    arrow(62, 43.5, 62, 40.5, "#009E73"); arrow(62, 30.5, 62, 27.5, "#009E73")
    arrow(73.6, 24, 76.4, 37, "#E69F00"); arrow(73.6, 24, 76.4, 29, "#E69F00"); arrow(73.6, 47, 76.4, 47)
    arrow(88, 25.5, 88, 23.4, "#E69F00")
    ax.text(62, 15.4, "robustness: 5 independent panels; TBV coverage; unseen subpopulation (LOSO) and environment (LOEO); 80 to 800 training lines",
            ha="center", va="center", fontsize=6.0, style="italic", color="#444444")
    ax.add_patch(FancyBboxPatch((3, 1), 94, 12, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#F4F4F4", ec="#CCCCCC"))
    q = ["The four questions",
         "i.  How accurate are the models, at 80 to 800 training lines, relative to the heritability ceiling?",
         "ii.  Are their native uncertainty statements calibrated, for the phenotype and for the breeding value?",
         "iii.  Does split-conformal calibration restore coverage, at what width, and under subpopulation or environment shift?",
         "iv.  Can calibrated intervals improve a selection decision?"]
    for i, s_ in enumerate(q):
        ax.text(50, 11.2 - i * 2.4, s_, ha="center", va="center", fontsize=7.4 if i else 8.6, fontweight="bold" if i == 0 else "normal")
    save(fig, "Figure1")


def fig2(d, N):
    X = d["X"].astype(float)
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=1.2)
    # a LD decay chromosome 1
    ax = fig.add_subplot(gs[0, :2])
    c1 = d["chrom"] == 1
    X1, p1 = X[:, c1], d["pos"][c1]
    R = np.corrcoef(X1.T) ** 2
    iu = np.triu_indices(len(p1), 1)
    dist, r2 = np.abs(p1[iu[0]] - p1[iu[1]]), R[iu]
    edges = np.logspace(3, 7, 25)
    mid, mean, q1, q3 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() > 20:
            mid.append(np.sqrt(lo * hi)); mean.append(r2[m].mean())
            q1.append(np.quantile(r2[m], .25)); q3.append(np.quantile(r2[m], .75))
    mid, mean = np.array(mid), np.array(mean)
    ax.fill_between(mid / 1e3, q1, q3, color=OI[0], alpha=0.2, lw=0)
    ax.plot(mid / 1e3, mean, "o-", color=OI[0], ms=2.5, lw=1)
    half = N["ld_half_kb"]
    ax.axvline(half, ls="--", color="grey", lw=0.8)
    ax.text(half * 1.2, max(mean) * 0.9, f"half-decay\n≈{half:.0f} kb", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("Physical distance (kb)"); ax.set_ylabel("r²")
    ax.set_title("a  LD decay (chr 1)", loc="left")
    # b PCA
    ax = fig.add_subplot(gs[0, 2:4])
    Z = X - X.mean(0)
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    ve = s ** 2 / np.sum(s ** 2) * 100
    for g in range(3):
        m = d["group"] == g
        ax.scatter(u[m, 0] * s[0], u[m, 1] * s[1], s=5, color=OI[g], label=f"cluster {g+1} (n={m.sum()})", alpha=0.8, lw=0)
    ax.set_xlabel(f"PC1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ve[1]:.1f}%)")
    ax.legend(fontsize=6, markerscale=2); ax.set_title("b  Structure", loc="left")
    # c markers per chromosome
    ax = fig.add_subplot(gs[0, 4:])
    cnt = np.bincount(d["chrom"])[1:]
    ax.bar(np.arange(1, 14), cnt, color=OI[4])
    ax.set_xlabel("Chromosome"); ax.set_ylabel("SNPs after QC"); ax.set_xticks(range(1, 14, 2))
    ax.set_title(f"c  Markers ({d['X'].shape[1]:,})", loc="left")
    # d phenotypes E1
    ax = fig.add_subplot(gs[1, :3])
    vals = [(d[f"{t}_obs"][:, 0] - d[f"{t}_obs"][:, 0].mean()) / d[f"{t}_obs"][:, 0].std() for t in C.TRAITS]
    vp = ax.violinplot(vals, showmedians=True)
    for b, c in zip(vp["bodies"], OI):
        b.set_facecolor(c); b.set_alpha(0.5)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([SHORT[t] for t in C.TRAITS], fontsize=6.5)
    for i, t in enumerate(C.TRAITS):
        ax.text(i + 1, 7.0, f"h²={C.TRAIT_SPEC[t]['h2']:.2f}", ha="center", fontsize=6)
    ax.set_ylim(top=7.8)
    ax.set_ylabel("Standardised phenotype (E1)"); ax.set_title("d  Phenotypes, environment E1", loc="left")
    # e genetic correlations
    ax = fig.add_subplot(gs[1, 3:])
    rg = np.corrcoef(d["TBV"].T)
    im = ax.imshow(rg, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{rg[i,j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(C.TRAITS, fontsize=6.5); ax.set_yticklabels(C.TRAITS, fontsize=6.5)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("e  Genetic correlations (TBV)", loc="left")
    save(fig, "Figure2")


def fig3(cv, N):
    pt = cv[(cv.model == "GBLUP") | (cv.source == "ENS")]
    h2 = N["realised_h2"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    x = np.arange(5); w = 0.2
    for k, m in enumerate(MODELS):
        mu = [pt[(pt.trait == t) & (pt.model == m)].pa.mean() for t in C.TRAITS]
        er = [se(pt[(pt.trait == t) & (pt.model == m)].pa) for t in C.TRAITS]
        axs[0].bar(x + (k - 1.5) * w, mu, w, yerr=er, color=MCOL[m], label=LABEL[m], error_kw=dict(lw=0.6))
        axs[1].bar(x + (k - 1.5) * w, np.array(mu) / np.sqrt([h2[t] for t in C.TRAITS]), w, color=MCOL[m])
    for i, t in enumerate(C.TRAITS):
        axs[0].plot([i - 0.45, i + 0.45], [np.sqrt(h2[t])] * 2, "k--", lw=0.8)
    axs[1].axhline(1, color="k", ls="--", lw=0.8)
    for a, ttl, yl in ((axs[0], "a  Predictive ability", "r (prediction, phenotype)"),
                       (axs[1], "b  Fraction of ceiling", "r / h")):
        a.set_xticks(x); a.set_xticklabels([SHORT[t] for t in C.TRAITS], fontsize=6); a.set_title(ttl, loc="left"); a.set_ylabel(yl)
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    for m in MODELS:
        s = pt[pt.model == m]
        axs[2].scatter(s.pa, s.ta, s=6, color=MCOL[m], alpha=0.7, lw=0)
    lim = [min(pt.pa.min(), pt.ta.min()) - 0.05, max(pt.pa.max(), pt.ta.max()) + 0.05]
    axs[2].plot(lim, lim, "k:", lw=0.8)
    axs[2].set_xlabel("Predictive ability"); axs[2].set_ylabel("True accuracy"); axs[2].set_title("c  What field data cannot see", loc="left")
    fig.tight_layout()
    save(fig, "Figure3")


def fig4(cv, N):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    lev = [80, 90, 95]
    ax = axs[0, 0]
    for c in CONS:
        ax.plot([.8, .9, .95], [cv[f"cov_{c}_{L}"].mean() for L in lev], "o-", color=CCOL[c], label=CLAB[c], ms=3)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlim(0.75, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage"); ax.legend(fontsize=6)
    ax.set_title("a  Calibration, all models and traits", loc="left")
    ax = axs[0, 1]
    cv = cv.assign(entry=cv.model + "-" + cv.source)
    order = ["GBLUP-PEV", "MLP-MC", "MLP-ENS", "CNN-MC", "CNN-ENS", "TRF-MC", "TRF-ENS"]
    y = np.arange(len(order))
    nat = [cv[cv.entry == e].cov_native_90.mean() - 0.9 for e in order]
    con = [cv[cv.entry == e].cov_conformal_90.mean() - 0.9 for e in order]
    ax.barh(y + 0.2, nat, 0.4, color=CCOL["native"], label="model's own")
    ax.barh(y - 0.2, con, 0.4, color=CCOL["conformal"], label="split conformal")
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([e.replace("CNN", "1D-CNN").replace("TRF", "Transformer") for e in order], fontsize=6)
    ax.set_xlabel("Coverage error at nominal 90%"); ax.legend(fontsize=6); ax.set_title("b  Per model", loc="left")
    ax = axs[1, 0]
    x = np.arange(5)
    for k, c in enumerate(["native", "plugin", "conformal", "adaptive"]):
        ax.bar(x + (k - 1.5) * 0.2, [cv[cv.trait == t][f"cov_{c}_90"].mean() for t in C.TRAITS], 0.2, color=CCOL[c], label=CLAB[c])
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in C.TRAITS], fontsize=6)
    ax.set_ylabel("Empirical coverage (nominal 0.90)"); ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.legend(fontsize=6, ncol=2, loc="upper center"); ax.set_title("c  By trait", loc="left")
    ax = axs[1, 1]
    for i, t in enumerate(C.TRAITS):
        s = cv[(cv.trait == t) & (cv.model != "GBLUP")]
        ax.plot([s.width_native_90.mean(), s.width_conformal_90.mean()], [s.cov_native_90.mean(), s.cov_conformal_90.mean()],
                "o-", color=OI[i], label=t, ms=3)
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_xlabel("Mean 90% interval width (phenotypic SD)")
    ax.set_ylabel("Empirical coverage"); ax.legend(fontsize=6, title="deep models,\nnative → conformal", title_fontsize=6)
    ax.set_title("d  The price of honest intervals", loc="left")
    fig.tight_layout()
    save(fig, "Figure4")


def fig5(cv, N):
    pt = cv[(cv.model == "GBLUP") | (cv.source == "ENS")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    x = np.arange(5)
    for k, m in enumerate(MODELS):
        s = [pt[(pt.trait == t) & (pt.model == m)].gain_point for t in C.TRAITS]
        axs[0].bar(x + (k - 1.5) * 0.2, [v.mean() for v in s], 0.2, yerr=[se(v) for v in s], color=MCOL[m], label=LABEL[m], error_kw=dict(lw=0.6))
    for i, t in enumerate(C.TRAITS):
        axs[0].plot([i - 0.45, i + 0.45], [pt[pt.trait == t].oracle.mean()] * 2, "k--", lw=0.8)
    axs[0].set_xticks(x); axs[0].set_xticklabels([SHORT[t] for t in C.TRAITS], fontsize=6)
    axs[0].set_ylabel("Gain (TBV SD)"); axs[0].legend(fontsize=6, ncol=2); axs[0].set_title("a  Gain at 10% selection", loc="left")
    for m in MODELS:
        s = pt[pt.model == m]
        axs[1].scatter(s.gain_point, s.gain_lb, s=6, color=MCOL[m], alpha=0.7, lw=0)
    lim = [min(pt.gain_point.min(), pt.gain_lb.min()), max(pt.gain_point.max(), pt.gain_lb.max())]
    axs[1].plot(lim, lim, "k:", lw=0.8); axs[1].set_xlabel("Gain, point ranking"); axs[1].set_ylabel("Gain, lower-bound ranking")
    axs[1].set_title("b  Per fold", loc="left")
    dd = [pt[pt.model == m].gain_lb.values - pt[pt.model == m].gain_point.values for m in MODELS]
    axs[2].barh(np.arange(4), [v.mean() for v in dd], xerr=[se(v) for v in dd], color=[MCOL[m] for m in MODELS])
    axs[2].axvline(0, color="k", lw=0.8); axs[2].set_yticks(range(4)); axs[2].set_yticklabels([LABEL[m] for m in MODELS], fontsize=6.5)
    axs[2].set_xlabel("Δ gain (lower bound − point)"); axs[2].set_title("c  Effect of the rule", loc="left")
    fig.tight_layout()
    save(fig, "Figure5")


def fig6(N):
    att = N["attention"]
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 2, hspace=0.85, wspace=0.35)
    for r, t in enumerate(["SCC", "DTI"]):
        A, dens = np.load(C.RESULTS / f"attention_profile_{t}.npy")
        ax = fig.add_subplot(gs[r, :])
        z = (A - A.mean()) / A.std()
        ax.bar(np.arange(len(z)), z, color=OI[0], alpha=0.6, width=0.9)
        ax2 = ax.twinx(); ax2.plot(np.arange(len(dens)), dens, color=OI[5], lw=1)
        ax2.set_ylabel("true QTL per patch", color=OI[5], fontsize=7); ax2.spines["right"].set_visible(True)
        ax.set_ylabel("attention (z)"); ax.set_xlabel("genomic patch (64 SNPs)")
        a = att[t]
        ax.set_title(f"{'ab'[r]}  {C.TRAIT_NAMES[t]}: Spearman ρ = {a['spearman_rho']:+.3f}, P = {a['p']:.2f}", loc="left")
    ax = fig.add_subplot(gs[2, 0])
    rho = [att[t]["spearman_rho"] for t in C.TRAITS]
    sig = [att[t]["p"] < 0.05 for t in C.TRAITS]
    ax.barh(range(5), rho, color=[OI[0] if s_ else "white" for s_ in sig], edgecolor=OI[0])
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(5)); ax.set_yticklabels(C.TRAITS)
    ax.set_xlabel("Spearman ρ (attention, QTL density)"); ax.set_title("c  All traits (filled: P < 0.05)", loc="left")
    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(5)
    ax.bar(x - 0.2, [att[t]["qtl_top_quintile"] for t in C.TRAITS], 0.4, color=OI[1], label="top attention quintile")
    ax.bar(x + 0.2, [att[t]["qtl_other"] for t in C.TRAITS], 0.4, color="#BBBBBB", label="other patches")
    ax.set_xticks(x); ax.set_xticklabels(C.TRAITS); ax.set_ylabel("mean true QTL per patch"); ax.legend(fontsize=6)
    ax.set_title("d  QTL enrichment", loc="left")
    save(fig, "Figure6")


def main():
    d = np.load(C.DATA / "panel.npz")
    N = json.load(open(C.RESULTS / "numbers.json"))
    cv = pd.read_csv(C.RESULTS / "cv_main.csv")
    fig1(d, N); fig2(d, N); fig3(cv, N); fig4(cv, N); fig5(cv, N); fig6(N)


if __name__ == "__main__":
    main()

"""Simulate a sesame-like diversity panel: genome, founder-mosaic accessions,
quality control, five traits with pleiotropy, epistasis and GxE in six
environments. Writes data/panel.npz and results/panel_summary.json.

Usage:  python simulate_genome.py
"""
import json
import numpy as np
from scipy.stats import norm
import config as C


def rng_for(tag):
    """Independent, reproducible random stream per named step."""
    tag_int = sum(ord(c) * (i + 1) for i, c in enumerate(tag))
    return np.random.default_rng(np.random.SeedSequence([C.MASTER_SEED, tag_int]))


def simulate_markers():
    rng = rng_for("markers")
    chrom, pos = [], []
    for c, L in enumerate(C.CHR_LEN_MB):
        m = int(round(L * C.MARKERS_PER_MB))
        p = np.sort(rng.uniform(0, L * 1e6, m))
        chrom.append(np.full(m, c + 1)); pos.append(p)
    return np.concatenate(chrom), np.concatenate(pos)


def simulate_founders(chrom, pos):
    """Founder haplotypes: thresholded latent AR(1) process per founder and chromosome."""
    rng = rng_for("founders")
    M = len(pos)
    p_anc = rng.uniform(0.05, 0.95, M)
    a = p_anc * (1 - C.FST) / C.FST
    b = (1 - p_anc) * (1 - C.FST) / C.FST
    p_grp = np.clip(np.vstack([rng.beta(a, b) for _ in C.GROUP_SIZES]), 1e-3, 1 - 1e-3)   # (3, M)
    f_group = np.repeat(np.arange(3), C.FOUNDERS_PER_GROUP)                                  # (46,)
    thr = norm.ppf(p_grp[f_group])                                                           # (46, M)
    H = np.zeros((C.N_FOUNDERS, M), dtype=np.int8)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        d_kb = np.diff(pos[idx]) / 1e3
        rho = 0.5 ** (d_kb / C.LD_HALF_LIFE_KB)
        z = rng.standard_normal(C.N_FOUNDERS)
        H[:, idx[0]] = z < thr[:, idx[0]]
        for j in range(1, len(idx)):
            z = rho[j - 1] * z + np.sqrt(1 - rho[j - 1] ** 2) * rng.standard_normal(C.N_FOUNDERS)
            H[:, idx[j]] = z < thr[:, idx[j]]
    return H, f_group, p_anc


def simulate_accessions(H, f_group, chrom, pos):
    """Accessions as recombinant mosaics of founders; fully homozygous, coded 0/2."""
    rng = rng_for("mosaic")
    group = np.repeat(np.arange(3), C.GROUP_SIZES)
    n, M = len(group), H.shape[1]
    X = np.zeros((n, M), dtype=np.int8)
    for i in range(n):
        w = np.where(f_group == group[i], C.W_OWN, C.W_OTHER)
        w = w / w.sum()
        for c in np.unique(chrom):
            idx = np.where(chrom == c)[0]
            L = C.CHR_LEN_MB[c - 1] * 1e6
            start = 0.0
            while start < L:
                end = start + rng.exponential(C.SEGMENT_MEAN_MB * 1e6)
                f = rng.choice(C.N_FOUNDERS, p=w)
                sel = idx[(pos[idx] >= start) & (pos[idx] < end)]
                X[i, sel] = 2 * H[f, sel]
                start = end
    return X, group


def quality_control(X):
    rng = rng_for("qc")
    Xf = X.astype(float)
    miss = rng.random(X.shape) < C.MISSING_RATE
    Xf[miss] = np.nan
    call = 1 - np.isnan(Xf).mean(0)
    keep = call >= C.MIN_CALL_RATE
    mean = np.nanmean(Xf, 0)
    Xi = np.where(np.isnan(Xf), mean, Xf)
    p = Xi.mean(0) / 2
    maf = np.minimum(p, 1 - p)
    keep &= (maf >= C.MIN_MAF) & (Xi.std(0) > 0)
    return Xi[:, keep], keep


def std_cols(W):
    s = W.std(0); s[s == 0] = 1
    return (W - W.mean(0)) / s


def standardise(v):
    return (v - v.mean()) / v.std()


def rank3_loadings():
    k = len(C.TRAITS)
    R = np.eye(k)
    for (a, b), r in C.TARGET_RG.items():
        i, j = C.TRAITS.index(a), C.TRAITS.index(b)
        R[i, j] = R[j, i] = r
    val, vec = np.linalg.eigh(R)
    top = np.argsort(val)[::-1][: C.N_SHARED_FACTORS]
    L = vec[:, top] * np.sqrt(val[top])
    norms = np.linalg.norm(L, axis=1)
    L[norms > 0.95] *= (0.95 / norms[norms > 0.95])[:, None]
    return R, L


def simulate_traits(Xtrue):
    """Xtrue: true (pre-QC, error-free) genotypes, 0/2."""
    rng = rng_for("traits")
    n, M = Xtrue.shape
    Wall = None
    segregating = np.where(Xtrue.std(0) > 0)[0]
    R_target, L = rank3_loadings()
    # shared pleiotropic factors
    F = np.zeros((n, C.N_SHARED_FACTORS))
    factor_loci = []
    for f in range(C.N_SHARED_FACTORS):
        loci = rng.choice(segregating, C.LOCI_PER_FACTOR, replace=False)
        factor_loci.append(loci)
        F[:, f] = standardise(std_cols(Xtrue[:, loci].astype(float)) @ rng.standard_normal(C.LOCI_PER_FACTOR))
    out = {}
    for t_i, t in enumerate(C.TRAITS):
        s = C.TRAIT_SPEC[t]
        q = rng.choice(segregating, s["n_qtl"], replace=False)
        Wq = std_cols(Xtrue[:, q].astype(float))
        if s["dist"] == "gamma":
            eff = rng.gamma(C.GAMMA_SHAPE, 1.0, s["n_qtl"]) * rng.choice([-1, 1], s["n_qtl"])
        elif s["dist"] == "normal":
            eff = rng.standard_normal(s["n_qtl"])
        else:  # oligogenic: five inflated effects
            eff = rng.standard_normal(s["n_qtl"])
            eff[:5] *= 6.0
        A_own = standardise(Wq @ eff)
        l = L[t_i]
        A = standardise(F @ l + np.sqrt(max(1 - l @ l, 0)) * A_own)
        pairs = rng.choice(s["n_qtl"], (s["n_epi"], 2), replace=True)
        AA = standardise((Wq[:, pairs[:, 0]] * Wq[:, pairs[:, 1]]) @ rng.standard_normal(s["n_epi"]))
        e = s["epi"]
        G = np.sqrt(1 - e) * A + np.sqrt(e) * AA
        tbv = np.sqrt(1 - e) * A                       # additive (transmissible) value, latent scale
        gxe_loci = rng.choice(s["n_qtl"], s["n_qtl"] // 4, replace=False)
        GE = np.column_stack([standardise(Wq[:, gxe_loci] @ rng.standard_normal(len(gxe_loci)))
                              for _ in C.ENVS]) * np.sqrt(s["gxe"] * G.var())
        # residual variance for the target line-mean heritability over 6 environments
        nE = len(C.ENVS)
        sig2e = nE * (tbv.var() / s["h2"] - G.var() - s["gxe"] * G.var() / nE)
        mu = np.array([(C.DROUGHT_SHIFT[w] if t in C.STRESS_TRAITS else 0.0) + C.YEAR_SHIFT[y] for w, y in C.ENVS])
        lat = mu[None, :] + G[:, None] + GE + rng.normal(0, np.sqrt(sig2e), (n, nE))
        obs = link(lat, s["link"], t)
        out[t] = dict(qtl=q, eff=eff, pairs=q[pairs], gxe_loci=q[gxe_loci], A=A, G=G, tbv=tbv,
                      latent=lat, obs=obs, sig2e=sig2e)
    return out, R_target, factor_loci


def link(lat, kind, t):
    if kind == "lognormal":
        return 9.0 * np.exp(0.22 * lat)
    if kind == "logistic":
        return 0.12 + 0.83 / (1 + np.exp(-0.9 * lat))
    scale = {"OIL": (50.0, 2.4), "PH": (135.0, 17.0), "SCC": (45.0, 10.5)}[t]
    return scale[0] + scale[1] * lat


def main():
    chrom, pos = simulate_markers()
    H, f_group, p_anc = simulate_founders(chrom, pos)
    Xtrue, group = simulate_accessions(H, f_group, chrom, pos)
    X, keep = quality_control(Xtrue)
    traits, R_target, factor_loci = simulate_traits(Xtrue)
    Y = np.column_stack([traits[t]["obs"].mean(1) for t in C.TRAITS])      # line means over 6 envs
    TBV = np.column_stack([traits[t]["tbv"] for t in C.TRAITS])
    save = dict(X=X.astype(np.int8), chrom=chrom[keep], pos=pos[keep], group=group, Y=Y, TBV=TBV,
                chrom_all=chrom, pos_all=pos, keep=keep, R_target=R_target)
    for t in C.TRAITS:
        for k in ("qtl", "pairs", "gxe_loci", "obs", "latent", "G"):
            save[f"{t}_{k}"] = traits[t][k]
    np.savez_compressed(C.DATA / "panel.npz", **save)
    summ = dict(n_accessions=int(X.shape[0]), n_markers_simulated=int(len(pos)), n_markers_qc=int(X.shape[1]),
                genome_mb=float(sum(C.CHR_LEN_MB)), mean_spacing_kb=float(sum(C.CHR_LEN_MB) * 1e3 / X.shape[1]),
                group_sizes=C.GROUP_SIZES,
                realised_h2_linemean={t: float(np.corrcoef(TBV[:, i], Y[:, i])[0, 1] ** 2) for i, t in enumerate(C.TRAITS)})
    json.dump(summ, open(C.RESULTS / "panel_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

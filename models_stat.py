"""GBLUP with REML (EMMA-style spectral likelihood), prediction error variance,
and the conformal / Gaussian interval constructions shared by all models."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def vanraden_G(X, p=None):
    """VanRaden (2008) method 1 for 0/2-coded genotypes; p from the reference set if given."""
    if p is None:
        p = X.mean(0) / 2
    Z = X - 2 * p
    return Z @ Z.T / (2 * np.sum(p * (1 - p)))


class GBLUP:
    """y = 1*mu + u + e,  u ~ N(0, G s2g),  e ~ N(0, I s2e); lambda = s2e/s2g by REML."""

    def fit(self, G_ff, y):
        n = len(y)
        self.G_ff, self.y = G_ff, y
        one = np.ones(n)
        S = np.eye(n) - np.outer(one, one) / n
        xi, U = np.linalg.eigh(S @ G_ff @ S)
        xi, U = xi[1:], U[:, 1:]                       # drop the null direction of the intercept
        eta = U.T @ y
        dof = n - 1

        def neg_reml(log_d):
            d = np.exp(log_d)
            return 0.5 * dof * np.log(np.sum(eta ** 2 / (xi + d))) + 0.5 * np.sum(np.log(xi + d))

        res = minimize_scalar(neg_reml, bounds=(np.log(1e-4), np.log(1e4)), method="bounded")
        self.lam = float(np.exp(res.x))
        self.s2g = float(np.sum(eta ** 2 / (xi + self.lam)) / dof)
        self.s2e = self.lam * self.s2g
        H = G_ff + self.lam * np.eye(n)
        self.Hinv = np.linalg.inv(H)
        self.h1 = self.Hinv @ one
        self.c = float(one @ self.h1)
        self.mu = float(self.h1 @ y / self.c)
        self.alpha = self.Hinv @ (y - self.mu)
        return self

    def predict(self, G_nf, g_nn_diag):
        """Point prediction, PEV of u, and predictive SD of a new phenotype (PEV + s2e)."""
        yhat = self.mu + G_nf @ self.alpha
        GH = G_nf @ self.Hinv
        pev = self.s2g * (g_nn_diag - np.sum(GH * G_nf, 1) + (1 - G_nf @ self.h1) ** 2 / self.c)
        pred_sd = np.sqrt(np.maximum(pev, 0) + self.s2e)
        return yhat, pev, pred_sd


# ---------------------------------------------------------------- intervals
def conformal_q(scores, level):
    """Split-conformal quantile: the ceil((n+1)*level)-th smallest score."""
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def intervals(yhat_cal, y_cal, sd_cal, yhat_te, sd_te, level):
    """Five constructions; each returns (lower, upper) for the test set.

    native      : yhat +- z * sd                  (model's own uncertainty)
    plugin      : yhat +- z * sd_res               (homoscedastic Gaussian, calibration residual SD)
    epi_ale     : yhat +- z * sqrt(sd^2 + sd_res^2)
    conformal   : yhat +- q  (absolute residual scores)
    adaptive    : yhat +- q * sd  (normalised scores |y - yhat| / sd)
    """
    z = norm.ppf(0.5 + level / 2)
    res = y_cal - yhat_cal
    sd_res = float(np.sqrt(np.mean(res ** 2)))
    q = conformal_q(np.abs(res), level)
    qa = conformal_q(np.abs(res) / sd_cal, level)
    half = {
        "native": z * sd_te,
        "plugin": np.full_like(yhat_te, z * sd_res),
        "epi_ale": z * np.sqrt(sd_te ** 2 + sd_res ** 2),
        "conformal": np.full_like(yhat_te, q),
        "adaptive": qa * sd_te,
    }
    return {k: (yhat_te - h, yhat_te + h) for k, h in half.items()}, qa

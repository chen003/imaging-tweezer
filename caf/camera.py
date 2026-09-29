"""Photon-count histograms and threshold detection fidelity."""
import numpy as np
from scipy import stats

from . import constants as k


def background_mean(s_tot, t_us, eta, zeta=1e-3, dark_per_ms=0.01):
    """Stray imaging light (fraction zeta of a two-level scatterer's collected light) + dark counts."""
    stray = zeta * eta * 0.5 * k.GAMMA * s_tot * t_us * 1e-6
    return stray + dark_per_ms * t_us * 1e-3


def bright_pmf(n606, eta, bg, n_max=None):
    """Mixture over molecules of Binomial(N_i, eta) convolved with Poisson(bg)."""
    n606 = np.asarray(n606)
    n_max = int(max(10, np.max(n606) * eta * 1.5 + 10 * np.sqrt(bg + 1) + 20)) if n_max is None else n_max
    grid = np.arange(n_max + 1)
    vals, cnt = np.unique(n606, return_counts=True)
    pm = (stats.binom.pmf(grid[None, :], vals[:, None], eta) * cnt[:, None]).sum(0) / cnt.sum()
    pb = stats.poisson.pmf(grid, bg)
    return np.convolve(pm, pb)[: n_max + 1]


def dark_pmf(bg, n_max):
    return stats.poisson.pmf(np.arange(n_max + 1), bg)


def fidelity(p_bright, p_dark):
    """Best threshold: call 'bright' if n > n_th. Returns (F, n_th, eps_bright, eps_dark)."""
    cb = np.cumsum(p_bright)  # P(n <= n_th | bright)
    cd = np.cumsum(p_dark)
    f = 1 - 0.5 * (cb + (1 - cd))
    i = int(np.argmax(f))
    return f[i], i, cb[i], 1 - cd[i]


# ----------------------------------------------------------------------------- EMCCD model
def emccd_sigma_for(eps01, threshold):
    """Gaussian ROI noise (photon units) giving false-positive probability eps01 at `threshold`."""
    return threshold / stats.norm.isf(eps01)


def emccd_eps(n606, eta, bg, sigma, thresholds):
    """False-negative / false-positive vs threshold for an EMCCD at high gain.

    A site with k photoelectrons gives summed signal S = Gamma(k, 1) + N(0, sigma^2)
    (excess-noise factor 2; k = 0 gives pure read/CIC noise). Bright sites have
    k ~ Binomial(N_i, eta) + Poisson(bg) mixed over molecules; dark sites have k ~ Poisson(bg).
    Returns (eps10, eps01) arrays over `thresholds`.
    """
    thresholds = np.atleast_1d(np.asarray(thresholds, float))
    pk_b = bright_pmf(n606, eta, bg)
    pk_d = dark_pmf(bg, len(pk_b) - 1)
    k = np.arange(len(pk_b))
    x = np.linspace(0, max(40.0, 3 * len(pk_b)), 4000)
    dx = x[1] - x[0]
    # P(S <= th | k): k = 0 -> Phi(th/sigma); k >= 1 -> integral of gamma pdf * Phi((th - x)/sigma)
    cdf = np.zeros((len(k), len(thresholds)))
    cdf[0] = stats.norm.cdf(thresholds / sigma)
    phi = stats.norm.cdf((thresholds[None, :] - x[:, None]) / sigma)  # (nx, nth)
    for kk in range(1, len(k)):
        w = stats.gamma.pdf(x, kk) * dx
        cdf[kk] = w @ phi
    eps10 = pk_b @ cdf
    eps01 = 1 - pk_d @ cdf
    return eps10, eps01

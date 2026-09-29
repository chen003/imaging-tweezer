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

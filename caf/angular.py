"""Angular-momentum helpers (cached Wigner symbols, spin matrices, spherical vectors)."""
from functools import lru_cache

import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j


def _r(x):
    return Rational(int(round(2 * x)), 2)


@lru_cache(maxsize=None)
def threej(j1, j2, j3, m1, m2, m3):
    return float(wigner_3j(_r(j1), _r(j2), _r(j3), _r(m1), _r(m2), _r(m3)))


@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, j, m):
    return float(clebsch_gordan(_r(j1), _r(j2), _r(j), _r(m1), _r(m2), _r(m)))


def mvals(j):
    return np.arange(j, -j - 1, -1)


def spin_ops(j):
    """Spherical components {+1, 0, -1} of angular momentum j in |j m> basis (m descending)."""
    m = mvals(j)
    d = len(m)
    jp = np.zeros((d, d))
    for i in range(1, d):
        jp[i - 1, i] = np.sqrt(j * (j + 1) - m[i] * (m[i] + 1))
    return {1: -jp / np.sqrt(2), 0: np.diag(m).astype(float), -1: jp.T / np.sqrt(2)}


def cart_to_sph(v):
    """Complex Cartesian vector (x, y, z) -> spherical components {+1, 0, -1}."""
    v = np.asarray(v, dtype=complex)
    return {1: -(v[0] + 1j * v[1]) / np.sqrt(2), 0: v[2], -1: (v[0] - 1j * v[1]) / np.sqrt(2)}


def sph_harm_c2(nhat):
    """Renormalised spherical harmonics C^2_q(nhat) for a real unit vector."""
    x, y, z = nhat
    return {0: 0.5 * (3 * z * z - 1),
            1: -np.sqrt(1.5) * z * (x + 1j * y), -1: np.sqrt(1.5) * z * (x - 1j * y),
            2: np.sqrt(3 / 8) * (x + 1j * y) ** 2, -2: np.sqrt(3 / 8) * (x - 1j * y) ** 2}

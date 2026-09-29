"""CaF level structure: X2Sigma+(v=0) N<=3 and A2Pi1/2(v=0) J'=1/2, 3/2, plus E1 dipole matrices.

Conventions
-----------
* Energies in MHz. The quantisation axis z is the magnetic-field direction.
* X states are built in the uncoupled basis |N mN; mS; mI> and diagonalised
  with the full hyperfine + Zeeman (+ optional tweezer tensor light shift) Hamiltonian.
* A states are built in Hund's case (a) |Omega', J', M'> (x) |mI>; parity
  eigenstates are the combinations (|Omega'=+1/2> + sigma |Omega'=-1/2>)/sqrt2, and
  the sign sigma belonging to each parity is fixed by the E1 selection rule.
  A(+) couples only to odd N, and A(-) only to even N.
* Dipole matrices ``A[q][e, g] = <e| d_q |g>`` (q = +1, 0, -1) are normalised so
  that every excited state has sum_{g,q} |<g|d_q|e>|^2 = 1 over the complete X
  rotational ladder. Vibrational branching (b00) is applied elsewhere.
"""
from dataclasses import dataclass, field

import numpy as np

from . import constants as k
from .angular import cg, mvals, sph_harm_c2, spin_ops, threej

S_E = 0.5
I_F = 0.5


# --------------------------------------------------------------------------- X state
class XState:
    def __init__(self, n_max=3):
        self.n_max = n_max
        self.basis = [(n, mn, ms, mi) for n in range(n_max + 1) for mn in mvals(n)
                      for ms in mvals(S_E) for mi in mvals(I_F)]
        self.dim = len(self.basis)
        self._build()

    def _op(self, elem):
        m = np.zeros((self.dim, self.dim), dtype=complex)
        for i, a in enumerate(self.basis):
            for j, b in enumerate(self.basis):
                m[i, j] = elem(a, b)
        return m

    def _build(self):
        sh = spin_ops(S_E)
        hi = {0.5: 0, -0.5: 1}

        def s_el(q, a, b):
            return sh[q][hi[a[2]], hi[b[2]]] if (a[0], a[1], a[3]) == (b[0], b[1], b[3]) else 0

        def i_el(q, a, b):
            return sh[q][hi[a[3]], hi[b[3]]] if (a[0], a[1], a[2]) == (b[0], b[1], b[2]) else 0

        def n_el(q, a, b):
            if (a[0], a[2], a[3]) != (b[0], b[2], b[3]):
                return 0
            ms = list(mvals(a[0]))
            return spin_ops(a[0])[q][ms.index(a[1]), ms.index(b[1])]

        def c2_el(q, a, b):
            if (a[2], a[3]) != (b[2], b[3]):
                return 0
            n1, m1, n2, m2 = a[0], a[1], b[0], b[1]
            return ((-1) ** int(round(m1)) * np.sqrt((2 * n1 + 1) * (2 * n2 + 1))
                    * threej(n1, 2, n2, 0, 0, 0) * threej(n1, 2, n2, -m1, q, m2))

        S = {q: self._op(lambda a, b, q=q: s_el(q, a, b)) for q in (1, 0, -1)}
        I = {q: self._op(lambda a, b, q=q: i_el(q, a, b)) for q in (1, 0, -1)}
        N = {q: self._op(lambda a, b, q=q: n_el(q, a, b)) for q in (1, 0, -1)}
        self.C2 = {q: self._op(lambda a, b, q=q: c2_el(q, a, b)) for q in range(-2, 3)}
        dot = lambda A, B: sum((-1) ** q * A[q] @ B[-q] for q in (1, 0, -1))
        IS2 = {q: sum(cg(1, q1, 1, q - q1, 2, q) * I[q1] @ S[q - q1]
                      for q1 in (1, 0, -1) if abs(q - q1) <= 1) for q in range(-2, 3)}
        h_dip = k.C_HF * np.sqrt(2 / 3) * sum((-1) ** q * IS2[q] @ self.C2[-q] for q in range(-2, 3))
        n_sq = np.diag([b[0] * (b[0] + 1) for b in self.basis]).astype(complex)
        self.H0 = (k.B_X * n_sq + k.GAMMA_SR * dot(N, S) + (k.B_HF + k.C_HF / 3) * dot(I, S)
                   + h_dip + k.C_I * dot(N, I))
        self.HZ = k.G_S * k.MU_B * S[0] - k.G_I * k.MU_N * I[0]  # per gauss, B || z
        self.Fz = N[0] + S[0] + I[0]
        self.Nvals = np.array([b[0] for b in self.basis])
        # case (b) -> case (a) map, rows (Omega, J, M, mI)
        self.case_a = [(om, j, m, mi) for j in np.arange(0.5, self.n_max + 1.0)
                       for om in (0.5, -0.5) for m in mvals(j) for mi in mvals(I_F)]
        idx = {s: i for i, s in enumerate(self.case_a)}
        W = np.zeros((len(self.case_a), self.dim))
        for c, (n, mn, ms, mi) in enumerate(self.basis):
            for j in (n - 0.5, n + 0.5):
                if j < 0:
                    continue
                m = mn + ms
                if abs(m) > j:
                    continue
                cgc = cg(n, mn, S_E, ms, j, m)
                if cgc == 0:
                    continue
                for om in (0.5, -0.5):
                    amp = (-1) ** int(round(j - S_E)) * np.sqrt(2 * n + 1) * threej(j, S_E, n, om, -om, 0)
                    W[idx[(om, j, m, mi)], c] += cgc * amp
        self.W = W

    def tensor_op(self, eps_tw):
        """P2(n . eps) for a real unit tweezer-polarisation vector."""
        c = sph_harm_c2(np.asarray(eps_tw, float) / np.linalg.norm(eps_tw))
        return sum((-1) ** q * self.C2[q] * c[-q] for q in range(-2, 3))

    def hamiltonian(self, b_gauss=0.0, u_tensor=0.0, eps_tw=(0, 0, 1)):
        """H_X in MHz. u_tensor = U_s * kappa_t (MHz); adds -u_tensor * P2(n.eps_tw)."""
        h = self.H0 + b_gauss * self.HZ
        if u_tensor:
            h = h - u_tensor * self.tensor_op(eps_tw)
        return h

    def eigen(self, **kw):
        e, v = np.linalg.eigh(self.hamiltonian(**kw))
        nexp = np.real(np.einsum("ij,j,ji->i", v.conj().T, self.Nvals, v))
        mf = np.real(np.einsum("ij,jk,ki->i", v.conj().T, self.Fz, v))
        return e, v, np.round(nexp).astype(int), mf


def kappa_tensor_for_spread(x: XState, spread=0.2):
    """kappa_t such that the first-order N=1 polarisability spread is `spread` (fraction of U_s)."""
    e0, v0, n0, _ = x.eigen(b_gauss=1e-6)
    sel = np.where(n0 == 1)[0]
    p2 = x.tensor_op((0, 0, 1))
    vals = []
    # degenerate perturbation theory inside each hyperfine manifold
    groups = {}
    for i in sel:
        groups.setdefault(round(e0[i] - e0[sel].min()), []).append(i)
    for g in groups.values():
        vv = v0[:, g]
        vals.extend(np.linalg.eigvalsh(vv.conj().T @ p2 @ vv).real)
    return spread / (max(vals) - min(vals))


# --------------------------------------------------------------------------- A state
@dataclass
class ALevel:
    j: float
    parity: int  # +1 / -1
    energy: float  # MHz relative to A(J'=1/2,+) centre


def a_levels(lambda_sign=+1):
    """A2Pi1/2 rotational/parity levels. lambda_sign=+1: (+) above (-) for J'=1/2."""
    ld = k.LAMBDA_DOUBLING * lambda_sign
    c12, c32 = 0.0, 3 * k.B_A_EFF
    # e levels: parity (-1)^(J-1/2); e above f when lambda_sign=+1
    lv = []
    for j, cen in ((0.5, c12), (1.5, c32)):
        e_par = (-1) ** int(round(j - 0.5))
        lv.append(ALevel(j, e_par, cen + ld * (j + 0.5) / 2))
        lv.append(ALevel(j, -e_par, cen - ld * (j + 0.5) / 2))
    ref = [l.energy for l in lv if l.j == 0.5 and l.parity == +1][0]
    for l in lv:
        l.energy -= ref
    return lv


class AState:
    """A(v'=0) Omega'=1/2 levels with J' in `js`, parity-resolved, with hyperfine and Zeeman."""

    def __init__(self, x: XState, js=(0.5, 1.5), lambda_sign=+1):
        self.x = x
        self.case_a = [(om, j, m, mi) for j in js for om in (0.5, -0.5) for m in mvals(j) for mi in mvals(I_F)]
        self.levels = a_levels(lambda_sign)
        self.js = js
        self._dipole_case_a()
        self._parity_states()

    def _dipole_case_a(self):
        """D[q][a_row, x_uncoupled] = <A case-a| d_q | X uncoupled>."""
        xa = self.x.case_a
        D = {q: np.zeros((len(self.case_a), len(xa))) for q in (1, 0, -1)}
        for r, (om1, j1, m1, mi1) in enumerate(self.case_a):
            for c, (om, j, m, mi) in enumerate(xa):
                if mi1 != mi or abs(j1 - j) > 1:
                    continue
                p = om1 - om
                if abs(abs(p) - 1) > 1e-9:
                    continue
                q = m1 - m
                if abs(q) > 1:
                    continue
                D[int(round(q))][r, c] = ((-1) ** int(round(m1 - om1)) * np.sqrt((2 * j1 + 1) * (2 * j + 1))
                                          * threej(j1, 1, j, -m1, q, m) * threej(j1, 1, j, -om1, p, om))
        self.Dcase = {q: D[q] @ self.x.W for q in D}  # to X uncoupled basis

    def _parity_states(self):
        """Parity combinations; identify sign by E1 selection rule (A(+) <-> odd N)."""
        idx = {s: i for i, s in enumerate(self.case_a)}
        nv = self.x.Nvals
        even = (nv % 2 == 0)
        self.pbasis = []  # rows: vectors in case_a space
        labels = []
        for j in self.js:
            for sigma in (+1, -1):
                vecs = []
                for m in mvals(j):
                    for mi in mvals(I_F):
                        v = np.zeros(len(self.case_a))
                        v[idx[(0.5, j, m, mi)]] = 1 / np.sqrt(2)
                        v[idx[(-0.5, j, m, mi)]] = sigma / np.sqrt(2)
                        vecs.append(v)
                vecs = np.array(vecs)
                w_even = sum(np.abs(vecs @ self.Dcase[q][:, even]) ** 2 for q in (1, 0, -1)).sum()
                w_odd = sum(np.abs(vecs @ self.Dcase[q][:, ~even]) ** 2 for q in (1, 0, -1)).sum()
                assert min(w_even, w_odd) < 1e-12 * max(w_even, w_odd), "parity selection rule broken"
                parity = +1 if w_even < w_odd else -1  # A(+) couples to odd N (parity -)
                for (m, mi), v in zip([(m, mi) for m in mvals(j) for mi in mvals(I_F)], vecs):
                    self.pbasis.append(v)
                    labels.append((j, parity, m, mi))
        self.pbasis = np.array(self.pbasis)
        self.plabels = labels

    def block(self, j, parity, b_gauss=0.0, a_hf=None, g_j=None):
        """Eigenstates of one (J', parity) level: energies (MHz), dipoles to X uncoupled, F' labels."""
        a_hf = (k.A_HF_J12 if j == 0.5 else 0.0) if a_hf is None else a_hf
        g_j = (k.G_J_A12 if j == 0.5 else 0.0) if g_j is None else g_j
        rows = [i for i, l in enumerate(self.plabels) if l[0] == j and l[1] == parity]
        e0 = [l.energy for l in self.levels if l.j == j and l.parity == parity][0]
        jo, io = spin_ops(j), spin_ops(I_F)
        dj, di = len(mvals(j)), 2
        JI = sum((-1) ** q * np.kron(jo[q], io[-q]) for q in (1, 0, -1))
        h = (e0 * np.eye(dj * di) + a_hf * (JI - np.diag(np.diag(JI)) * 0) + b_gauss
             * (g_j * k.MU_B * np.kron(jo[0], np.eye(2)) - k.G_I * k.MU_N * np.kron(np.eye(dj), io[0])))
        # row ordering in self.plabels is (m, mi) with m outer -> matches kron(j, i)
        en, vec = np.linalg.eigh(h)
        states = vec.T @ self.pbasis[rows]  # (n_states, case_a)
        dip = {q: states @ self.Dcase[q] for q in (1, 0, -1)}
        fz = np.real(np.diag(vec.T @ (np.kron(jo[0], np.eye(2)) + np.kron(np.eye(dj), io[0])) @ vec))
        return en - (a_hf * 0.25 if j == 0.5 else 0.0), dip, fz


# --------------------------------------------------------------------------- dipoles
def normalise_dipoles(dip_to_uncoupled):
    """Scale so every excited state has unit total line strength over the full X ladder."""
    tot = sum(np.abs(dip_to_uncoupled[q]) ** 2 for q in (1, 0, -1)).sum(axis=1)
    return {q: dip_to_uncoupled[q] / np.sqrt(tot)[:, None] for q in dip_to_uncoupled}


def project(dip, xvecs):
    """<e|d_q|g> for X eigenvectors (columns of xvecs, uncoupled basis)."""
    return {q: dip[q] @ xvecs for q in dip}


def eps_dot_d(dip, eps):
    """Matrix <e| eps . d |g> for complex Cartesian polarisation eps (absorption)."""
    ex, ey, ez = np.asarray(eps, complex) / np.linalg.norm(eps)
    dx = (dip[-1] - dip[1]) / np.sqrt(2)
    dy = 1j * (dip[-1] + dip[1]) / np.sqrt(2)
    return ex * dx + ey * dy + ez * dip[0]


@dataclass
class Manifold:
    """Selected X eigenstates plus bookkeeping."""
    energies: np.ndarray
    vecs: np.ndarray
    mf: np.ndarray
    hf_label: np.ndarray = field(default=None)  # hyperfine manifold index (0..3 for N=1)


def x_manifold(x: XState, n, b_gauss=0.0, u_tensor=0.0, eps_tw=(0, 0, 1)):
    e, v, nn, mf = x.eigen(b_gauss=b_gauss, u_tensor=u_tensor, eps_tw=eps_tw)
    sel = np.where(nn == n)[0]
    man = Manifold(e[sel], v[:, sel], mf[sel])
    if n == 1:
        # hyperfine manifold by nearest zero-field energy (J=1/2 F=1, F=0, J=3/2 F=1, F=2)
        e0, _, nn0, _ = x.eigen(b_gauss=1e-6)
        ref = np.sort(np.unique(np.round(e0[nn0 == 1], 1)))
        centres = []
        for r in ref:
            if not centres or abs(r - centres[-1]) > 5:
                centres.append(r)
        man.hf_label = np.array([int(np.argmin(np.abs(np.array(centres) - ei))) for ei in man.energies])
        man.hf_centres = np.array(centres)
    elif n == 0:
        man.hf_label = (man.energies - man.energies.min() > 60).astype(int)
    return man

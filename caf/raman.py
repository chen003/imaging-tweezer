"""Off-resonant (Kramers-Heisenberg) photon scattering of the imaging light by X(v=0, N=0).

The imaging light (one component per N=1 hyperfine manifold, frequency
nu_h = -centre_h + delta, with the A(J'=1/2,+) centre as the zero of energy) is
detuned by about 19 GHz from X(N=0) -> A(J'=1/2,-) and about 52 GHz from
X(N=0) -> A(J'=3/2,-). Both are included as intermediate states, with their
hyperfine and Zeeman structure and with coherent (interfering) amplitudes.

Final states: X(v=0,N=0) (4), X(v=0,N=2) (20), and X(v>=1) (lumped, weight 1-b00).
Each laser component scatters incoherently with respect to the others, because
the scattered photons have different frequencies.
"""
import numpy as np

from . import constants as k
from .structure import AState, XState, eps_dot_d, normalise_dipoles, project, x_manifold


class N0Raman:
    def __init__(self, x: XState, a: AState, b_gauss=1.0, u_tensor=0.0, eps_tw=(0, 0, 1), b00=k.B00):
        self.m0 = x_manifold(x, 0, b_gauss=b_gauss, u_tensor=u_tensor, eps_tw=eps_tw)
        self.m2 = x_manifold(x, 2, b_gauss=b_gauss, u_tensor=u_tensor, eps_tw=eps_tw)
        self.b00 = b00
        ens, d0, d2 = [], [], []
        for j in (0.5, 1.5):
            en, dip, _ = a.block(j, -1, b_gauss=b_gauss)
            dip = normalise_dipoles(dip)
            ens.append(en)
            d0.append(project(dip, self.m0.vecs))
            d2.append(project(dip, self.m2.vecs))
        self.e_exc = np.concatenate(ens)
        self.d0 = {q: np.vstack([d[q] for d in d0]) for q in (1, 0, -1)}  # (12, 4)
        self.d2 = {q: np.vstack([d[q] for d in d2]) for q in (1, 0, -1)}  # (12, 20)

    def rates(self, s, delta, eps, centres, offsets=(0, 0, 0, 0)):
        """Scattering-rate matrix [1/s] from each N=0 state.

        s: per-component saturation parameters (len 4); eps: (4,3) polarisations;
        centres: N=1 manifold centre energies (MHz) that the components are tuned to.
        Returns dict with 'to_N0' (4x4), 'to_N2' (4x20), 'to_v1' (4,), 'total' (4,).
        """
        eps = np.asarray(eps, complex)
        if eps.ndim == 1:
            eps = np.tile(eps, (4, 1))
        gam = k.GAMMA
        to0 = np.zeros((4, 4))
        to2 = np.zeros((4, 20))
        tot = np.zeros(4)
        for h in range(4):
            nu = -centres[h] + delta + offsets[h]
            om = gam * np.sqrt(s[h] / 2) * np.sqrt(self.b00)
            m = eps_dot_d(self.d0, eps[h])  # (12 exc, 4 N0)
            det = nu - (self.e_exc[:, None] - self.m0.energies[None, :])  # MHz
            c = 0.5 * om * m / (2 * np.pi * det * 1e6)  # (12, 4) excited amplitudes per initial state
            tot += gam * np.sum(np.abs(c) ** 2, axis=0)
            for q in (1, 0, -1):
                # <f|d_q|e> = conj(<e|d_q|f>)
                a0 = self.d0[q].conj().T @ c  # (4 f, 4 i)
                a2 = self.d2[q].conj().T @ c  # (20 f, 4 i)
                to0 += gam * self.b00 * np.abs(a0.T) ** 2
                to2 += gam * self.b00 * np.abs(a2.T) ** 2
        v1 = (1 - self.b00) * tot
        return {"to_N0": to0, "to_N2": to2, "to_v1": v1, "total": tot,
                "raman": tot - np.diag(to0), "rayleigh": np.diag(to0)}

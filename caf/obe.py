"""Optical Bloch equations for imaging X(v=0,N=1) on X -> A(v'=0, J'=1/2, +).

States: 12 X(N=1) eigenstates (with B along z and an optional tweezer tensor shift),
4 A(J'=1/2,+) states and one reservoir level r lumping X(v>=1). Molecules in r are
returned to A by the vibrational repumpers at rate k_rep.

Units: time in microseconds, angular frequency in rad/us (= 2 pi x MHz).

The laser has one frequency component per N=1 hyperfine manifold h (4 components).
Each component is centred on that manifold, so it sits at detuning ``delta``
(MHz, common to all) from it. A component can have its own polarisation and
saturation parameter s_h, normalised to the two-level I_sat.

``secular=True`` keeps only the coupling of each component to its own manifold.
That gives a time-independent Liouvillian with an exact steady state.
``secular=False`` keeps every cross-coupling with its beat note and is integrated
in time.
"""
import numpy as np

from . import constants as k
from .structure import AState, XState, eps_dot_d, normalise_dipoles, project, x_manifold

TWO_PI = 2 * np.pi
GAM = k.GAMMA * 1e-6  # rad/us


def _comm(h):
    n = h.shape[0]
    eye = np.eye(n)
    return -1j * (np.kron(eye, h) - np.kron(h.T, eye))


def _diss(l):
    n = l.shape[0]
    eye = np.eye(n)
    ll = l.conj().T @ l
    return np.kron(l.conj(), l) - 0.5 * np.kron(eye, ll) - 0.5 * np.kron(ll.T, eye)


class N1Imaging:
    def __init__(self, x: XState, a: AState, b_gauss=1.0, u_tensor=0.0, eps_tw=(0, 0, 1),
                 b00=k.B00, k_rep=TWO_PI * 1.0, laser_centres=None):
        self.man = x_manifold(x, 1, b_gauss=b_gauss, u_tensor=u_tensor, eps_tw=eps_tw)
        en_e, dip_e, fz_e = a.block(0.5, +1, b_gauss=b_gauss)
        dip_e = normalise_dipoles(dip_e)
        self.d = project(dip_e, self.man.vecs)  # (4, 12)
        self.e_exc = en_e
        self.ng, self.ne = 12, 4
        self.n = self.ng + self.ne + 1
        self.b00, self.k_rep = b00, k_rep
        self.hf = self.man.hf_label
        # laser components are tuned to the manifold centres (mean energy of each manifold,
        # incl. tensor shift) unless fixed laser frequencies are supplied
        own = np.array([self.man.energies[self.hf == h].mean() for h in range(4)])
        self.centres = own if laser_centres is None else np.asarray(laser_centres, float)
        self._build_static()

    # ------------------------------------------------------------ building blocks
    def _idx(self):
        g = np.arange(self.ng)
        e = self.ng + np.arange(self.ne)
        return g, e, self.n - 1

    def _build_static(self):
        g, e, r = self._idx()
        # dissipators (secular: per-manifold split of the decay operator; full: joint)
        self.D_sec = np.zeros((self.n ** 2, self.n ** 2), complex)
        self.D_full = np.zeros_like(self.D_sec)
        amp = np.sqrt(GAM * self.b00)
        for q in (1, 0, -1):
            lq = np.zeros((self.n, self.n), complex)
            lq[np.ix_(g, e)] = amp * self.d[q].conj().T
            self.D_full += _diss(lq)
            for h in range(4):
                lh = np.zeros_like(lq)
                sel = g[self.hf == h]
                lh[np.ix_(sel, e)] = lq[np.ix_(sel, e)]
                self.D_sec += _diss(lh)
        rep = np.zeros_like(self.D_sec)
        for ei in e:
            l_in = np.zeros((self.n, self.n), complex)
            l_in[r, ei] = np.sqrt(GAM * (1 - self.b00))
            l_out = np.zeros((self.n, self.n), complex)
            l_out[ei, r] = np.sqrt(self.k_rep / self.ne)
            rep += _diss(l_in) + _diss(l_out)
        self.D_sec += rep
        self.D_full += rep

    def _couplings(self, s, eps):
        """Per-component absorption matrices H_h^+ (|e><g| blocks) in rad/us."""
        g, e, _ = self._idx()
        out = []
        for h in range(4):
            om = GAM * np.sqrt(s[h] / 2) * np.sqrt(self.b00)
            m = eps_dot_d(self.d, eps[h])  # (4, 12)
            hp = np.zeros((self.n, self.n), complex)
            hp[np.ix_(e, g)] = 0.5 * om * m
            out.append(hp)
        return out

    def _norm_args(self, s_tot, weights, eps):
        weights = np.ones(4) / 4 if weights is None else np.asarray(weights, float) / np.sum(weights)
        s = s_tot * weights
        if eps is None:
            eps = (1, 0, 0)
        eps = np.asarray(eps, complex)
        if eps.ndim == 1:
            eps = np.tile(eps, (4, 1))
        return s, eps

    # ------------------------------------------------------------ secular steady state
    def liouvillian_secular(self, s_tot, delta, weights=None, eps=None, offsets=None):
        s, eps = self._norm_args(s_tot, weights, eps)
        offsets = np.zeros(4) if offsets is None else np.asarray(offsets)
        g, e, r = self._idx()
        h0 = np.zeros((self.n, self.n), complex)
        # frame: ground g of manifold h at E_g - centre_h + delta_h, excited at e_hf
        h0[g, g] = TWO_PI * (self.man.energies - self.centres[self.hf] + delta + offsets[self.hf])
        h0[e, e] = TWO_PI * self.e_exc
        hc = self._couplings(s, eps)
        hint = np.zeros_like(h0)
        for h in range(4):
            mask = np.zeros((self.n, self.n), bool)
            mask[np.ix_(e, g[self.hf == h])] = True
            hint += np.where(mask, hc[h], 0)
        hint = hint + hint.conj().T
        return _comm(h0 + hint) + self.D_sec

    def steady_state(self, s_tot, delta, **kw):
        L = self.liouvillian_secular(s_tot, delta, **kw)
        n = self.n
        A = L.copy()
        A[0, :] = np.eye(n).reshape(-1)  # trace row (column-stacked vec: diag at i*(n+1))
        b = np.zeros(n * n, complex)
        b[0] = 1
        rho = np.linalg.solve(A, b).reshape(n, n, order="F")
        return rho

    def rates(self, rho):
        """Photon scattering rates [1/s]: total and on the 606 nm (v=0) channel."""
        _, e, r = self._idx()
        pe = np.real(np.trace(rho[np.ix_(e, e)]))
        return {"R_tot": GAM * pe * 1e6, "R_606": GAM * self.b00 * pe * 1e6, "p_e": pe,
                "p_res": np.real(rho[r, r])}

    def scattering_rate(self, s_tot, delta, **kw):
        return self.rates(self.steady_state(s_tot, delta, **kw))["R_606"]

    # ------------------------------------------------------------ time-dependent (full / modulated)
    def evolve(self, s_tot, delta, t_end, dt=2e-4, weights=None, eps=None, pol_mod=None,
               secular=False, t_avg=None):
        """RK4 time evolution from a mixed N=1 state; returns time-averaged R_606 over the last t_avg us.

        pol_mod=(f_MHz, eps_a, eps_b): polarisation rotates as eps_a cos(wt) + eps_b sin(wt).
        """
        s, eps = self._norm_args(s_tot, weights, eps)
        g, e, r = self._idx()
        n = self.n
        h0 = np.zeros((n, n), complex)
        if secular:
            h0[g, g] = TWO_PI * (self.man.energies - self.centres[self.hf] + delta)
            D = self.D_sec
        else:
            nu = -self.centres + delta  # component frequencies (A centre = 0)
            nu_ref = nu[0]
            h0[g, g] = TWO_PI * (self.man.energies + nu_ref)
            D = self.D_full
        h0[e, e] = TWO_PI * self.e_exc
        L0 = _comm(h0) + D
        terms = []  # (callable(t)->complex coefficient, superop)

        def add(hp, phase_rate, mod=None):
            Sp, Sm = _comm(hp), _comm(hp.conj().T)
            if mod is None:
                terms.append((lambda t, w=phase_rate: np.exp(-1j * w * t), Sp))
                terms.append((lambda t, w=phase_rate: np.exp(1j * w * t), Sm))
            else:
                terms.append((lambda t, w=phase_rate, f=mod: f(t) * np.exp(-1j * w * t), Sp))
                terms.append((lambda t, w=phase_rate, f=mod: np.conj(f(t)) * np.exp(1j * w * t), Sm))

        if pol_mod is None:
            eps_list = [(eps, None)]
        else:
            f, ea, eb = pol_mod
            w = TWO_PI * f
            eps_list = [(np.tile(np.asarray(ea, complex), (4, 1)), lambda t, w=w: np.cos(w * t)),
                        (np.tile(np.asarray(eb, complex), (4, 1)), lambda t, w=w: np.sin(w * t))]
        for eps_i, mod in eps_list:
            hc = self._couplings(s, eps_i)
            for h in range(4):
                if secular:
                    mask = np.zeros((n, n), bool)
                    mask[np.ix_(e, g[self.hf == h])] = True
                    add(np.where(mask, hc[h], 0), 0.0, mod)
                else:
                    add(hc[h], TWO_PI * (nu[h] - nu_ref), mod)

        rho = np.zeros((n, n), complex)
        rho[g, g] = 1 / self.ng
        v = rho.reshape(-1, order="F")

        def deriv(t, v):
            out = L0 @ v
            for c, S in terms:
                out = out + c(t) * (S @ v)
            return out

        steps = int(round(t_end / dt))
        t_avg = t_end / 2 if t_avg is None else t_avg
        n_avg = int(round(t_avg / dt))
        diag_e = np.array([ei * (n + 1) for ei in e])
        acc = 0.0
        t = 0.0
        for i in range(steps):
            k1 = deriv(t, v)
            k2 = deriv(t + dt / 2, v + dt / 2 * k1)
            k3 = deriv(t + dt / 2, v + dt / 2 * k2)
            k4 = deriv(t + dt, v + dt * k3)
            v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if i >= steps - n_avg:
                acc += np.real(v[diag_e].sum())
        pe = acc / n_avg
        return GAM * self.b00 * pe * 1e6

"""Scattering-rate lookups averaged over the local field of several imaging beams (parallel).

Each field sample is (intensity factor, lab-frame polarisation). Averaging R over samples is the
fast-mixing limit: the molecule moves through the standing-wave pattern faster than R changes.
"""
from multiprocessing import get_context

import numpy as np

from .lookup import frame_from_b, mk_to_mhz
from .obe import N1Imaging
from .raman import N0Raman
from .structure import AState, XState

_W = {}


def _init(kappa_t):
    x = XState()
    _W["x"], _W["a"], _W["kappa_t"] = x, AState(x), kappa_t


def _one(args):
    b_gauss, bdir, eps_tw_lab, s_eff, eps_lab, u0_mk, weights, offsets, delta, u = args
    x, a, kt = _W["x"], _W["a"], _W["kappa_t"]
    rot = frame_from_b(bdir)
    eps_tw = np.real(rot @ np.asarray(eps_tw_lab, complex))
    eps = rot @ np.asarray(eps_lab, complex)
    b = max(b_gauss, 1e-3)
    ut0 = kt * mk_to_mhz(u0_mk)
    centres = N1Imaging(x, a, b_gauss=b, u_tensor=ut0, eps_tw=eps_tw).centres
    R = np.zeros((len(u), len(delta)))
    for iu, uu in enumerate(u):
        sysm = N1Imaging(x, a, b_gauss=b, u_tensor=ut0 * uu, eps_tw=eps_tw, laser_centres=centres)
        for idl, d in enumerate(delta):
            R[iu, idl] = sysm.rates(sysm.steady_state(s_eff, d, eps=eps, weights=weights,
                                                      offsets=offsets))["R_tot"]
    return R


def _raman(args):
    b_gauss, bdir, eps_tw_lab, eps_lab, weights, offsets, u0_mk = args
    x, a, kt = _W["x"], _W["a"], _W["kappa_t"]
    rot = frame_from_b(bdir)
    eps_tw = np.real(rot @ np.asarray(eps_tw_lab, complex))
    eps = rot @ np.asarray(eps_lab, complex)
    b = max(b_gauss, 1e-3)
    centres = N1Imaging(x, a, b_gauss=b, u_tensor=kt * mk_to_mhz(u0_mk), eps_tw=eps_tw).centres
    ram = N0Raman(x, a, b_gauss=b, eps_tw=eps_tw)
    w = np.asarray(weights, float) / np.sum(weights)
    out = ram.rates(w, 0.0, eps, centres, offsets=offsets)
    return out["raman"], out["total"], np.round(ram.m0.mf).astype(int), ram.m0.hf_label


class EnsembleLookup:
    def __init__(self, kappa_t, processes=4):
        self.pool = get_context("fork").Pool(processes, initializer=_init, initargs=(kappa_t,))

    def close(self):
        self.pool.close()
        self.pool.join()

    def table(self, b_gauss, bdir, eps_tw_lab, samples, s_tot, u0_mk, weights, offsets=(0, 0, 0, 0),
              delta=np.arange(-20, 20.01, 1.0), u=np.linspace(0, 1, 6)):
        """Return (delta, u, R_mean[u, delta], R_samples[n, u, delta])."""
        jobs = [(b_gauss, bdir, eps_tw_lab, s_tot * f, e, u0_mk, weights, offsets, delta, u) for f, e in samples]
        Rs = np.array(self.pool.map(_one, jobs))
        weights_s = np.ones(len(samples)) / len(samples)
        return np.asarray(delta), np.asarray(u), np.tensordot(weights_s, Rs, axes=1), Rs

    def raman(self, b_gauss, bdir, eps_tw_lab, samples, weights, offsets=(0, 0, 0, 0), u0_mk=0.93):
        """Raman and total N=0 scattering rates per unit s_tot, intensity-weighted over the samples."""
        jobs = [(b_gauss, bdir, eps_tw_lab, e, weights, offsets, u0_mk) for f, e in samples]
        res = self.pool.map(_raman, jobs)
        f = np.array([s[0] for s in samples])
        ram = np.sum([fi * r[0] for fi, r in zip(f, res)], axis=0) / len(samples)
        tot = np.sum([fi * r[1] for fi, r in zip(f, res)], axis=0) / len(samples)
        return {"raman": ram, "total": tot, "mf": res[0][2], "F": res[0][3]}

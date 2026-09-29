"""Scattering-rate lookup tables R(delta, u) for the Monte-Carlo.

u = U(r)/U0 is the local tweezer intensity relative to the trap centre. The
tensor light shift scales with u, while the laser frequencies stay fixed at the
trap-centre manifold centres. The scalar differential shift is applied in the
Monte-Carlo through delta.
"""
from dataclasses import dataclass, field

import numpy as np

from . import constants as k
from .obe import N1Imaging
from .raman import N0Raman
from .structure import AState, XState, kappa_tensor_for_spread


def mk_to_mhz(u_mk):
    return u_mk * 1e-3 * k.KB / k.H / 1e6


def frame_from_b(bdir):
    z = np.asarray(bdir, float)
    z /= np.linalg.norm(z)
    tmp = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    xx = tmp - z * np.dot(tmp, z)
    xx /= np.linalg.norm(xx)
    yy = np.cross(z, xx)
    return np.vstack([xx, yy, z])  # rows: B-frame axes in lab coordinates


@dataclass
class Geometry:
    """Lab-frame description. Tweezer along z_lab with polarisation eps_tw; imaging beam(s) with eps_img."""
    b_gauss: float = 1.0
    b_dir: tuple = (1, 0, 0)
    eps_tw: tuple = (1, 0, 0)
    eps_img: tuple = (np.cos(np.radians(50)), 0, np.sin(np.radians(50)))
    beams: tuple = ((0, 1, 0), (0, -1, 0))  # retro-reflected imaging beam along y_lab
    weights: tuple = (2, 1, 2, 3)  # power split over (J1/2 F1, F0, J3/2 F1, F2)
    # small per-sideband detuning offsets (MHz); exact tuning of all four sidebands to the
    # manifold centres produces accidental two-photon (coherent) dark resonances
    offsets: tuple = (1.0, -1.0, 2.0, -2.0)
    name: str = ""

    def in_b_frame(self, v):
        return frame_from_b(self.b_dir) @ np.asarray(v, complex)


@dataclass
class Lookup:
    delta: np.ndarray
    u: np.ndarray
    R_tot: np.ndarray  # (n_u, n_delta) [1/s]
    raman_per_s: np.ndarray = field(default=None)  # Raman rate per unit s_tot for each N=0 state [1/s]
    raman_detail: dict = field(default=None)


class Model:
    """Caches the X/A structure (slow to build) and the tensor coefficient."""

    def __init__(self, tensor_spread=0.2):
        self.x = XState()
        self.a = AState(self.x)
        self.kappa_t = kappa_tensor_for_spread(self.x, tensor_spread) if tensor_spread else 0.0

    def lookup(self, geom: Geometry, s_tot, u0_mk, delta=np.arange(-30, 30.01, 0.5),
               u=np.linspace(0, 1, 11)):
        b = max(geom.b_gauss, 1e-3)
        eps_tw = np.real(geom.in_b_frame(geom.eps_tw))
        eps_img = geom.in_b_frame(geom.eps_img)
        ut0 = self.kappa_t * mk_to_mhz(u0_mk)
        centre_sys = N1Imaging(self.x, self.a, b_gauss=b, u_tensor=ut0, eps_tw=eps_tw)
        centres = centre_sys.centres
        R = np.zeros((len(u), len(delta)))
        for iu, uu in enumerate(u):
            sysm = N1Imaging(self.x, self.a, b_gauss=b, u_tensor=ut0 * uu, eps_tw=eps_tw, laser_centres=centres)
            for idl, d in enumerate(delta):
                R[iu, idl] = sysm.rates(sysm.steady_state(s_tot, d, eps=eps_img, weights=geom.weights,
                                                          offsets=geom.offsets))["R_tot"]
        ram = N0Raman(self.x, self.a, b_gauss=b, eps_tw=eps_tw)
        w = np.asarray(geom.weights, float) / np.sum(geom.weights)
        out = ram.rates(w * 1.0, 0.0, eps_img, centres, offsets=geom.offsets)  # per unit s_tot
        return Lookup(np.asarray(delta), np.asarray(u), R, out["raman"], out)

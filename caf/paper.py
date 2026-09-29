"""Experimental configuration of Holland et al., PRX (2025) / arXiv:2406.02391 (values quoted from the paper).

Rapid resonant imaging (Sec. IV, App. E):
* four X(v=0,N=1) -> A(J'=1/2,+) hyperfine components, power ratio 5 : 2.7 : 1.1 : 3.8
  (taken here in order J=1/2 F=1, F=0, J=3/2 F=1, F=2; the paper does not state the order),
  delivered by two retro-reflected beams, total I = 4.5 mW/cm^2, duration 3 ms;
* tweezer 781 nm, w0 = 730 nm, NA 0.65, imaging depth 930(20) uK, reached by a 1 ms ramp
  from 130 uK (state-error image) or 39 uK (erasure image);
* state-error image at B = 4.4 G, 53 deg from the tweezer polarisation;
  composite-erasure image at B = 2.0 G, 90 deg;
* about 24 photons collected in 3 ms; inferred R_scat ~ 1.6e5 /s (their free-space OBE: 2.8e5 /s);
* EMCCD, threshold 4.8, eps01 = 7.2 %, eps10 = 3.3 % (state-error image) / 6.3 %, 4.5 % (erasure image);
* N=1 differential ac Stark shifts about 10 % of the trap depth (RMS);
* N=0 |F=1,mF=-1> loss from imaging light: gamma = 0.76(11) /s measured, 0.24 /s golden-rule estimate;
* composite erasure: imaging-light contribution 3.0(3)e-3 per image for |1> = |N=0,F=1,mF=0>.
"""
import numpy as np

from . import constants as k

I_IMG = 4.5  # mW/cm^2, total
S_TOT = I_IMG / k.I_SAT  # ~0.93
T_IMG_US = 3000.0
U_IMG_MK = 0.930
W0_UM = 0.73
LAMBDA_TW_NM = 781.0
WEIGHTS = (5.0, 2.7, 1.1, 3.8)
TENSOR_RMS = 0.10
PHOTONS_COLLECTED = 24.0
R_SCAT_MEASURED = 1.6e5
R_SCAT_PAPER_OBE = 2.8e5
ETA_PAPER = PHOTONS_COLLECTED / (R_SCAT_MEASURED * T_IMG_US * 1e-6)  # ~5 %
THRESHOLD = 4.8
EPS01 = 0.072
EPS10 = {"state": 0.033, "erasure": 0.045}
GAMMA_MINUS_MEAS = 0.76  # /s, |F=1,mF=-1>
GAMMA_MINUS_THEORY = 0.24  # /s
EP_IMG_ERASURE = 3.0e-3  # per 3 ms image, |1>
T0_UK = 80.0  # calibrated: reproduces eps10 = 3.3 %, ~23 collected photons and ~70 % survival of the
              # 4.4 G state-error image (120 uK gives eps10 = 7.5 %, 30 uK gives 1.2 %)

CONFIGS = {  # (|B| in G, angle between B and tweezer polarisation in deg)
    "state_4.4G_53": (4.4, 53.0),
    "erasure_2.0G_90": (2.0, 90.0),
    "1G_53": (1.0, 53.0),
    "1G_90": (1.0, 90.0),
    "0G": (0.0, 53.0),
}


def b_dir(theta_deg):
    """Tweezer polarisation along x_lab, tweezer axis z_lab; B in the x-z plane at theta from x."""
    t = np.radians(theta_deg)
    return (np.cos(t), 0.0, np.sin(t))


def field_samples(n=24, seed=3, pol_a=None, pol_b=None):
    """Local field samples of two retro-reflected beams along x_lab and y_lab (both perpendicular to the
    tweezer axis).

    Beam 1 is polarised in the y-z plane at angle a, beam 2 in the x-z plane at angle b. Each sample
    draws random standing-wave phases and a random relative temporal phase. Returns a list of
    (intensity factor, unit complex polarisation) with mean intensity factor 1. If pol_a or pol_b
    is None, that angle is drawn at random too (unknown polarisation).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = rng.uniform(0, np.pi) if pol_a is None else pol_a
        b = rng.uniform(0, np.pi) if pol_b is None else pol_b
        e1 = np.array([0, np.cos(a), np.sin(a)])
        e2 = np.array([np.cos(b), 0, np.sin(b)])
        f1, f2, psi = rng.uniform(0, 2 * np.pi, 3)
        # retro-reflected standing waves: amplitude 2 cos(phase); four passes of equal intensity I/4
        e = np.sqrt(0.25) * (2 * np.cos(f1) * e1 + 2 * np.cos(f2) * np.exp(1j * psi) * e2)
        inten = float(np.vdot(e, e).real)
        out.append([inten, e / np.sqrt(inten) if inten > 0 else e1.astype(complex)])
    mean = np.mean([o[0] for o in out])
    return [(o[0] / mean, o[1]) for o in out]

"""Semiclassical Monte-Carlo of a molecule scattering imaging photons in a Gaussian tweezer.

Each molecule follows a classical trajectory (velocity Verlet) in
V = -U0 u(r), with u(r) = exp(-2 rho^2 / w(z)^2) / (1 + z^2/zR^2).
Photon events are drawn in every time step with probability R(delta_loc) dt.
R comes from the OBE lookup table, and the local detuning is

    delta_loc = Delta + kappa_s (1 - u(r)) - k_b . v / 2pi        [MHz]

kappa_s = (1 - alpha_A/alpha_X) U0/h is the differential scalar light shift of
the X-A line at the trap bottom. Each event gives an absorption kick along the
beam and an emission kick in a random direction. The photon is at 606 nm (and
so detectable) with probability b00.

Loss channels:
* escape from the trap (|rho| > 3 w0 or |z| > 3 zR);
* trap-light-induced loss while excited, with probability p_trap * u(r) per event;
* decay to unrepumped vibrational levels, with probability b_loss per event.
"""
import numpy as np
from numba import njit, prange

from . import constants as k


@njit(cache=True)
def _interp2(xg0, dxg, ug0, dug, yg, x, uu):
    """Bilinear interpolation of yg[u, delta], clamped at the grid edges."""
    nu, nx = yg.shape
    f = (x - xg0) / dxg
    g = (uu - ug0) / dug
    f = min(max(f, 0.0), nx - 1.000001)
    g = min(max(g, 0.0), nu - 1.000001)
    i = int(f)
    j = int(g)
    t = f - i
    s = g - j
    return ((1 - s) * ((1 - t) * yg[j, i] + t * yg[j, i + 1])
            + s * ((1 - t) * yg[j + 1, i] + t * yg[j + 1, i + 1]))


@njit(cache=True)
def _u_and_grad(x, y, z, w0, zr):
    s = 1.0 + (z / zr) ** 2
    rho2 = x * x + y * y
    u = np.exp(-2.0 * rho2 / (w0 * w0 * s)) / s
    gx = u * (-4.0 * x / (w0 * w0 * s))
    gy = u * (-4.0 * y / (w0 * w0 * s))
    gz = u * (2.0 * z / (zr * zr)) / s * (-1.0 + 2.0 * rho2 / (w0 * w0 * s))
    return u, gx, gy, gz


@njit(parallel=True, cache=True)
def _run(n_mol, U0, w0, zr, mass, T0, xg0, dxg, ug0, dug, Rg, beams, fbeam, delta0, kappa_s,
         p_trap, b_loss, b00, dt, n_steps, ck_steps, vrec, kwave, seed):
    n_ck = ck_steps.shape[0]
    counts = np.zeros((n_mol, n_ck), dtype=np.int32)
    events = np.zeros((n_mol, n_ck), dtype=np.int32)
    t_dead = np.full(n_mol, -1.0)
    e_final = np.zeros(n_mol)
    kb = 1.380649e-23
    wr = np.sqrt(4 * U0 / (mass * w0 * w0))
    wz = np.sqrt(2 * U0 / (mass * zr * zr))
    sv = np.sqrt(kb * T0 / mass)
    nb = beams.shape[0]
    for m in prange(n_mol):
        np.random.seed(seed + m)
        x = np.random.normal() * sv / wr
        y = np.random.normal() * sv / wr
        z = np.random.normal() * sv / wz
        vx = np.random.normal() * sv
        vy = np.random.normal() * sv
        vz = np.random.normal() * sv
        u, gx, gy, gz = _u_and_grad(x, y, z, w0, zr)
        ax, ay, az = U0 * gx / mass, U0 * gy / mass, U0 * gz / mass
        c = 0
        ev = 0
        ic = 0
        alive = True
        rb = np.zeros(nb)
        for it in range(n_steps):
            if alive:
                # velocity Verlet
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                vz += 0.5 * dt * az
                x += dt * vx
                y += dt * vy
                z += dt * vz
                u, gx, gy, gz = _u_and_grad(x, y, z, w0, zr)
                ax, ay, az = U0 * gx / mass, U0 * gy / mass, U0 * gz / mass
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                vz += 0.5 * dt * az
                if (x * x + y * y) > 9 * w0 * w0 or abs(z) > 3 * zr:
                    alive = False
                    t_dead[m] = (it + 1) * dt
                else:
                    shift = delta0 + kappa_s * (1.0 - u)
                    rtot = 0.0
                    for b in range(nb):
                        dop = kwave * (beams[b, 0] * vx + beams[b, 1] * vy + beams[b, 2] * vz) / (2 * np.pi) * 1e-6
                        rb[b] = fbeam[b] * _interp2(xg0, dxg, ug0, dug, Rg, shift - dop, u)
                        rtot += rb[b]
                    if np.random.random() < rtot * dt:
                        # choose beam
                        r = np.random.random() * rtot
                        bsel = 0
                        acc = rb[0]
                        while acc < r and bsel < nb - 1:
                            bsel += 1
                            acc += rb[bsel]
                        # emission direction (isotropic)
                        cth = 2 * np.random.random() - 1
                        sth = np.sqrt(1 - cth * cth)
                        ph = 2 * np.pi * np.random.random()
                        vx += vrec * (beams[bsel, 0] + sth * np.cos(ph))
                        vy += vrec * (beams[bsel, 1] + sth * np.sin(ph))
                        vz += vrec * (beams[bsel, 2] + cth)
                        ev += 1
                        if np.random.random() < b00:
                            c += 1
                        if np.random.random() < p_trap * u + b_loss:
                            alive = False
                            t_dead[m] = (it + 1) * dt
            if ic < n_ck and it + 1 == ck_steps[ic]:
                counts[m, ic] = c
                events[m, ic] = ev
                ic += 1
        e_final[m] = 0.5 * mass * (vx * vx + vy * vy + vz * vz) - U0 * u
    return counts, events, t_dead, e_final


def run_mc(lookup_delta, lookup_u, lookup_R, t_ck_us, n_mol=4000, U0_mK=1.5, w0_um=1.0, lam_tw_nm=780.0,
           T0_uK=40.0, delta0=0.0, kappa_s_MHz_per_mK=5.0, beams=((1, 0, 0), (-1, 0, 0)),
           fbeam=None, p_trap=1e-4, b_loss=k.B_LOSS, b00=k.B00, dt_ns=None, seed=1):
    """Run the trajectory Monte-Carlo.

    lookup_delta, lookup_u: uniform grids of detuning (MHz) and local intensity u = U(r)/U0;
    lookup_R[u, delta]: total event rate [1/s] for all beams together. Returns a dict with `n606` and `nev`,
    both (n_mol, n_ck) cumulative counts at the checkpoint times t_ck_us, and
    `t_dead` (us, -1 if the molecule survived).
    """
    U0 = U0_mK * 1e-3 * k.KB
    w0 = w0_um * 1e-6
    zr = np.pi * w0 ** 2 / (lam_tw_nm * 1e-9)
    beams = np.asarray(beams, float)
    beams /= np.linalg.norm(beams, axis=1)[:, None]
    fbeam = np.ones(len(beams)) / len(beams) if fbeam is None else np.asarray(fbeam, float)
    rmax = float(np.max(lookup_R))
    dt = (dt_ns * 1e-9) if dt_ns else min(20e-9, 0.1 / max(rmax, 1.0))
    t_ck = np.asarray(t_ck_us, float) * 1e-6
    n_steps = int(np.ceil(t_ck[-1] / dt))
    ck_steps = np.maximum(1, np.round(t_ck / dt).astype(np.int64))
    ck_steps[-1] = n_steps
    dg = np.asarray(lookup_delta, float)
    ug = np.asarray(lookup_u, float)
    counts, events, t_dead, e_final = _run(
        n_mol, U0, w0, zr, k.M_CAF, T0_uK * 1e-6, dg[0], dg[1] - dg[0], ug[0], ug[1] - ug[0],
        np.ascontiguousarray(lookup_R, float),
        beams, fbeam, float(delta0), float(kappa_s_MHz_per_mK * U0_mK), float(p_trap), float(b_loss),
        float(b00), dt, n_steps, ck_steps, k.HBAR * k.K_WAVE / k.M_CAF, k.K_WAVE, seed)
    t_dead = np.where(t_dead > 0, t_dead * 1e6, -1.0)
    return {"n606": counts, "nev": events, "t_dead_us": t_dead, "E_final_uK": e_final / k.KB * 1e6,
            "t_ck_us": np.asarray(t_ck_us, float), "dt_ns": dt * 1e9}

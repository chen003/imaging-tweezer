"""Back-of-envelope numbers quoted in PLAN.md (not the full simulation).

1. Diagonalises the CaF X(v=0) N=0-3 hyperfine + Zeeman Hamiltonian
   (uncoupled |N mN; mS; mI> basis) and prints N=0 / N=1 level shifts at 1 G.
2. Evaluates the simple scaling formulas for heating and N=0 cross talk.
3. Draws docs/figures/scales.png (Zeeman map + ladder of frequency scales).

Run:  python scripts/plan_estimates.py
"""
import os

import numpy as np
from sympy.physics.wigner import clebsch_gordan, wigner_3j

# ---------------------------------------------------------------- constants
# X2Sigma+(v=0) [MHz], Childs, Goodman & Goodman, Phys. Rev. A (1981)
B_ROT, GAMMA_SR, B_HF, C_HF, C_I = 10267.54, 39.65891, 109.1839, 40.1190, 2.876e-2
MU_B = 1.39962449  # MHz/G
G_S, G_I = 2.00231930, 5.25774  # electron, 19F nuclear g
MU_N = MU_B / 1836.15267
# A2Pi1/2(v=0)
TAU_A = 19.2e-9  # s, Wall et al. PRA 78, 062509 (2008)
GAMMA_A = 1 / TAU_A  # s^-1
LAMBDA_AX = 606.3e-9  # m
LAMBDA_DOUBLING_J12 = 1.36e3  # MHz, |p+2q| for A(J=1/2); sign/value to be verified
M_CAF = 59.08 * 1.66053907e-27  # kg
H, HBAR, KB, C = 6.62607015e-34, 1.054571817e-34, 1.380649e-23, 2.99792458e8


# ---------------------------------------------------------------- X-state H
def _jmats(j):
    m = np.arange(j, -j - 1, -1)
    jp = np.zeros((len(m), len(m)))
    for i in range(1, len(m)):
        jp[i - 1, i] = np.sqrt(j * (j + 1) - m[i] * (m[i] + 1))
    return {1: -jp / np.sqrt(2), 0: np.diag(m), -1: jp.T / np.sqrt(2)}


def x_state_hamiltonian(n_max=3):
    """Return (H0, HZ per gauss, basis, N-operator diag) in MHz."""
    basis = [(n, mn, ms, mi) for n in range(n_max + 1) for mn in range(n, -n - 1, -1)
             for ms in (0.5, -0.5) for mi in (0.5, -0.5)]
    dim = len(basis)
    s_half = _jmats(0.5)
    half_idx = {0.5: 0, -0.5: 1}

    def build(elem):
        m = np.zeros((dim, dim))
        for i, a in enumerate(basis):
            for j, b in enumerate(basis):
                m[i, j] = elem(a, b)
        return m

    def s_elem(q, a, b):
        same = a[0] == b[0] and a[1] == b[1] and a[3] == b[3]
        return s_half[q][half_idx[a[2]], half_idx[b[2]]] if same else 0.0

    def i_elem(q, a, b):
        same = a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
        return s_half[q][half_idx[a[3]], half_idx[b[3]]] if same else 0.0

    def n_elem(q, a, b):
        if a[0] != b[0] or a[2] != b[2] or a[3] != b[3]:
            return 0.0
        ms = list(np.arange(a[0], -a[0] - 1, -1))
        return _jmats(a[0])[q][ms.index(a[1]), ms.index(b[1])]

    def c2_elem(q, a, b):
        if a[2] != b[2] or a[3] != b[3]:
            return 0.0
        n1, m1, n2, m2 = a[0], a[1], b[0], b[1]
        return float((-1) ** m1 * np.sqrt((2 * n1 + 1) * (2 * n2 + 1))
                     * wigner_3j(n1, 2, n2, 0, 0, 0) * wigner_3j(n1, 2, n2, -m1, q, m2))

    S = {q: build(lambda a, b, q=q: s_elem(q, a, b)) for q in (1, 0, -1)}
    I = {q: build(lambda a, b, q=q: i_elem(q, a, b)) for q in (1, 0, -1)}
    N = {q: build(lambda a, b, q=q: n_elem(q, a, b)) for q in (1, 0, -1)}
    C2 = {q: build(lambda a, b, q=q: c2_elem(q, a, b)) for q in range(-2, 3)}
    dot = lambda A, B: sum((-1) ** q * A[q] @ B[-q] for q in (1, 0, -1))
    IS2 = {q: sum(float(clebsch_gordan(1, 1, 2, q1, q - q1, q)) * I[q1] @ S[q - q1]
                  for q1 in (1, 0, -1) if abs(q - q1) <= 1) for q in range(-2, 3)}
    # H_hf = b_F I.S + c (I.n)(S.n) - (c/3) I.S ; rank-2 part = c sqrt(2/3) [I x S]^2 . C^2
    h_dip = C_HF * np.sqrt(2 / 3) * sum((-1) ** q * IS2[q] @ C2[-q] for q in range(-2, 3))
    n_sq = np.diag([b[0] * (b[0] + 1) for b in basis])
    h0 = (B_ROT * n_sq + GAMMA_SR * dot(N, S) + (B_HF + C_HF / 3) * dot(I, S)
          + h_dip + C_I * dot(N, I))
    hz = G_S * MU_B * S[0] - G_I * MU_N * I[0]
    fz = N[0] + S[0] + I[0]
    return np.real(h0), np.real(hz), fz, np.array([b[0] for b in basis])


def levels(h0, hz, fz, nvals, b_gauss):
    e, v = np.linalg.eigh(h0 + b_gauss * hz)
    mf = np.einsum("ij,jk,ki->i", v.T, fz, v)
    nexp = np.einsum("ij,j,ji->i", v.T, nvals, v)
    return e, np.round(mf).astype(int), np.round(nexp).astype(int)


# ---------------------------------------------------------------- estimates
def main():
    h0, hz, fz, nvals = x_state_hamiltonian()
    e0, mf0, n0 = levels(h0, hz, fz, nvals, 1e-6)
    e1, _, _ = levels(h0, hz, fz, nvals, 1.0)
    print("CaF X(v=0) levels, B=0 energy and shift at B = 1 G [MHz]")
    for nsel in (0, 1):
        sel = np.where(n0 == nsel)[0]
        ref = e0[sel].min()
        for i in sel:
            print(f"  N={nsel}  E0 = {e0[i] - ref:8.3f}  mF = {mf0[i]:+d}  dE(1 G) = {e1[i] - e0[i]:+.4f}")
    two_b = np.mean(e0[n0 == 1]) - np.mean(e0[n0 == 0])

    gamma_mhz = GAMMA_A / (2 * np.pi) / 1e6
    i_sat = np.pi * H * C * GAMMA_A / (3 * LAMBDA_AX ** 3) / 10  # mW/cm^2
    e_rec = (H / LAMBDA_AX) ** 2 / (2 * M_CAF) / KB  # K
    print(f"\nGamma/2pi = {gamma_mhz:.2f} MHz,  I_sat(2-level) = {i_sat:.2f} mW/cm^2")
    print(f"E_rec/kB = {e_rec * 1e6:.3f} uK -> heating per scattered photon ~ 2 E_rec = {2 * e_rec * 1e6:.2f} uK")
    for u_mk in (0.5, 1, 2, 4):
        print(f"  depth {u_mk:3.1f} mK : ~{u_mk * 1e-3 / (2 * e_rec):6.0f} photons to boil out (no cooling)")

    for label, d0 in (("A(J'=1/2,-) below A(+)", two_b - LAMBDA_DOUBLING_J12),
                      ("A(J'=1/2,-) above A(+)", two_b + LAMBDA_DOUBLING_J12)):
        eps = (gamma_mhz / (2 * d0)) ** 2
        print(f"\nN=0 detuning Delta0 = {d0 / 1e3:.2f} GHz  [{label}]  (Gamma/2Delta0)^2 = {eps:.2e}")
        # scattering on N=0 per photon scattered on N=1:  p = a (Gamma/2Delta0)^2 (Gamma/2) s_tot / R1
        for s_tot, r1 in ((1, 1.5e6), (3, 2.5e6), (10, 3.5e6), (30, 4.5e6)):
            r0 = (GAMMA_A / 2) * s_tot * eps / 3  # a = 1/3 angular factor (placeholder)
            dls = (gamma_mhz * 1e3) ** 2 * s_tot / (8 * d0 * 1e3)  # kHz, common shift
            print(f"   s_tot={s_tot:3d}: R0 = {r0:6.2f} /s, R1 = {r1:.1e} /s, "
                  f"p/photon = {r0 / r1:.1e}, 500 photons -> {500 * r0 / r1:.1e}; "
                  f"light shift ~{dls:.1f} kHz, differential ~{dls * 122.6 / d0 * 1e3:.0f} Hz")

    make_figure(h0, hz, fz, nvals, gamma_mhz, two_b)


def make_figure(h0, hz, fz, nvals, gamma_mhz, two_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})

    bs = np.linspace(0, 3, 121)
    e_ref, mf_ref, n_ref = levels(h0, hz, fz, nvals, 1e-6)
    sel = np.where(n_ref == 1)[0]
    ref = e_ref[sel].min()
    # label N=1 manifolds by zero-field energy
    manifolds = {0: "J=1/2, F=1", 76: "J=1/2, F=0", 123: "J=3/2, F=1", 148: "J=3/2, F=2"}
    traj = np.array([levels(h0, hz, fz, nvals, b)[0][sel] - ref for b in bs])
    drawn = set()
    for k in range(len(sel)):
        key = min(manifolds, key=lambda x: abs(x - traj[0, k]))
        idx = list(manifolds).index(key)
        ax1.plot(bs, traj[:, k] - traj[0, k], color=series[idx], lw=2,
                 label=None if key in drawn else manifolds[key])
        drawn.add(key)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False, fontsize=8.5, title="N=1 hyperfine level",
               title_fontsize=8.5)
    ax1.axvline(1.0, color=ink2, lw=1, ls=":")
    ax1.axhspan(-gamma_mhz / 2, gamma_mhz / 2, color=grid, zorder=0)
    ax1.text(0.05, gamma_mhz / 2 + 0.25, "shaded: ±Γ/2  (natural linewidth Γ/2π = 8.3 MHz)", color=ink2, fontsize=8.5)
    ax1.text(1.04, -4.85, "1 G", color=ink2, fontsize=8.5)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel("magnetic field B (G)")
    ax1.set_ylabel("Zeeman shift within each hyperfine level (MHz)")
    ax1.set_title("X(N=1) sublevels: 1 G shifts are ≤ 0.17 Γ", color=ink, loc="left", fontsize=11)

    scales = [("Zeeman, N=1 at 1 G (max)", 1.40), ("natural linewidth Γ/2π", gamma_mhz),
              ("N=1 hyperfine spacings", 25), ("N=0 qubit splitting F=0↔1", 122.6),
              ("A(J'=1/2) Λ-doubling", 1360), ("imaging light → N=0 line  (≈2B−Λ)", (two_b - 1360)),
              ("rotational 2B  (N=0↔N=1)", two_b)]
    names = [s[0] for s in scales][::-1]
    vals = [s[1] for s in scales][::-1]
    ax2.barh(names, vals, color=series[0], height=0.55)
    for y, v in enumerate(vals):
        txt = f"{v / 1e3:.1f} GHz" if v >= 1e3 else f"{v:.1f} MHz"
        ax2.text(v * 1.25, y, txt, va="center", color=ink, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlim(0.5, 3e5)
    ax2.set_xlabel("frequency (MHz, log scale)")
    ax2.grid(axis="x", color=grid, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("Hierarchy of scales: 1 G ≪ Γ ≪ 2B", color=ink, loc="left", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "scales.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()

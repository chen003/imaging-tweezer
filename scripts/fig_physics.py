"""Physics figures from the OBE and Raman models (no Monte-Carlo).

docs/figures/scattering_map.png  R1 vs (B, polarisation angle) and vs B for selected angles
docs/figures/efficiency.png      photons per unit fluence vs saturation, and Raman cost per photon
docs/figures/raman_breakdown.png where the N=0 off-resonant scattering goes, per N=0 state
Also writes results/physics.json with the plotted numbers.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from caf.lookup import Geometry, Model, mk_to_mhz  # noqa: E402
from caf.obe import N1Imaging  # noqa: E402
from caf.raman import N0Raman  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "docs", "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

U0 = 1.5  # mK
WEIGHTS = (2, 1, 2, 3)


def img_eps(chi_deg):
    c = np.radians(chi_deg)
    return (np.cos(c), 0, np.sin(c))


def rate(model, b, chi, s=1.0, delta=0.0, tensor=True, b_dir=(1, 0, 0)):
    g = Geometry(b_gauss=b, b_dir=b_dir, eps_img=img_eps(chi))
    sysm = N1Imaging(model.x, model.a, b_gauss=max(b, 1e-3),
                     u_tensor=model.kappa_t * mk_to_mhz(U0) if tensor else 0.0,
                     eps_tw=np.real(g.in_b_frame(g.eps_tw)))
    return sysm.scattering_rate(s, delta, eps=g.in_b_frame(g.eps_img), weights=WEIGHTS, offsets=g.offsets)


def main():
    model = Model()
    out = {}
    # ---------------------------------------------------------------- map R(B, chi)
    bs = np.linspace(0, 3, 31)
    chis = np.arange(0, 91, 5)
    rmap = np.array([[rate(model, b, c) for b in bs] for c in chis]) / 1e6
    out["map"] = {"B": bs.tolist(), "chi": chis.tolist(), "R": rmap.tolist()}
    lines = {}
    for c in (50, 90, 0):
        lines[f"chi{c}"] = [rate(model, b, c) / 1e6 for b in bs]
    lines["chi50_notensor"] = [rate(model, b, 50, tensor=False) / 1e6 for b in bs]
    out["lines"] = lines

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    im = a1.pcolormesh(bs, chis, rmap, cmap="Blues", shading="nearest", vmin=0)
    cb = fig.colorbar(im, ax=a1, pad=0.02)
    cb.set_label("photon scattering rate R₁ (10⁶ s⁻¹)")
    a1.axvline(1.0, color="#ffffff", lw=1, ls=":")
    a1.set_xlabel("magnetic field |B| (G), B ∥ tweezer polarisation")
    a1.set_ylabel("angle χ between imaging and tweezer polarisation (deg)")
    a1.set_title("R₁ at s_tot = 1, Δ = 0, U₀ = 1.5 mK", loc="left", fontsize=11, color=INK)
    a2.plot(bs, lines["chi50"], color=SER[0], lw=2, label="χ = 50°, in 1.5 mK tweezer")
    a2.plot(bs, lines["chi90"], color=SER[1], lw=2, label="χ = 90°, in 1.5 mK tweezer")
    a2.plot(bs, lines["chi0"], color=SER[3], lw=2, label="χ = 0°, in 1.5 mK tweezer")
    a2.plot(bs, lines["chi50_notensor"], color=SER[2], lw=2, ls="--", label="χ = 50°, no tensor shift")
    a2.axvline(1.0, color=INK2, lw=1, ls=":")
    a2.set_xlabel("magnetic field |B| (G)")
    a2.set_ylabel("R₁ (10⁶ s⁻¹)")
    a2.set_ylim(0, None)
    a2.grid(color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=8.5, loc="lower right")
    a2.set_title("1 G is enough if the polarisations are not aligned", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scattering_map.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- efficiency vs s
    ss = np.geomspace(0.1, 30, 16)
    ram = N0Raman(model.x, model.a, b_gauss=1.0, eps_tw=(0, 0, 1))
    eff = {}
    for name, b in (("B0", 0.0), ("B1", 1.0)):
        eff[name] = [rate(model, b, 50, s=s) / s for s in ss]
    # Raman per unit s_tot (worst N=0 state) for the same polarisation, B=1 G
    g = Geometry(b_gauss=1.0, eps_img=img_eps(50))
    centres = N1Imaging(model.x, model.a, b_gauss=1.0, u_tensor=model.kappa_t * mk_to_mhz(U0),
                        eps_tw=np.real(g.in_b_frame(g.eps_tw))).centres
    w = np.asarray(WEIGHTS, float) / np.sum(WEIGHTS)
    rr = ram.rates(w, 0.0, g.in_b_frame(g.eps_img), centres, offsets=g.offsets)["raman"].max()
    out["efficiency"] = {"s": ss.tolist(), **eff, "raman_per_s": rr}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for i, (name, lab) in enumerate((("B1", "B = 1 G"), ("B0", "B = 0"))):
        a1.plot(ss, np.array(eff[name]) / 1e6, color=SER[i], lw=2, marker="o", ms=4, label=lab)
        a2.plot(ss, rr / np.array(eff[name]), color=SER[i], lw=2, marker="o", ms=4, label=lab)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("total saturation parameter s_tot")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    a1.set_ylabel("R₁ / s_tot  (10⁶ photons s⁻¹)")
    a1.set_title("Photons per unit fluence (χ = 50°, Δ = 0)", loc="left", fontsize=11)
    a2.set_yscale("log")
    a2.set_ylabel("N=0 Raman probability per N=1 photon")
    a2.set_title("Cross-talk cost of each imaging photon", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "efficiency.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- Raman breakdown
    brk = {}
    for name, b in (("B0", 1e-3), ("B1", 1.0)):
        r = N0Raman(model.x, model.a, b_gauss=b, eps_tw=np.real(g.in_b_frame(g.eps_tw)))
        o = r.rates(w, 0.0, g.in_b_frame(g.eps_img), centres, offsets=g.offsets)
        other = o["to_N0"].sum(1) - np.diag(o["to_N0"])
        brk[name] = {"mf": np.round(r.m0.mf).tolist(), "F": r.m0.hf_label.tolist(),
                     "within_N0": other.tolist(), "to_N2": o["to_N2"].sum(1).tolist(),
                     "to_v1": o["to_v1"].tolist(), "rayleigh": o["rayleigh"].tolist()}
    out["raman"] = brk
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    labels, x0 = [], 0
    for name, lab in (("B0", "B = 0"), ("B1", "B = 1 G")):
        d = brk[name]
        order = np.lexsort((d["mf"], d["F"]))
        for j in order:
            bottom = 0
            for key, col, lg in (("within_N0", SER[0], "Raman within N=0 (F or m_F change)"),
                                 ("to_N2", SER[1], "Raman to N=2"), ("to_v1", SER[3], "to v≥1")):
                ax.bar(x0, d[key][j], bottom=bottom, color=col, width=0.7, edgecolor="#fcfcfb", lw=1,
                       label=lg if x0 == 0 else None)
                bottom += d[key][j]
            labels.append(f"{lab}\nF={d['F'][j]}, m_F={int(d['mf'][j]):+d}")
            x0 += 1
        x0 += 0.6
    ax.set_xticks([i + (0.6 if i >= 4 else 0) for i in range(8)])
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("rate per unit s_tot (s⁻¹)")
    ax.set_title("State-changing scattering of N=0 by the imaging light (s_tot = 1, χ = 50°)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_ylim(0, 0.3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "raman_breakdown.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(ROOT, "results", "physics.json"), "w") as f:
        json.dump(out, f)
    print("Raman per unit s (worst N=0 state):", rr)
    print("efficiency B1:", np.round(np.array(eff["B1"]) / 1e6, 3))
    print("efficiency B0:", np.round(np.array(eff["B0"]) / 1e6, 3))


if __name__ == "__main__":
    main()

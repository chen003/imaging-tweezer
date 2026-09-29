"""Model vs Holland et al. (arXiv:2406.02391) at their rapid-imaging settings, plus the 1 G question.

Writes results/paper_validation.json and docs/figures/paper_bscan.png.
Run: PYTHONPATH=. python scripts/paper_validate.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from caf import paper  # noqa: E402
from caf.camera import emccd_eps, emccd_sigma_for  # noqa: E402
from caf.ensemble import EnsembleLookup  # noqa: E402
from caf.motion import run_mc  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
EPS_TW = (1.0, 0.0, 0.0)
KAPPA_T = 0.559  # N=1 differential ac Stark shift RMS = 10% of U (paper, App. E)
SIGMA = emccd_sigma_for(paper.EPS01, paper.THRESHOLD)
BEAMS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))


def mc_config(ens, cfg, samples, s_tot=paper.S_TOT, delta0=0.0, offsets=(0, 0, 0, 0), tck=None, n_mol=6000,
              t0=paper.T0_UK, kappa_s=5.0, p_trap=4e-5 * 0.93 / 1.5, seed=5):
    bmag, th = paper.CONFIGS[cfg]
    d, u, R, Rs = ens.table(bmag, paper.b_dir(th), EPS_TW, samples, s_tot, paper.U_IMG_MK, paper.WEIGHTS, offsets)
    tck = np.geomspace(20, paper.T_IMG_US, 30) if tck is None else tck
    out = run_mc(d, u, R, tck, n_mol=n_mol, U0_mK=paper.U_IMG_MK, w0_um=paper.W0_UM, lam_tw_nm=paper.LAMBDA_TW_NM,
                 T0_uK=t0, delta0=delta0, kappa_s_MHz_per_mK=kappa_s, beams=BEAMS, p_trap=p_trap, seed=seed)
    return out, (d, u, R, Rs)


def main():
    ens = EnsembleLookup(KAPPA_T)
    samples = paper.field_samples(24, seed=3)
    res = {"sigma_emccd": SIGMA, "configs": {}}
    # ------------------------------------------------------------ 1. paper configurations
    for cfg in paper.CONFIGS:
        for off_name, off in (("exact", (0, 0, 0, 0)), ("offset", (1, -1, 2, -2))):
            out, (d, u, R, Rs) = mc_config(ens, cfg, samples, offsets=off)
            n_end = out["n606"][:, -1]
            alive = out["t_dead_us"] < 0
            r_bottom = float(R[-1, np.argmin(np.abs(d))])
            e10, e01 = emccd_eps(n_end, paper.ETA_PAPER, 0.0, SIGMA, [paper.THRESHOLD])
            row = {"R_trap_bottom": r_bottom,
                   "R_mean_first_ms": float(out["nev"][:, np.argmin(np.abs(out["t_ck_us"] - 1000))].mean() / 1e-3),
                   "photons_scattered_3ms": float(out["nev"][:, -1].mean()),
                   "photons_collected_eta5": float(n_end.mean() * paper.ETA_PAPER),
                   "photons_collected_eta5_survivors": float(n_end[alive].mean() * paper.ETA_PAPER) if alive.any() else 0,
                   "survival_3ms": float(alive.mean()), "eps10_emccd_eta5": float(e10[0]), "eps01_emccd": float(e01[0])}
            for eta in (0.02, 0.04):
                ths = np.linspace(0.5, 12, 116)
                e10s, e01s = emccd_eps(n_end, eta, 0.0, SIGMA, ths)
                i48 = int(np.argmin(np.abs(ths - paper.THRESHOLD)))
                row[f"eps10_emccd_eta{int(eta * 100)}_th4.8"] = float(e10s[i48])
                row[f"best_F_emccd_eta{int(eta * 100)}"] = float(np.max(1 - 0.5 * (e10s + e01s)))
            ram = ens.raman(*paper.CONFIGS[cfg][:1], paper.b_dir(paper.CONFIGS[cfg][1]), EPS_TW, samples,
                            paper.WEIGHTS, off)
            row["raman_per_s"] = ram["raman"].tolist()
            row["total_per_s"] = ram["total"].tolist()
            row["n0_mf"] = ram["mf"].tolist()
            row["n0_F"] = ram["F"].tolist()
            res["configs"][f"{cfg}|{off_name}"] = row
            print(cfg, off_name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                  if not isinstance(v, list)}, flush=True)
            print("    N=0 Raman per unit s_tot (F,mF):", list(zip(ram["F"].tolist(), ram["mf"].tolist(),
                                                                   np.round(ram["raman"], 3).tolist())), flush=True)
    # ------------------------------------------------------------ 2. B scan at trap bottom (steady state)
    bs = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4, 5.0])
    scan = {}
    pol_cases = {"random polarisations": None, "both beams vertical (along tweezer axis)": (np.pi / 2, np.pi / 2),
                 "both beams horizontal": (0.0, 0.0)}
    for th in (53.0, 90.0):
        for name, pol in pol_cases.items():
            smp = paper.field_samples(24, seed=3, pol_a=None if pol is None else pol[0],
                                      pol_b=None if pol is None else pol[1])
            vals = []
            for b in bs:
                d, u, R, Rs = ens.table(b, paper.b_dir(th), EPS_TW, smp, paper.S_TOT, paper.U_IMG_MK, paper.WEIGHTS,
                                        (0, 0, 0, 0), delta=np.array([0.0]), u=np.array([1.0]))
                vals.append(float(R[0, 0]))
            scan[f"{th:.0f}|{name}"] = vals
            print("B scan", th, name, np.round(np.array(vals) / 1e5, 2), flush=True)
    res["bscan"] = {"B": bs.tolist(), "R": scan}
    ens.close()
    with open(os.path.join(ROOT, "results", "paper_validation.json"), "w") as f:
        json.dump(res, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, th in zip(axes, (53, 90)):
        for i, name in enumerate(pol_cases):
            ax.plot(bs, np.array(scan[f"{th}|{name}"]) / 1e5, color=SER[i], lw=2, marker="o", ms=4, label=name)
        for bb, lab in ((1.0, "1 G"), (4.4 if th == 53 else 2.0, "paper")):
            ax.axvline(bb, color=INK2, lw=1, ls=":")
            ax.text(bb + 0.05, 2.95, lab, color=INK2, fontsize=8.5)
        ax.axhline(paper.R_SCAT_MEASURED / 1e5, color=SER[3], lw=1.5, ls="--")
        ax.text(0.1, paper.R_SCAT_MEASURED / 1e5 - 0.22, "measured at 4.4 G, 53°: 1.6×10⁵ s⁻¹",
                color=INK2, fontsize=8.5)
        ax.set_xlabel("|B| (G)")
        ax.set_title(f"B at {th}° from tweezer polarisation", loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("R₁ at trap bottom (10⁵ s⁻¹)")
    axes[0].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["imaging beams: " + x for x in l], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Paper settings (I = 4.5 mW/cm², powers 5:2.7:1.1:3.8, 930 µK): photon rate vs field",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(ROOT, "docs", "figures", "paper_bscan.png"), dpi=150)


if __name__ == "__main__":
    main()

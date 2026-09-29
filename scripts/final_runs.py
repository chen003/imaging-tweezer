"""High-statistics runs at the optimum found by analyze_scan.py.

1. Histograms at the optimum for B = 0 and B = 1 G, for eta = 2% and 4%
   (docs/figures/histograms.png).
2. Data for the interactive page: distributions of detectable photons N_606(T)
   for every s_tot at the best (U0, delta0) of each geometry
   (results/artifact_data.json).
3. One-at-a-time sensitivity of the optimum P_R to the model's uncertain inputs
   (results/sensitivity.json).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from caf.camera import background_mean, bright_pmf, dark_pmf, fidelity  # noqa: E402
from caf.lookup import Geometry, Model  # noqa: E402
from caf.motion import run_mc  # noqa: E402
from run_scan import GEOMS  # noqa: E402
from analyze_scan import t_at  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
P_TRAP = 4e-5
KAPPA_S = 5.0
ZETA = 1e-3


def mc(model, geom, s, u0, d0, tck, n_mol, kappa_s=KAPPA_S, p_trap=P_TRAP, seed=7, lk=None):
    lk = model.lookup(geom, s, u0) if lk is None else lk
    out = run_mc(lk.delta, lk.u, lk.R_tot, tck, n_mol=n_mol, U0_mK=u0, delta0=d0, kappa_s_MHz_per_mK=kappa_s,
                 beams=geom.beams, p_trap=p_trap * u0 / 1.5, seed=seed)
    return out, lk


def f_curve(out, s, eta, zeta=ZETA):
    fs = []
    for i, tt in enumerate(out["t_ck_us"]):
        bg = background_mean(s, tt, eta, zeta=zeta)
        pb = bright_pmf(out["n606"][:, i], eta, bg)
        fs.append(fidelity(pb, dark_pmf(bg, len(pb) - 1))[0])
    return np.array(fs)


def p_r_at(out, lk, s, eta, target=0.99, zeta=ZETA):
    tt = t_at(out["t_ck_us"], f_curve(out, s, eta, zeta), target)
    return (None, None) if tt is None else (max(lk.raman_per_s) * s * tt * 1e-6, tt)


def main():
    model = Model()
    opt = json.load(open(os.path.join(ROOT, "results", "optimum.json")))["best"]
    scan = json.load(open(os.path.join(ROOT, "results", "scan.json")))["records"]

    # ---------------------------------------------------------------- 1. histograms
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex="col")
    hist_out = {}
    for col, eta in enumerate((0.02, 0.04)):
        for row, g in enumerate(("B0", "B1_par_tw")):
            ax = axes[row, col]
            b = opt.get(f"{g}|eta={eta}|F=0.99")
            label = "F = 99%"
            if b is None:  # fall back to the best-fidelity point
                cand = [r for r in scan if r["geom"] == g]
                r = max(cand, key=lambda r: max(r[f"F_{eta}"]))
                i = int(np.argmax(r[f"F_{eta}"]))
                b = {"U0": r["U0"], "s": r["s"], "d0": r["d0"], "T_us": r["t"][i]}
                label = "best reachable"
            out, lk = mc(model, GEOMS[g], b["s"], b["U0"], b["d0"], np.array([b["T_us"]]), 20000)
            bg = background_mean(b["s"], b["T_us"], eta)
            pb = bright_pmf(out["n606"][:, 0], eta, bg)
            pd = dark_pmf(bg, len(pb) - 1)
            F, nth, eb, ed = fidelity(pb, pd)
            n = np.arange(len(pb))
            ax.bar(n, pd * 0.5, width=0.9, color="#b7b6b0", label="empty / N=0 (dark)")
            ax.bar(n, pb * 0.5, width=0.9, color=SER[0], alpha=0.85, label="N=1 molecule (bright)")
            ax.axvline(nth + 0.5, color=SER[1], lw=1.5, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_xlim(-0.5, max(12, np.searchsorted(np.cumsum(pb), 0.999) + 2))
            pr = max(lk.raman_per_s) * b["s"] * b["T_us"] * 1e-6
            ax.set_title(f"{GEOMS[g].name}, η = {eta:.0%} ({label})", loc="left", fontsize=10)
            ax.text(0.98, 0.95, f"s_tot = {b['s']}, Δ = {b['d0']} MHz, U₀ = {b['U0']} mK\n"
                    f"T = {b['T_us']:.0f} µs, F = {F:.4f}\nN=0 Raman P = {pr:.1e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("detected photons in ROI")
            if col == 0:
                ax.set_ylabel("probability (50/50 prior)")
            hist_out[f"{g}|{eta}"] = {**b, "F": F, "n_th": int(nth), "eps_bright": eb, "eps_dark": ed, "P_R": pr,
                                      "bg": bg, "p_bright": pb.tolist(), "p_dark": pd.tolist(),
                                      "survival": float(np.mean(out["t_dead_us"] < 0))}
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "histograms.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 2. artifact data
    art = {"meta": {"zeta": ZETA, "gamma_half": 0.5 * 5.2083e7, "p_trap": P_TRAP, "kappa_s": KAPPA_S}, "runs": []}
    s_vals = sorted({r["s"] for r in scan})
    for g in ("B0", "B1_par_tw"):
        for s in s_vals:
            cand = [r for r in scan if r["geom"] == g and r["s"] == s]
            # best (U0, d0) for this s: smallest T to F=99% at eta=2%, else highest max F
            def key(r):
                tt = t_at(r["t"], r["F_0.02"], 0.99)
                return (0, tt) if tt is not None else (1, -max(r["F_0.02"]))
            r = min(cand, key=key)
            tck = np.geomspace(10, r["t"][-1], 36)
            out, lk = mc(model, GEOMS[g], s, r["U0"], r["d0"], tck, 6000, seed=11)
            n606 = out["n606"]
            edges = np.unique(np.round(np.geomspace(1, max(10, n606.max() + 2), 120)).astype(int))
            edges = np.concatenate([[0], edges])
            hists = [np.histogram(n606[:, i], bins=edges)[0].tolist() for i in range(len(tck))]
            art["runs"].append({"geom": g, "s": s, "U0": r["U0"], "d0": r["d0"], "t": tck.tolist(),
                                "edges": edges.tolist(), "hist": hists, "n_mol": int(n606.shape[0]),
                                "raman_per_s": max(lk.raman_per_s),
                                "surv": [float(np.mean((out["t_dead_us"] < 0) | (out["t_dead_us"] > tt))) for tt in tck]})
            print("artifact run", g, s, r["U0"], r["d0"], flush=True)
    with open(os.path.join(ROOT, "results", "artifact_data.json"), "w") as f:
        json.dump(art, f)

    # ---------------------------------------------------------------- 3. sensitivity
    sens = {}
    for eta in (0.02, 0.04):
        b = opt.get(f"B1_par_tw|eta={eta}|F=0.99")
        if b is None:
            continue
        geom = GEOMS["B1_par_tw"]
        tck = np.geomspace(10, 4 * b["T_us"], 50)
        base_out, lk = mc(model, geom, b["s"], b["U0"], b["d0"], tck, 6000, seed=21)
        rows = {"baseline": p_r_at(base_out, lk, b["s"], eta)}
        for name, kw in (("kappa_s=0", {"kappa_s": 0.0}), ("kappa_s=15 MHz/mK", {"kappa_s": 15.0}),
                         ("p_trap=0", {"p_trap": 0.0}), ("p_trap=1e-4", {"p_trap": 1e-4})):
            o, _ = mc(model, geom, b["s"], b["U0"], b["d0"], tck, 6000, seed=21, lk=lk, **kw)
            rows[name] = p_r_at(o, lk, b["s"], eta)
        for z in (1e-4, 3e-3):
            rows[f"zeta={z}"] = p_r_at(base_out, lk, b["s"], eta, zeta=z)
        # tensor spread and B-direction variants need new lookups
        for name, geom2, m2 in (("tensor spread 10%", geom, Model(tensor_spread=0.1)),
                                ("B || tweezer axis", Geometry(b_gauss=1.0, b_dir=(0, 0, 1), eps_img=geom.eps_img), model)):
            o, lk2 = mc(m2, geom2, b["s"], b["U0"], b["d0"], tck, 6000, seed=21)
            rows[name] = p_r_at(o, lk2, b["s"], eta)
        sens[f"eta={eta}"] = {"point": b, "rows": rows}
        print("sensitivity", eta, rows, flush=True)
    with open(os.path.join(ROOT, "results", "sensitivity.json"), "w") as f:
        json.dump(sens, f, indent=1)


if __name__ == "__main__":
    main()

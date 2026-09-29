"""Optimise rapid resonant imaging at 1 G with the hardware of Holland et al. (arXiv:2406.02391).

Fixed: 781 nm tweezer, w0 = 730 nm, 930 uK imaging depth, T0 = 120 uK, two retro-reflected beams
(random-polarisation field ensemble), hyperfine power ratio 5:2.7:1.1:3.8, EMCCD noise calibrated to
eps01 = 7.2 % at threshold 4.8, no stray-light background (as reported).
Scanned: total saturation s_tot and common detuning delta0; imaging time from the checkpoints.

Criteria:
* "paper": EMCCD false-negative eps10 <= 3.3 % at threshold 4.8 (the paper's operating criterion);
* "F99": 99 % fidelity with an ideal photon-counting camera.

For each criterion, find the smallest intensity x time (hence the smallest N=0 Raman probability).

Writes results/paper_optimum.json and docs/figures/paper_pareto.png.
Run: OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/paper_optimize.py
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
from caf.camera import bright_pmf, dark_pmf, emccd_eps, emccd_sigma_for, fidelity  # noqa: E402
from caf.ensemble import EnsembleLookup  # noqa: E402
from caf.motion import run_mc  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
EPS_TW = (1.0, 0.0, 0.0)
KAPPA_T = 0.559
SIGMA = emccd_sigma_for(paper.EPS01, paper.THRESHOLD)
BEAMS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
S_GRID = [0.3, 0.6, 0.93, 1.5, 2.5, 4.0, 8.0]
D_GRID = [-4.0, -2.0, 0.0, 2.0]
CFGS = ["1G_53", "1G_90", "state_4.4G_53", "erasure_2.0G_90"]
ETAS = (0.02, 0.04)


def t_first(t, ok):
    idx = np.where(ok)[0]
    return float(t[idx[0]]) if len(idx) else None


def run_grid(cfgs, s_grid, d_grid, u0, out_path):
    ens = EnsembleLookup(KAPPA_T)
    samples = paper.field_samples(24, seed=3)
    recs = []
    raman = {}
    for cfg in cfgs:
        bmag, th = paper.CONFIGS[cfg]
        rm = ens.raman(bmag, paper.b_dir(th), EPS_TW, samples, paper.WEIGHTS)
        raman[cfg] = {"raman": rm["raman"].tolist(), "total": rm["total"].tolist(), "mf": rm["mf"].tolist(),
                      "F": rm["F"].tolist()}
        for s in s_grid:
            d, u, R, _ = ens.table(bmag, paper.b_dir(th), EPS_TW, samples, s, u0, paper.WEIGHTS)
            t_max = min(10e3, 2500 / max(R.max(), 1.0) * 1e6)
            tck = np.geomspace(30, t_max, 40)
            for d0 in d_grid:
                # T0 scales like sqrt(U0) for an adiabatic ramp from the same starting trap
                t0 = paper.T0_UK * np.sqrt(u0 / paper.U_IMG_MK)
                out = run_mc(d, u, R, tck, n_mol=4000, U0_mK=u0, w0_um=paper.W0_UM,
                             lam_tw_nm=paper.LAMBDA_TW_NM, T0_uK=t0, delta0=d0, kappa_s_MHz_per_mK=5.0,
                             beams=BEAMS, p_trap=4e-5 * u0 / 1.5, seed=int(100 * s + d0 + 17))
                rec = {"cfg": cfg, "U0": u0, "s": s, "d0": d0, "t": tck.tolist(),
                       "mean_ev": out["nev"].mean(0).tolist(),
                       "surv": [float(np.mean((out["t_dead_us"] < 0) | (out["t_dead_us"] > tt))) for tt in tck]}
                for eta in ETAS:
                    e10s, f99 = [], []
                    for i in range(len(tck)):
                        n = out["n606"][:, i]
                        e10, _ = emccd_eps(n, eta, 0.0, SIGMA, [paper.THRESHOLD])
                        e10s.append(float(e10[0]))
                        bg = 0.01 * tck[i] * 1e-3
                        pb = bright_pmf(n, eta, bg)
                        f99.append(float(fidelity(pb, dark_pmf(bg, len(pb) - 1))[0]))
                    rec[f"eps10_{eta}"] = e10s
                    rec[f"F_{eta}"] = f99
                recs.append(rec)
                tp = t_first(tck, np.array(rec["eps10_0.02"]) <= paper.EPS10["state"])
                print(f"{cfg} U0={u0} s={s} d0={d0}: R_bottom={R[-1, np.argmin(np.abs(d))] / 1e5:.2f}e5 "
                      f"T(eps10<=3.3%, eta2%)={tp}  min eps10(2%)={min(rec['eps10_0.02']):.4f} "
                      f"maxF99(2%)={max(rec['F_0.02']):.4f}", flush=True)
    ens.close()
    with open(out_path, "w") as f:
        json.dump({"raman": raman, "records": recs}, f)


def analyze(paths, u0_filter=None):
    recs, raman = [], {}
    for pth in paths:
        dat = json.load(open(pth))
        raman.update(dat["raman"])
        for r in dat["records"]:
            r.setdefault("U0", paper.U_IMG_MK)
            if u0_filter is None or abs(r["U0"] - u0_filter) < 1e-6:
                recs.append(r)
    best = {}
    cfgs = [c for c in CFGS if any(r["cfg"] == c for r in recs)]
    for cfg in cfgs:
        r_worst = max(raman[cfg]["raman"])
        for eta in ETAS:
            for crit in ("paper", "F99"):
                cands = []
                for r in recs:
                    if r["cfg"] != cfg:
                        continue
                    t = np.array(r["t"])
                    ok = (np.array(r[f"eps10_{eta}"]) <= paper.EPS10["state"]) if crit == "paper" \
                        else (np.array(r[f"F_{eta}"]) >= 0.99)
                    tt = t_first(t, ok)
                    if tt is None:
                        continue
                    i = int(np.argmin(np.abs(t - tt)))
                    cands.append({"U0": r["U0"], "s": r["s"], "d0": r["d0"], "T_us": tt, "fluence": r["s"] * tt,
                                  "P_R": r_worst * r["s"] * tt * 1e-6, "photons": r["mean_ev"][i],
                                  "survival": r["surv"][i]})
                key = f"{cfg}|eta={eta}|{crit}"
                best[key] = min(cands, key=lambda c: c["P_R"]) if cands else None
                fr = []
                for c in sorted(cands, key=lambda c: c["T_us"]):
                    if not fr or c["P_R"] < fr[-1]["P_R"]:
                        fr.append(c)
                best[key + "|front"] = fr
                b = best[key]
                print(key, "->", None if b is None else {k: round(v, 5) if isinstance(v, float) else v
                                                        for k, v in b.items()}, flush=True)
    # the paper's own operating point, evaluated in the model
    paper_pt = {}
    for cfg in cfgs:
        for eta in ETAS + (paper.ETA_PAPER,):
            sel = [x for x in recs if x["cfg"] == cfg and x["s"] == 0.93 and x["d0"] == 0.0
                   and abs(x["U0"] - paper.U_IMG_MK) < 1e-6]
            if not sel:
                continue
            r = sel[0]
            t = np.array(r["t"])
            i = int(np.argmin(np.abs(t - paper.T_IMG_US)))
            if eta in ETAS:
                paper_pt[f"{cfg}|eta={eta}"] = {"T_us": float(t[i]), "eps10": r[f"eps10_{eta}"][i],
                                                "F99": r[f"F_{eta}"][i], "photons": r["mean_ev"][i],
                                                "survival": r["surv"][i],
                                                "P_R": max(raman[cfg]["raman"]) * 0.93 * t[i] * 1e-6}
    tag = "" if u0_filter is None else f"_U{u0_filter:g}"
    with open(os.path.join(ROOT, "results", f"paper_optimum{tag}.json"), "w") as f:
        json.dump({"best": best, "raman": raman, "paper_point": paper_pt}, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    lab = {"1G_53": "1 G, 53°", "1G_90": "1 G, 90°", "state_4.4G_53": "4.4 G, 53° (paper)",
           "erasure_2.0G_90": "2.0 G, 90° (paper)"}
    for ax, eta in zip(axes, ETAS):
        for i, cfg in enumerate(cfgs):
            fr = best[f"{cfg}|eta={eta}|paper|front"]
            if fr:
                ax.plot([c["T_us"] for c in fr], [c["P_R"] for c in fr], color=SER[i], lw=2, marker="o", ms=4,
                        label=lab[cfg])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 300, 1000, 3000, 10000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("imaging time T for ε₁₀ ≤ 3.3% (EMCCD, threshold 4.8) (µs)")
        ax.set_title(f"η = {eta:.0%}", loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8, which="both")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("intrinsic N=0 Raman probability per image")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Paper hardware (930 µK, 781 nm, EMCCD): cross talk vs speed at the paper's detection "
                 "criterion", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", f"paper_pareto{tag}.png"), dpi=150)
    plt.close(fig)
    return best


if __name__ == "__main__":
    import argparse

    import matplotlib.ticker  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cfgs", default=",".join(CFGS))
    ap.add_argument("--s", default=",".join(map(str, S_GRID)))
    ap.add_argument("--d", default=",".join(map(str, D_GRID)))
    ap.add_argument("--u0", type=float, default=paper.U_IMG_MK)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "paper_grid.json"))
    ap.add_argument("--analyze", nargs="*", default=None)
    ap.add_argument("--u0-filter", type=float, default=None)
    a = ap.parse_args()
    if a.run:
        run_grid(a.cfgs.split(","), [float(x) for x in a.s.split(",")], [float(x) for x in a.d.split(",")],
                 a.u0, a.out)
    if a.analyze is not None:
        analyze(a.analyze or [a.out], a.u0_filter)

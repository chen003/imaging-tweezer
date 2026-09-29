"""Turn the scan into the optimisation answer and figures.

For every scan point and target fidelity F*, find the shortest imaging time T*
with F(T*) >= F*. The imaging fluence is s_tot * T*, and the N=0 Raman
probability is P_R = r_R * s_tot * T*, where r_R is the worst N=0 state's rate
per unit s_tot. The optimum is the point with the smallest P_R.

Writes results/optimum.json and docs/figures/pareto.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
TARGETS = (0.98, 0.99, 0.995)
GEOM_LABEL = {"B0": "B = 0", "B1_par_tw": "B = 1 G"}


def t_at(t, f, target):
    t, f = np.asarray(t), np.asarray(f)
    idx = np.where(f >= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    # log-linear interpolation between the bracketing checkpoints
    x0, x1, y0, y1 = np.log(t[i - 1]), np.log(t[i]), f[i - 1], f[i]
    return float(np.exp(x0 + (target - y0) / (y1 - y0) * (x1 - x0)))


def evaluate(rec, eta, target):
    tt = t_at(rec["t"], rec[f"F_{eta}"], target)
    if tt is None:
        return None
    rr = max(rec["raman_per_s"])
    i = int(np.argmin(np.abs(np.log(np.asarray(rec["t"]) / tt))))
    return {"T_us": tt, "fluence": rec["s"] * tt, "P_R": rr * rec["s"] * tt * 1e-6, "raman_per_s": rr,
            "photons": float(np.interp(np.log(tt), np.log(rec["t"]), rec["mean_ev"])),
            "detected": float(np.interp(np.log(tt), np.log(rec["t"]), rec["mean_606"])) * eta,
            "survival": float(rec["surv"][i]), "bg": float(np.interp(np.log(tt), np.log(rec["t"]), rec[f"bg_{eta}"]))}


def main(path=os.path.join(ROOT, "results", "scan.json")):
    data = json.load(open(path))
    recs, meta = data["records"], data["meta"]
    etas = meta["etas"]
    best = {}
    for g in GEOM_LABEL:
        for eta in etas:
            for tgt in TARGETS:
                cands = []
                for r in recs:
                    if r["geom"] != g:
                        continue
                    ev = evaluate(r, eta, tgt)
                    if ev:
                        cands.append({**ev, "U0": r["U0"], "s": r["s"], "d0": r["d0"]})
                key = f"{g}|eta={eta}|F={tgt}"
                best[key] = min(cands, key=lambda c: c["P_R"]) if cands else None
                fmax = max(max(r[f"F_{eta}"]) for r in recs if r["geom"] == g)
                if best[key]:
                    b = best[key]
                    print(f"{key:30s} P_R={b['P_R']:.2e}  U0={b['U0']} s={b['s']} d0={b['d0']} "
                          f"T={b['T_us']:.0f}us photons={b['photons']:.0f} det={b['detected']:.1f} "
                          f"bg={b['bg']:.2f} surv={b['survival']:.3f}")
                else:
                    print(f"{key:30s} not reached (max F = {fmax:.4f})")
    with open(os.path.join(ROOT, "results", "optimum.json"), "w") as f:
        json.dump({"meta": meta, "best": best}, f, indent=1)

    # figure: P_R at F=99% vs s, per U0 (best d0), rows = eta, cols = geometry
    u0s = sorted({r["U0"] for r in recs})
    ss = sorted({r["s"] for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, eta in zip(axes, etas):
        for gi, g in enumerate(GEOM_LABEL):
            for ui, u0 in enumerate(u0s):
                ys = []
                for s in ss:
                    evs = [evaluate(r, eta, 0.99) for r in recs if r["geom"] == g and r["U0"] == u0 and r["s"] == s]
                    evs = [e for e in evs if e]
                    ys.append(min(e["P_R"] for e in evs) if evs else np.nan)
                ax.plot(ss, ys, color=SER[ui], lw=2, ls="-" if g != "B0" else "--", marker="o" if g != "B0" else "s",
                        ms=4, label=f"{GEOM_LABEL[g]}, U₀ = {u0} mK")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total saturation parameter s_tot")
        ax.set_title(f"collection efficiency η = {eta:.0%}", loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8, which="both")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("N=0 Raman probability per image at F = 99%")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    fig.suptitle("Cross talk needed for a 99%-fidelity histogram (missing points: 99% not reachable)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "pareto.png"), dpi=150)
    plt.close(fig)
    return best


if __name__ == "__main__":
    main(*sys.argv[1:])

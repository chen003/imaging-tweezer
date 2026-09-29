"""Plot docs/figures/histograms.png from results/histograms.json (written by final_runs.py)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b7b6b0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"B0": "B = 0", "B1_par_tw": "B = 1 G ∥ tweezer polarisation"}


def plot(hist):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True, sharey=True)
    for col, eta in enumerate((0.02, 0.04)):
        for row, g in enumerate(("B0", "B1_par_tw")):
            ax, h = axes[row, col], hist[f"{g}|{eta}"]
            pb, pd = np.array(h["p_bright"]), np.array(h["p_dark"])
            n = np.arange(len(pb))
            ax.bar(n - 0.21, pd, width=0.42, color=GREY, label="empty trap or N=0 molecule")
            ax.bar(n + 0.21, pb, width=0.42, color=BLUE, label="N=1 molecule")
            ax.axvline(h["n_th"] + 0.5, color=ORANGE, lw=1.5, ls="--", label="best threshold")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_xlim(-0.8, 24.8)
            ax.set_title(f"{NAMES[g]}, η = {eta:.0%}", loc="left", fontsize=10.5)
            ax.text(0.985, 0.97, f"s_tot = {h['s']:g}, Δ = {h['d0']:+.0f} MHz, U₀ = {h['U0']} mK, "
                    f"T = {h['T_us']:.0f} µs\nF = {100 * h['F']:.2f}%,  N=0 Raman P = {h['P_R']:.1e}\n"
                    f"{h['photons']:.0f} photons scattered, {h['bg']:.2f} bkg counts",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="none", alpha=0.85))
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("detected photons in the molecule's region of interest")
            if col == 0:
                ax.set_ylabel("probability")
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Single-shot histograms at the rapid optimum (T ≤ 400 µs, 20 000 simulated molecules)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(ROOT, "docs", "figures", "histograms.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot(json.load(open(os.path.join(ROOT, "results", "histograms.json"))))

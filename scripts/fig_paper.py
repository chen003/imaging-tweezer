"""docs/figures/paper_tradeoff.png: cross talk vs imaging time for the paper's hardware (930 uK)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402

from caf import paper  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
LAB = {"1G_53": "1 G, 53°", "1G_90": "1 G, 90°", "state_4.4G_53": "4.4 G, 53° (paper, state-error image)",
       "erasure_2.0G_90": "2.0 G, 90° (paper, erasure image)"}


def main():
    opt = json.load(open(os.path.join(ROOT, "results", "paper_optimum_U0.93.json")))
    best, raman = opt["best"], opt["raman"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    ax = axes[0]
    for i, cfg in enumerate(LAB):
        fr = best.get(f"{cfg}|eta=0.04|paper|front") or []
        ax.plot([c["T_us"] for c in fr], [c["P_R"] for c in fr], color=SER[i], lw=2, marker="o", ms=4, label=LAB[cfg])
    p_paper = max(raman["state_4.4G_53"]["raman"]) * paper.S_TOT * paper.T_IMG_US * 1e-6
    ax.plot([paper.T_IMG_US], [p_paper], marker="*", ms=14, color=INK, ls="none",
            label="paper's operating point (3 ms, 4.5 mW/cm², η ≈ 5%)")
    ax.set_title("EMCCD as in the paper: ε₁₀ ≤ 3.3% at threshold 4.8, η = 4%", loc="left", fontsize=10.5)
    ax.text(0.03, 0.04, "η = 2%: not reachable at 930 µK\n(best ε₁₀ ≈ 7.4%, heating-limited)",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_ylabel("intrinsic N=0 Raman probability per image")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax = axes[1]
    for i, cfg in enumerate(LAB):
        for eta, ls in ((0.02, "-"), (0.04, "--")):
            fr = best.get(f"{cfg}|eta={eta}|F99|front") or []
            ax.plot([c["T_us"] for c in fr], [c["P_R"] for c in fr], color=SER[i], lw=2, ls=ls, marker="o", ms=3.5,
                    )
    ax.set_title("Photon counting, F = 99%  (solid η = 2%, dashed η = 4%)", loc="left", fontsize=10.5)
    ax.text(0.97, 0.96, "colours as in the left panel", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 300, 1000, 3000, 10000])
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlim(100, 10000)
        ax.set_xlabel("imaging time T (µs)")
        ax.grid(color=GRID, lw=0.8, which="both")
        ax.set_axisbelow(True)
    axes[0].set_ylim(5e-5, 2e-3)
    fig.suptitle("Paper hardware (781 nm, 930 µK, powers 5:2.7:1.1:3.8): least cross talk vs imaging time",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(os.path.join(ROOT, "docs", "figures", "paper_tradeoff.png"), dpi=150)
    print("paper point P_R (intrinsic):", p_paper)


if __name__ == "__main__":
    main()

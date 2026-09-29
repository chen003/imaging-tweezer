"""Collect headline numbers into results/summary.json (used by build_explorer.py and RESULTS.md)."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def e(x, d=1):
    s = f"{x:.{d}e}"
    m, ex = s.split("e")
    return f"{m}×10<sup>{int(ex)}</sup>"


def main():
    opt = json.load(open(os.path.join(ROOT, "results", "optimum.json")))
    best, fronts = opt["best"], opt["fronts"]
    phys = json.load(open(os.path.join(ROOT, "results", "physics.json")))
    scan = json.load(open(os.path.join(ROOT, "results", "scan.json")))["records"]
    sens = json.load(open(os.path.join(ROOT, "results", "sensitivity.json")))
    maxf = {f"{g}|{eta}": max(max(r[f"F_{eta}"]) for r in scan if r["geom"] == g)
            for g in ("B0", "B1_par_tw") for eta in (0.02, 0.04)}
    rr = phys["efficiency"]["raman_per_s"]
    eff = phys["efficiency"]
    i1 = min(range(len(eff["s"])), key=lambda i: abs(eff["s"][i] - 1))
    r1, r0 = best["B1_par_tw|eta=0.02|rapid"], best["B0|eta=0.02|rapid"]
    r14, r04 = best["B1_par_tw|eta=0.04|rapid"], best["B0|eta=0.04|rapid"]
    m1, m14 = best["B1_par_tw|eta=0.02|F=0.99"], best["B1_par_tw|eta=0.04|F=0.99"]
    answer = [
        f"<b>Yes, it works at 1 G.</b> With <b>B</b> along the tweezer polarisation and the imaging polarisation "
        f"about 50° away, N=1 scatters {eff['B1'][i1] / 1e6:.2f}×10⁶ photons/s per unit s_tot, against "
        f"{eff['B0'][i1] / 1e6:.2f}×10⁶ at zero field. A 99% histogram needs the same ≈{r1['photons']:.0f} photons "
        f"(η = 2%) either way. The one geometry to avoid is imaging polarisation parallel to <b>B</b>, which "
        f"leaves 8 dark states that nothing remixes.",
        f"<b>The cross talk on N=0 does not care about the field.</b> The imaging light is 19 GHz from the nearest "
        f"N=0 line, so P<sub>Raman</sub> = r<sub>R</sub>·s<sub>tot</sub>·T with r<sub>R</sub> = {rr:.2f} s⁻¹ at both "
        f"0 and 1 G. That is about {e(rr / eff['B1'][i1])} Raman events per imaging photon. Parity forbids an N=0 "
        f"molecule from ever landing in the bright peak.",
        f"<b>Rapid optimum, η = 2%:</b> s_tot = {r1['s']:g}, Δ = {r1['d0']:+.0f} MHz, U₀ = {r1['U0']} mK, "
        f"T = {r1['T_us']:.0f} µs gives F = 99% with P<sub>Raman</sub> = {e(r1['P_R'])} per image. "
        f"Going ten times slower (s_tot = {m1['s']:g}, T = {m1['T_us'] / 1000:.1f} ms) only lowers it to {e(m1['P_R'])}. "
        f"<b>η = 4%:</b> {r14['T_us']:.0f} µs and {e(r14['P_R'])}.",
        f"<b>Why the trade-off is so flat:</b> with 12 ground and 4 excited sublevels the line barely saturates. "
        f"Photons per unit fluence fall only from {eff['B1'][0] / 1e6:.2f} to "
        f"{eff['B1'][min(range(len(eff['s'])), key=lambda i: abs(eff['s'][i] - 4))] / 1e6:.2f}×10⁶ between s_tot = "
        f"{eff['s'][0]:g} and 4. So intensity × time for a fixed photon number is nearly constant, and speed is "
        f"almost free up to s_tot ≈ 4. Heating out of the trap (≈0.9 µK per photon) and trap-light loss cap the "
        f"fidelity at ≈{100 * maxf['B1_par_tw|0.02']:.1f}% (η = 2%) and ≈{100 * maxf['B1_par_tw|0.04']:.1f}% (η = 4%).",
    ]
    assume = [
        "<b>Molecule:</b> X(N≤3) hyperfine + Zeeman Hamiltonian (Childs 1981 constants); A(J′=½,±), A(J′=3/2,−) in case (a); Γ = 2π×8.29 MHz; b₀₀ = 0.975; v=1,2 repumped via A(v=0) at 2π×1 MHz; 2.5×10⁻⁵ leak to v≥3 per photon.",
        "<b>Tweezer:</b> 780 nm, w₀ = 1 µm, imaging depth U₀ = 1–2.5 mK (scanned), T₀ = 40 µK. N=1 tensor polarisability spread 20% of U₀ (Burchesky <i>et al.</i> 2021). X–A differential scalar shift 5 MHz/mK (unknown; see sensitivity).",
        "<b>Imaging light:</b> retro-reflected beam ⊥ tweezer, two passes added incoherently; linear polarisation 50° from the tweezer's; four sidebands with power 2:1:2:3 (J=½F=1, F=0, J=3/2 F=1, F=2), offsets +1, −1, +2, −2 MHz. Exactly equal detunings create accidental coherent dark states.",
        "<b>N=1 dynamics:</b> 17-level Lindblad OBE (secular), within 3–8% of the full multi-frequency model. The Monte-Carlo interpolates R(Δ, local intensity) along classical trajectories with recoil kicks, Doppler shifts and escape.",
        "<b>Losses:</b> trap-light-induced loss 4×10⁻⁵ per photon at 1.5 mK, scaled with depth (from 90% retention after 2700 photons in Λ-imaging, arXiv:1807.00740).",
        "<b>Detection:</b> η = photon collection × transmission × QE; photon-counting camera; stray light ζ = 10⁻³ of a two-level scatterer's collected light, plus 0.01 dark counts/ms; fidelity from the best threshold with a 50/50 prior.",
        "<b>N=0 cross talk:</b> Kramers–Heisenberg amplitudes through A(J′=½,−) at 19.2 GHz and A(J′=3/2,−) at 52 GHz, summed coherently. If the A(J′=½) Λ-doublet order is reversed, Δ₀ = 21.9 GHz and the Raman rate drops by ×0.77.",
    ]
    out = {"answer_html": answer, "assume_html": assume, "maxF": maxf, "raman_per_s": rr,
           "rapid": {"B0|0.02": r0, "B1|0.02": r1, "B0|0.04": r04, "B1|0.04": r14},
           "min_raman": {k: best[k] for k in best if "F=0.99" in k}, "sensitivity": sens}
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(answer))


if __name__ == "__main__":
    main()

"""Inject simulation results into docs/explorer_template.html -> docs/explorer.html."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def fmt_e(x):
    if x is None:
        return "–"
    e = int(f"{x:e}".split("e")[1])
    return f"{x / 10 ** e:.1f}×10<sup>{e}</sup>"


def paper_blocks():
    """Tables, field scan and headline answers from the paper-calibrated runs."""
    val = json.load(open(os.path.join(ROOT, "results", "paper_validation.json")))
    opt = json.load(open(os.path.join(ROOT, "results", "paper_optimum_U0.93.json")))["best"]
    opt15 = json.load(open(os.path.join(ROOT, "results", "paper_optimum_U1.5.json")))["best"]
    c = val["configs"]
    bs = val["bscan"]
    st, er, g1 = c["state_4.4G_53|exact"], c["erasure_2.0G_90|exact"], c["1G_53|exact"]
    s_tot = 4.5 / 4.86
    rows = [("", "paper (4.4 G, 53°)", "model 4.4 G", "model 1 G, 53°"),
            ("photon rate (s⁻¹)", "≈1.6×10⁵", f"{st['R_trap_bottom']:.2e}".replace("e+05", "×10⁵"),
             f"{g1['R_trap_bottom']:.2e}".replace("e+05", "×10⁵")),
            ("photons collected, 3 ms, η≈5%", "≈24", f"{st['photons_collected_eta5_survivors']:.1f}",
             f"{g1['photons_collected_eta5_survivors']:.1f}"),
            ("EMCCD ε₁₀ at 4.8", "3.3%", f"{100 * st['eps10_emccd_eta5']:.1f}%", f"{100 * g1['eps10_emccd_eta5']:.1f}%"),
            ("N=0 total scattering (s⁻¹)", "0.24 (theory), 0.76 (meas.)", f"{st['total_per_s'][1] * s_tot:.2f}",
             f"{g1['total_per_s'][1] * s_tot:.2f}"),
            ("N=0 Raman, qubit |1⟩, per 3 ms", "3.0×10<sup>-3</sup> (meas.)", fmt_e(st['raman_per_s'][2] * s_tot * 3e-3),
             fmt_e(g1['raman_per_s'][2] * s_tot * 3e-3))]
    val_table = "<tr>" + "".join(f"<th>{x}</th>" for x in rows[0]) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td{' class=num' if j else ''}>{x}</td>" for j, x in enumerate(r)) + "</tr>" for r in rows[1:])
    rec = [("EMCCD (paper), ε₁₀ ≤ 3.3%", 0.02, opt.get("1G_53|eta=0.02|paper"), "not reachable at 930 µK (best ε₁₀ ≈ 7.4%)"),
           ("EMCCD (paper), ε₁₀ ≤ 3.3%", 0.04, opt.get("1G_53|eta=0.04|paper"), ""),
           ("EMCCD, 1.5 mK trap", 0.02, opt15.get("1G_53|eta=0.02|paper"), ""),
           ("photon counting, F = 99%", 0.02, opt.get("1G_53|eta=0.02|F99"), ""),
           ("photon counting, F = 99%", 0.04, opt.get("1G_53|eta=0.04|F99"), "")]
    rt = ["<tr><th>camera / criterion at 1 G, 53°</th><th>η</th><th>s_tot</th><th>Δ (MHz)</th><th>T</th>"
          "<th>N=0 Raman (intrinsic)</th></tr>"]
    for name, eta, b, note in rec:
        if b is None:
            rt.append(f"<tr><td>{name}</td><td class='num'>{eta:.0%}</td><td colspan='4'>{note}</td></tr>")
        else:
            rt.append(f"<tr><td>{name}</td><td class='num'>{eta:.0%}</td><td class='num'>{b['s']:g}</td>"
                      f"<td class='num'>{b['d0']:+.0f}</td><td class='num'>{b['T_us'] / 1000:.2f} ms</td>"
                      f"<td class='num'>{fmt_e(b['P_R'])}</td></tr>")
    rec_table = "".join(rt) + ("<caption style='caption-side:bottom;text-align:left;padding-top:8px;color:var(--ink2);"
                               "font-size:0.85rem'>Least-cross-talk settings on the paper's hardware at 1 G. For the "
                               "paper's laser, the measured N=0 loss is about 5× the intrinsic Raman rate.</caption>")
    paper = {"B": bs["B"], "R53": bs["R"]["53|random polarisations"], "R90": bs["R"]["90|random polarisations"],
             "R90v": bs["R"]["90|both beams vertical (along tweezer axis)"], "valTable": val_table,
             "recTable": rec_table}
    answer = [
        "<b>Yes, it works at 1 G, and better than at the paper's 4.4 G.</b> With the paper's own recipe (4.5 mW/cm², "
        f"3 ms, on resonance) the model gives {g1['R_trap_bottom'] / 1e5:.2f}×10⁵ photons/s at 1 G (B 53° from the "
        f"tweezer polarisation), against {st['R_trap_bottom'] / 1e5:.2f}×10⁵ at 4.4 G, where the paper measured 1.6×10⁵. "
        f"The false-negative rate is the same: {100 * g1['eps10_emccd_eta5']:.1f}% vs {100 * st['eps10_emccd_eta5']:.1f}% "
        "(paper 3.3%).",
        "<b>Keep the imaging polarisation off the field axis.</b> If both beams are polarised along the tweezer axis and "
        "<b>B</b> points along it too, 8 dark states survive and the rate collapses about 6×.",
        "<b>The cross talk on N=0 does not depend on the field.</b> Raman scattering of the qubit |1⟩ is "
        f"{st['raman_per_s'][2] * s_tot:.2f} s⁻¹ at 4.5 mW/cm², the same at 1, 2 and 4.4 G. The paper measures about "
        "5× more and attributes the excess to laser spectral impurity.",
        "<b>At η = 2–4% the camera is the bottleneck, not the field.</b> With the paper's EMCCD, η = 2% cannot reach "
        "ε₁₀ = 3.3% in a 930 µK trap, and η = 4% needs about 4 ms. A photon-counting camera reaches 99% fidelity in "
        f"≈0.7 ms, with P<sub>Raman</sub> ≈ {fmt_e(opt['1G_53|eta=0.02|F99']['P_R'])} (η = 2%) or "
        f"{fmt_e(opt['1G_53|eta=0.04|F99']['P_R'])} (η = 4%). Detune 2–4 MHz red for Doppler cooling.",
    ]
    return paper, answer


def main():
    phys = json.load(open(os.path.join(ROOT, "results", "physics.json")))
    art = json.load(open(os.path.join(ROOT, "results", "artifact_data.json")))
    opt = json.load(open(os.path.join(ROOT, "results", "optimum.json")))["best"]
    summ = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    rows = ["<tr><th>setting</th><th>field</th><th>η</th><th>s_tot</th><th>Δ (MHz)</th><th>U₀ (mK)</th><th>T</th>"
            "<th>photons</th><th>N=0 Raman</th></tr>"]
    for kind, lab_k in (("rapid", "rapid, T ≤ 400 µs"), ("F=0.99", "least Raman")):
        for g, lab in (("B0", "B = 0"), ("B1_par_tw", "B = 1 G")):
            for eta in (0.02, 0.04):
                b = opt.get(f"{g}|eta={eta}|{kind}")
                if b is None:
                    continue
                rows.append(f"<tr><td>{lab_k}</td><td>{lab}</td><td class='num'>{eta:.0%}</td><td class='num'>{b['s']:g}</td>"
                            f"<td class='num'>{b['d0']:+.0f}</td><td class='num'>{b['U0']}</td>"
                            f"<td class='num'>{b['T_us']:.0f} µs</td><td class='num'>{b['photons']:.0f}</td>"
                            f"<td class='num'>{fmt_e(b['P_R'])}</td></tr>")
    table = "".join(rows) + ("<caption style='caption-side:bottom;text-align:left;padding-top:8px;color:var(--ink2);"
                             "font-size:0.85rem'>Settings that reach a 99% histogram fidelity with the least N=0 Raman "
                             "scattering, from the grid scan (3000 trajectories per point).</caption>")
    r1 = opt["B1_par_tw|eta=0.02|rapid"]
    data = {"physics": {"map": phys["map"]}, "runs": art["runs"],
            "meta": {**art["meta"], "raman_per_s": phys["efficiency"]["raman_per_s"]},
            "answer": summ["answer_html"], "table": table, "assume": summ["assume_html"],
            "defaults": {"geom": "B1_par_tw", "s": r1["s"], "T": r1["T_us"], "eta": 0.02}}
    data["paper"], data["answer"] = paper_blocks()
    tpl = open(os.path.join(ROOT, "docs", "explorer_template.html")).read()
    out = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    with open(os.path.join(ROOT, "docs", "explorer.html"), "w") as f:
        f.write(out)
    print("wrote docs/explorer.html", len(out) // 1024, "kB")


if __name__ == "__main__":
    main()

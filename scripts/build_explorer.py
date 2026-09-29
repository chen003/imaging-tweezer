"""Inject simulation results into docs/explorer_template.html -> docs/explorer.html."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def fmt_e(x):
    if x is None:
        return "–"
    e = int(f"{x:e}".split("e")[1])
    return f"{x / 10 ** e:.1f}×10<sup>{e}</sup>"


def main():
    phys = json.load(open(os.path.join(ROOT, "results", "physics.json")))
    art = json.load(open(os.path.join(ROOT, "results", "artifact_data.json")))
    opt = json.load(open(os.path.join(ROOT, "results", "optimum.json")))["best"]
    summ = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    rows = ["<tr><th>field</th><th>η</th><th>best s_tot</th><th>Δ (MHz)</th><th>U₀ (mK)</th><th>T</th>"
            "<th>photons</th><th>N=0 Raman</th></tr>"]
    for g, lab in (("B0", "B = 0"), ("B1_par_tw", "B = 1 G")):
        for eta in (0.02, 0.04):
            b = opt.get(f"{g}|eta={eta}|F=0.99")
            if b is None:
                rows.append(f"<tr><td>{lab}</td><td class='num'>{eta:.0%}</td><td colspan='6'>99% not reached "
                            f"(best {summ['maxF'][f'{g}|{eta}'] * 100:.1f}%)</td></tr>")
                continue
            rows.append(f"<tr><td>{lab}</td><td class='num'>{eta:.0%}</td><td class='num'>{b['s']}</td>"
                        f"<td class='num'>{b['d0']:+.0f}</td><td class='num'>{b['U0']}</td>"
                        f"<td class='num'>{b['T_us']:.0f} µs</td><td class='num'>{b['photons']:.0f}</td>"
                        f"<td class='num'>{fmt_e(b['P_R'])}</td></tr>")
    table = "".join(rows) + ("<caption style='caption-side:bottom;text-align:left;padding-top:8px;color:var(--ink2);"
                             "font-size:0.85rem'>Least-cross-talk settings that reach a 99% histogram fidelity "
                             "(grid scan, 3000 trajectories per point).</caption>")
    data = {"physics": {"map": phys["map"]}, "runs": art["runs"],
            "meta": {**art["meta"], "raman_per_s": phys["efficiency"]["raman_per_s"]},
            "answer": summ["answer_html"], "table": table, "assume": summ["assume_html"]}
    tpl = open(os.path.join(ROOT, "docs", "explorer_template.html")).read()
    out = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    with open(os.path.join(ROOT, "docs", "explorer.html"), "w") as f:
        f.write(out)
    print("wrote docs/explorer.html", len(out) // 1024, "kB")


if __name__ == "__main__":
    main()

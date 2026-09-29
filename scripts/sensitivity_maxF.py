"""Sensitivity of the eta = 2% rapid point: best fidelity reachable within 3 ms under each perturbation."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from caf.lookup import Geometry, Model  # noqa: E402
from final_runs import f_curve, mc  # noqa: E402
from run_scan import GEOMS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    opt = json.load(open(os.path.join(ROOT, "results", "optimum.json")))["best"]
    b = opt["B1_par_tw|eta=0.02|rapid"]
    model = Model()
    geom = GEOMS["B1_par_tw"]
    tck = np.geomspace(20, 3000, 45)
    base, lk = mc(model, geom, b["s"], b["U0"], b["d0"], tck, 6000, seed=31)
    rows = {}

    def summarise(name, out, lk_, zeta=1e-3):
        f = f_curve(out, b["s"], 0.02, zeta)
        i = int(np.argmax(f))
        rows[name] = {"maxF": float(f[i]), "T_at_maxF_us": float(tck[i]),
                      "P_R_at_maxF": float(max(lk_.raman_per_s) * b["s"] * tck[i] * 1e-6)}
        print(name, rows[name], flush=True)

    summarise("baseline", base, lk)
    for name, kw in (("kappa_s=0", {"kappa_s": 0.0}), ("kappa_s=15 MHz/mK", {"kappa_s": 15.0}),
                     ("p_trap=0", {"p_trap": 0.0}), ("p_trap=1e-4", {"p_trap": 1e-4})):
        o, _ = mc(model, geom, b["s"], b["U0"], b["d0"], tck, 6000, seed=31, lk=lk, **kw)
        summarise(name, o, lk)
    for z in (1e-4, 3e-3):
        summarise(f"zeta={z}", base, lk, zeta=z)
    for name, geom2, m2 in (("tensor spread 10%", geom, Model(tensor_spread=0.1)),
                            ("B || tweezer axis", Geometry(b_gauss=1.0, b_dir=(0, 0, 1), eps_img=geom.eps_img), model)):
        o, lk2 = mc(m2, geom2, b["s"], b["U0"], b["d0"], tck, 6000, seed=31)
        summarise(name, o, lk2)
    with open(os.path.join(ROOT, "results", "sensitivity_eta2.json"), "w") as f:
        json.dump({"point": b, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()

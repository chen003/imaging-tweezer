"""Grid scan: histogram fidelity vs imaging fluence at B = 0 and B = 1 G.

For every (geometry, trap depth U0, saturation s_tot, detuning delta0) this
builds the OBE lookup, runs the trajectory Monte-Carlo, and stores these
quantities versus imaging time:
* detection fidelity F(T) for eta = 2% and 4%;
* mean detected counts;
* survival.
It also stores the N=0 Raman rate per unit s_tot.

Usage: python scripts/run_scan.py [--quick] [--out results/scan.npz]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caf.camera import background_mean, bright_pmf, dark_pmf, fidelity  # noqa: E402
from caf.lookup import Geometry, Model  # noqa: E402
from caf.motion import run_mc  # noqa: E402

CHI = np.radians(50)
EPS_IMG = (np.cos(CHI), 0, np.sin(CHI))  # 50 deg from tweezer polarisation (x_lab), in x-z plane
GEOMS = {
    "B0": Geometry(b_gauss=0.0, b_dir=(1, 0, 0), eps_img=EPS_IMG, name="B = 0"),
    "B1_par_tw": Geometry(b_gauss=1.0, b_dir=(1, 0, 0), eps_img=EPS_IMG, name="B = 1 G || eps_tw"),
}
ETAS = (0.02, 0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "scan.npz"))
    ap.add_argument("--n-mol", type=int, default=3000)
    ap.add_argument("--kappa-s", type=float, default=5.0, help="differential scalar shift, MHz/mK")
    ap.add_argument("--zeta", type=float, default=1e-3)
    ap.add_argument("--p-trap", type=float, default=4e-5,
                    help="trap-light-induced loss per photon at U0=1.5 mK (scales with U0)")
    ap.add_argument("--geoms", default=",".join(GEOMS))
    args = ap.parse_args()

    s_grid = [0.3, 0.6, 1.0, 2.0, 4.0, 8.0] if not args.quick else [1.0, 4.0]
    d_grid = [-6.0, -4.0, -2.0, 0.0] if not args.quick else [-2.0]
    u_grid = [1.0, 1.5, 2.5] if not args.quick else [1.5]
    model = Model()
    records = []
    t0 = time.time()
    for gname in args.geoms.split(","):
        geom = GEOMS[gname]
        for u0 in u_grid:
            for s in s_grid:
                lk = model.lookup(geom, s, u0)
                rpeak = lk.R_tot.max()
                t_max = min(20e3, 3000 / rpeak * 1e6)
                tck = np.geomspace(5, t_max, 70)
                for d0 in d_grid:
                    out = run_mc(lk.delta, lk.u, lk.R_tot, tck, n_mol=args.n_mol, U0_mK=u0, delta0=d0,
                                 kappa_s_MHz_per_mK=args.kappa_s, beams=geom.beams,
                                 p_trap=args.p_trap * u0 / 1.5, seed=int(1000 * s + 10 * u0 + d0 + 50))
                    rec = {"geom": gname, "U0": u0, "s": s, "d0": d0, "t": tck.tolist(),
                           "raman_per_s": lk.raman_per_s.tolist(),
                           "mean_ev": out["nev"].mean(0).tolist(), "mean_606": out["n606"].mean(0).tolist(),
                           "surv": [(np.mean((out["t_dead_us"] < 0) | (out["t_dead_us"] > tt))) for tt in tck]}
                    for eta in ETAS:
                        fs, bgs, sig = [], [], []
                        for i, tt in enumerate(tck):
                            bg = background_mean(s, tt, eta, zeta=args.zeta)
                            pb = bright_pmf(out["n606"][:, i], eta, bg)
                            fs.append(fidelity(pb, dark_pmf(bg, len(pb) - 1))[0])
                            bgs.append(bg)
                        rec[f"F_{eta}"] = fs
                        rec[f"bg_{eta}"] = bgs
                    records.append(rec)
                    i99 = next((i for i, f in enumerate(rec["F_0.02"]) if f >= 0.99), None)
                    print(f"[{time.time() - t0:6.0f}s] {gname} U0={u0} s={s} d0={d0}: Rpeak={rpeak / 1e6:.2f}e6 "
                          f"T99(eta2%)={'%.0f us' % tck[i99] if i99 is not None else '--'} "
                          f"maxF2%={max(rec['F_0.02']):.4f} maxF4%={max(rec['F_0.04']):.4f}", flush=True)
    meta = {"kappa_s": args.kappa_s, "zeta": args.zeta, "p_trap": args.p_trap, "n_mol": args.n_mol,
            "chi_deg": 50, "etas": ETAS}
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump({"meta": meta, "records": records}, f)
    print("saved", args.out.replace(".npz", ".json"))


if __name__ == "__main__":
    main()

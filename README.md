# imaging-tweezer

Simulating fluorescence imaging of single CaF molecules in optical tweezers. The first question:

> Does the rapid resonant N=1 imaging scheme of [arXiv:2406.02391](https://arxiv.org/abs/2406.02391)
> still work at a finite bias field (e.g. 1 G)? What are its detection histogram and its cross talk onto a
> qubit stored in X(N=0), and which imaging parameters give the best trade-off?

**Status:** first study done. See [`RESULTS.md`](RESULTS.md) for the answer and the figures, and the
[interactive explorer](https://claude.ai/artifact/HjqQHnMSujqfdPKip5igzH) (private artifact).
[`PLAN.md`](PLAN.md) is the original plan.

**Short answer:** yes. The model is calibrated against the paper's measured scattering rate, photon
counts and false-negative rate. With the paper's recipe (4.5 mW/cm², 3 ms), 1 G gives the same
detection as their 4.4 G and 2.0 G fields, and 1.5× the scattering rate of 4.4 G. The imaging
polarisation must not be parallel to **B**. The N=0 Raman cross talk does not depend on the field:
about 5.6×10⁻⁴ per 3 ms image intrinsically, and about 5× more with the paper's laser. At
η = 2–4%, an EMCCD needs 3–4 ms images; a photon-counting camera reaches 99% in about 0.7 ms with
P_Raman ≈ 1–2×10⁻⁴.

## Layout

```
caf/structure.py   X(N<=3) hyperfine/Zeeman/tensor-light-shift Hamiltonian, A(J'=1/2,3/2) parity states, E1 dipoles
caf/obe.py         Lindblad OBE for N=1 imaging (secular steady state + full multi-frequency time evolution)
caf/raman.py       off-resonant Kramers-Heisenberg scattering of N=0 (Raman/Rayleigh, -> N=2, -> v>=1)
caf/motion.py      numba Monte-Carlo of trajectories, recoil heating, loss in a Gaussian tweezer
caf/camera.py      photon-count histograms and threshold fidelity
caf/lookup.py      R(detuning, local intensity) tables and lab-frame geometry
caf/ensemble.py    parallel lookups averaged over the standing-wave field of several imaging beams
caf/paper.py       experimental parameters of Holland et al. (arXiv:2406.02391)
scripts/           scan, analysis, figures, explorer page builder
tests/             level energies, selection rules, dark states, sum rules
results/           scan output (JSON) and summaries
docs/figures/      figures used in RESULTS.md; docs/explorer.html is the interactive page
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests
PYTHONPATH=. python scripts/fig_physics.py
PYTHONPATH=. python scripts/run_scan.py          # ~40 min on 4 cores
```

## References

- L. W. Cheuk *et al.*, Λ-enhanced imaging of molecules in an optical trap, PRL 121, 083201 (2018), [arXiv:1807.00740](https://arxiv.org/abs/1807.00740)
- C. M. Holland, Y. Lu, L. W. Cheuk, Bichromatic imaging of single molecules in an optical tweezer array, PRL 131, 053202 (2023), [arXiv:2208.12159](https://arxiv.org/abs/2208.12159)
- C. M. Holland, Y. Lu, S. J. Li, C. L. Welsh, L. W. Cheuk, Demonstration of (measurement-enhanced state preparation and) erasure conversion in a molecular tweezer array, PRX (2025), [arXiv:2406.02391](https://arxiv.org/abs/2406.02391)

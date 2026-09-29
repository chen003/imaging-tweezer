# imaging-tweezer

Simulating fluorescence imaging of single CaF molecules in optical tweezers. The first question:

> Does the rapid resonant N=1 imaging scheme of [arXiv:2406.02391](https://arxiv.org/abs/2406.02391)
> still work at a finite bias field (e.g. 1 G)? What are its detection histogram and its cross talk onto a
> qubit stored in X(N=0), and which imaging parameters give the best trade-off?

**Status:** first study done. See [`RESULTS.md`](RESULTS.md) for the answer and the figures, and the
[interactive explorer](https://claude.ai/artifact/HjqQHnMSujqfdPKip5igzH) (private artifact).
[`PLAN.md`](PLAN.md) is the original plan.

**Short answer:** yes. At 1 G, a 99%-fidelity histogram costs the N=0 qubit a Raman probability of
2.5×10⁻⁴ (η = 2%, 313 µs) or 1.0×10⁻⁴ (η = 4%, 257 µs) per image, the same as at zero field. The one
condition is that the imaging polarisation must not be parallel to **B**.

## Layout

```
caf/structure.py   X(N<=3) hyperfine/Zeeman/tensor-light-shift Hamiltonian, A(J'=1/2,3/2) parity states, E1 dipoles
caf/obe.py         Lindblad OBE for N=1 imaging (secular steady state + full multi-frequency time evolution)
caf/raman.py       off-resonant Kramers-Heisenberg scattering of N=0 (Raman/Rayleigh, -> N=2, -> v>=1)
caf/motion.py      numba Monte-Carlo of trajectories, recoil heating, loss in a Gaussian tweezer
caf/camera.py      photon-count histograms and threshold fidelity
caf/lookup.py      R(detuning, local intensity) tables and lab-frame geometry
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

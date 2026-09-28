# imaging-tweezer

Simulating fluorescence imaging of single CaF molecules in optical tweezers. The first question:

> Does the rapid resonant N=1 imaging scheme of [arXiv:2406.02391](https://arxiv.org/abs/2406.02391)
> still work at a finite bias field (e.g. 1 G)? What are its detection histogram and its cross talk onto a
> qubit stored in X(N=0), and which imaging parameters give the best trade-off?

**Status:** planning. See [`PLAN.md`](PLAN.md) for the model, the validation steps and the open questions.

## Layout

The model code will live in `caf/`; only the estimate script exists so far.

```
PLAN.md                    simulation plan (for review)
scripts/plan_estimates.py  X-state Hamiltonian + back-of-envelope numbers quoted in the plan
docs/figures/              figures
caf/                       (to come) structure, OBE, motion, camera, cross-talk, optimisation
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/plan_estimates.py
```

## References

- L. W. Cheuk *et al.*, Λ-enhanced imaging of molecules in an optical trap, PRL 121, 083201 (2018), [arXiv:1807.00740](https://arxiv.org/abs/1807.00740)
- C. M. Holland, Y. Lu, L. W. Cheuk, Bichromatic imaging of single molecules in an optical tweezer array, PRL 131, 053202 (2023), [arXiv:2208.12159](https://arxiv.org/abs/2208.12159)
- C. M. Holland, Y. Lu, S. J. Li, C. L. Welsh, L. W. Cheuk, Demonstration of (measurement-enhanced state preparation and) erasure conversion in a molecular tweezer array, PRX (2025), [arXiv:2406.02391](https://arxiv.org/abs/2406.02391)

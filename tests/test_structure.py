import numpy as np

from caf.structure import normalise_dipoles, project, x_manifold


def test_x_hyperfine_levels(xa):
    x, _ = xa
    m1 = x_manifold(x, 1, b_gauss=1e-6)
    e = np.sort(np.unique(np.round(m1.energies - m1.energies.min(), 1)))
    # CaF X(N=1): J=1/2 F=1, F=0, J=3/2 F=1, F=2 (literature: 0, 76.3, 122.9, 147.8 MHz)
    assert np.allclose(e, [0.0, 76.3, 122.9, 147.8], atol=0.15)
    m0 = x_manifold(x, 0, b_gauss=1e-6)
    assert abs(np.ptp(m0.energies) - 122.56) < 0.05


def test_zeeman_n0_at_1G(xa):
    x, _ = xa
    m0 = x_manifold(x, 0, b_gauss=1.0)
    e = np.sort(m0.energies - x_manifold(x, 0, b_gauss=1e-6).energies.mean())
    # F=1 mF=+-1 shift by +-g_S mu_B B / 2 = +-1.40 MHz
    assert abs((e[3] - e[1]) / 2 - 1.399) < 0.01


def test_parity_and_normalisation(xa):
    x, a = xa
    m0, m1, m2 = (x_manifold(x, n, b_gauss=1e-6) for n in (0, 1, 2))
    for j, par in ((0.5, 1), (0.5, -1), (1.5, -1)):
        _, dip, _ = a.block(j, par, b_gauss=1e-6)
        dip = normalise_dipoles(dip)
        w = {n: sum(np.abs(project(dip, m.vecs)[q]) ** 2 for q in (1, 0, -1)).sum(1)
             for n, m in ((0, m0), (1, m1), (2, m2))}
        if par == 1:
            assert np.allclose(w[1], 1) and np.allclose(w[0], 0) and np.allclose(w[2], 0)
        else:
            assert np.allclose(w[1], 0) and np.allclose(w[0] + w[2], 1)
    # A(J'=1/2,-): 2/3 to N=0, 1/3 to N=2
    _, dip, _ = a.block(0.5, -1, b_gauss=1e-6)
    dip = normalise_dipoles(dip)
    w0 = sum(np.abs(project(dip, m0.vecs)[q]) ** 2 for q in (1, 0, -1)).sum(1)
    assert np.allclose(w0, 2 / 3, atol=1e-3)


def test_delta_f_selection_rule(xa):
    x, a = xa
    m1 = x_manifold(x, 1, b_gauss=1e-6)
    en, dip, fz = a.block(0.5, 1, b_gauss=1e-6)
    dip = normalise_dipoles(dip)
    d = project(dip, m1.vecs)
    br = sum(np.abs(d[q]) ** 2 for q in (1, 0, -1))
    f0 = int(np.argmin(en))  # F'=0 lies 4.8 MHz below F'=1
    assert br[f0, m1.hf_label == 1].sum() < 1e-10  # F'=0 -> F=0 forbidden
    assert br[f0, m1.hf_label == 3].sum() < 1e-10  # F'=0 -> F=2 forbidden

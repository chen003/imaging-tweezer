import numpy as np

from caf import constants as k
from caf.obe import GAM, N1Imaging
from caf.raman import N0Raman


def test_dark_states_at_zero_field(xa):
    x, a = xa
    sysm = N1Imaging(x, a, b_gauss=1e-4)
    assert sysm.scattering_rate(1.0, 0.0, eps=(1, 0, 0)) < 1e3  # static polarisation, no remixing


def test_scattering_bounded_by_multilevel_limit(xa):
    x, a = xa
    sysm = N1Imaging(x, a, b_gauss=2.0)
    r = sysm.scattering_rate(300.0, 0.0, eps=(np.sqrt(0.5), 0, np.sqrt(0.5)))
    assert 0 < r < GAM * 1e6 * 4 / 16


def test_raman_sum_rule_and_scaling(xa):
    x, a = xa
    img = N1Imaging(x, a, b_gauss=1.0)
    ram = N0Raman(x, a, b_gauss=1.0)
    s = np.full(4, 0.25)
    out = ram.rates(s, 0.0, (1, 0, 0), img.centres)
    assert np.allclose(out["to_N0"].sum(1) + out["to_N2"].sum(1) + out["to_v1"], out["total"])
    out2 = ram.rates(2 * s, 0.0, (1, 0, 0), img.centres)
    assert np.allclose(out2["total"], 2 * out["total"])
    # order of magnitude: (Gamma/2) s (Gamma/2Delta)^2 with Delta ~ 19 GHz
    two_level = k.GAMMA / 2 * (k.GAMMA / (2 * 2 * np.pi * 19.2e9)) ** 2
    assert 0.05 * two_level < out["total"].mean() < 2 * two_level

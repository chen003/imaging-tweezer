"""Physical and CaF spectroscopic constants. Frequencies in MHz unless noted."""
import numpy as np

# fundamental
H = 6.62607015e-34
HBAR = H / (2 * np.pi)
KB = 1.380649e-23
C = 2.99792458e8
AMU = 1.66053906660e-27
MU_B = 1.39962449  # MHz / G
MU_N = MU_B / 1836.15267343

# X2Sigma+(v=0): Childs, Goodman & Goodman, Phys. Rev. A (1981)
B_X = 10267.54
GAMMA_SR = 39.65891
B_HF = 109.1839
C_HF = 40.1190
C_I = 2.876e-2
G_S = 2.00231930
G_I = 5.25774  # 19F

# A2Pi1/2(v=0)
TAU_A = 19.2e-9  # s, Wall et al. PRA 78, 062509 (2008)
GAMMA = 1.0 / TAU_A  # s^-1 (angular)
GAMMA_MHZ = GAMMA / (2 * np.pi) / 1e6  # 8.29 MHz
LAMBDA = 606.3e-9  # m, X-A(0,0)
K_WAVE = 2 * np.pi / LAMBDA
B_A_EFF = 10.4e3  # effective rotational constant of A(Omega=1/2) [MHz] (approx.)
LAMBDA_DOUBLING = 1.36e3  # |p+2q| [MHz]; J'=1/2 (+) assumed above (-)
A_HF_J12 = 4.8  # F'=1 - F'=0 splitting in A(J'=1/2) [MHz] (unresolved in experiments)
G_J_A12 = -0.021  # small g-factor of A(J'=1/2)

# vibrational branching of A(v'=0): b00 in range 0.964-0.987 in the literature
B00 = 0.975
B_LOSS = 2.5e-5  # decay to unrepumped vibrational levels (v>=3)

M_CAF = 59.08 * AMU
E_REC = (HBAR * K_WAVE) ** 2 / (2 * M_CAF)  # J
I_SAT = np.pi * H * C * GAMMA / (3 * LAMBDA ** 3) / 10  # mW/cm^2 (two-level)

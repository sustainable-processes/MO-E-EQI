import torch
from scipy.integrate import solve_ivp
import numpy as np

def reaction_simulator(x, NOISE_LEVEL=None, NOISE_STRUCTURE=None):
    tau = x[:,0]
    equiv_pldn = x[:,1]
    conc_dfnb = x[:,2]
    temperature = x[:,3]

    # Initial Concentrations in mM
    C_i = torch.zeros(5)
    C_i[0] = conc_dfnb
    C_i[1] = equiv_pldn * conc_dfnb

    # Flowrate and residence time
    V = 3  # mL
    q_tot = V / tau

    def fun(t, C, T):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 140 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
        k = lambda k_ref, E_a, temp: 0.6 * k_ref * torch.exp(-E_a / R * (1 / temp - 1 / T_ref))
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)

        # Reaction Rates
        r = torch.zeros(5)
        for i in [0, 1]:  # Set to reactants when close
            C[i] = 0 if C[i] < 1e-6 * C_i[i] else C[i]
        r[0] = -(k_a + k_b) * C[0] * C[1]
        r[1] = -(k_a + k_b) * C[0] * C[1] - k_c * C[1] * C[2] - k_d * C[1] * C[3]
        r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
        r[3] = k_b * C[0] * C[1] - k_d * C[1] * C[3]
        r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]

        # Deltas
        dcdtau = r
        return dcdtau

    # Integrate
    res = solve_ivp(lambda t, y: fun(t, y, temperature), [0, tau], C_i.numpy())
    C_final = torch.tensor(res.y[:, -1])

    # Calculate STY and E-factor
    M = torch.tensor([159.09, 87.12, 226.21, 226.21, 293.32])  # molecular weights (g/mol)
    sty = (M[2] * C_final[2] / (tau/60)) / 60

    if sty < 1e-6:
        sty = 1e-6

    mass_in = C_i[0] * M[0] * (3/1000) + C_i[1] * M[1] * (3/1000)
    mass_prod = M[2] * C_final[2] * (3/1000)
    e_factor = (mass_in - mass_prod) / mass_prod

    if e_factor > 1e3:
        e_factor = 1e3


    if NOISE_STRUCTURE == "LINEAR":
        sty = sty + sty * np.random.normal(scale=NOISE_LEVEL, size=1)
        e_factor = e_factor + e_factor * np.random.normal(scale=NOISE_LEVEL, size=1)

    elif NOISE_STRUCTURE == "LOGLINEAR_1":
        noise_slope = 0.8495
        noise_level = -1.699
        sty = sty + sty**noise_slope * 10**noise_level * np.random.normal(0, size=1)
        e_factor = e_factor + e_factor**noise_slope * 10**noise_level * np.random.normal(0, size=1)

    elif NOISE_STRUCTURE == "LOGLINEAR_2":
        noise_slope = 1.20
        noise_level = -1.30
        sty = sty + sty**noise_slope * 10**noise_level * np.random.normal(0, size=1)
        e_factor = e_factor + e_factor**noise_slope * 10**noise_level * np.random.normal(0, size=1)

    elif NOISE_STRUCTURE == "NA":
        sty = sty
        e_factor = torch.tensor([e_factor])

    return torch.cat((sty, -e_factor), dim=0).reshape(1, -1)

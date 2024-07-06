

# This is snar benchmark adapted from summit (correct some mistakes)

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from summit import *
import matplotlib.pyplot as plt

class ReactionSimulator(Experiment):


    def __init__(self, noise_level: float = 0):
        domain = self.setup_domain()
        self.noise_level = noise_level
        super().__init__(domain)

    @staticmethod
    def setup_domain():
        domain = Domain()

        # Decision variables
        domain += ContinuousVariable(
            name="res_time", description="residence_time", bounds=[0.5, 2.0]
        )
        domain += ContinuousVariable(
            name="equiv", description="equivalent", bounds=[1.0, 5.0]
        )
        domain += ContinuousVariable(
            name="conc_dfnb", description="initial_concentration", bounds=[0.1, 0.5]
        )
        domain += ContinuousVariable(
            name="temp", description="temperature", bounds=[30, 120]
        )

        # Objectives
        domain += ContinuousVariable(
            name="sty",
            description="Space Time Yield",
            bounds=[0, 1e6],
            is_objective=True,
            maximize=True,
        )
        domain += ContinuousVariable(
            name="e_factor",
            description="E Factor",
            bounds=[0, 1e4],
            is_objective=True,
            maximize=False,
        )

        return domain


    def _run(self, conditions: DataSet, plot: bool = False, **kwargs) -> DataSet:

        res_time = float(conditions["res_time"])
        equiv = float(conditions["equiv"])
        conc_dfnb = float(conditions["conc_dfnb"])
        temp = float(conditions["temp"])

        sty, e_factor = self.calculate_obj(res_time, equiv, conc_dfnb, temp)

        conditions["sty", "DATA"] = sty
        conditions["e_factor", "DATA"] = e_factor
        return conditions, {}


    def calculate_obj(self, tau, equiv_pldn, conc_dfnb, temperature):

        C_i = np.zeros(5)
        C_i[0] = conc_dfnb
        C_i[1] = equiv_pldn * conc_dfnb

        # Flowrate and residence time
        V = 3  # mL
        q_tot = V / tau


        def fun(t, C, T):

            R = 8.314 / 1000  # kJ/K/mol
            T_ref = 140 + 273.71  # Convert to deg K
            T = T + 273.71  # Convert to deg K
            # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
            k = (
                lambda k_ref, E_a, temp: 0.6
                * k_ref
                * np.exp(-E_a / R * (1 / temp - 1 / T_ref))
            )
            k_a = k(57.9, 33.3, T)
            k_b = k(2.70, 35.3, T)
            k_c = k(0.865, 38.9, T)
            k_d = k(1.63, 44.8, T)

            # Reaction Rates
            r = np.zeros(5)
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

        res = solve_ivp(fun, [0, tau], C_i, args=(temperature,))
        C_final = res.y[:, -1]
        # plt.plot(res.t, res.y[0,:],'o',res.t, res.y[1,:],res.t, res.y[2,:],res.t, res.y[3,:])
        # plt.show()

        # Calculate STY and E-factor
        M = [159.09, 87.12, 226.21, 226.21, 293.32]  # molecular weights (g/mol)
        sty = M[2] * C_final[2] / (tau/60) # unit: kg/(m^3 * h)

        if sty < 1e-6:
            sty = 1e-6

        mass_in = C_i[0] * M[0] * (3/1000) + C_i[1] * M[1] * (3/1000)
        mass_prod = M[2] * C_final[2] * (3/1000)
        e_factor = (mass_in - mass_prod) / mass_prod

        if e_factor > 1e3:
            e_factor = 1e3


        ## implement linear noises
        sty = sty + sty * np.random.normal(scale=self.noise_level, size=1)
        e_factor = e_factor + e_factor * np.random.normal(scale=self.noise_level, size=1)

        # # implement log-linear_1
        # noise_slope = 0.8495
        # noise_level = -1.699
        # sty = sty + sty ** noise_slope * 10 ** noise_level * np.random.normal(0, size=1)
        # e_factor = e_factor + e_factor ** noise_slope * 10 ** noise_level * np.random.normal(0, size=1)

        #
        # # implement log-linear_2
        # noise_slope = 1.20
        # noise_level = -1.30
        # sty = sty + sty ** noise_slope * 10 ** noise_level * np.random.normal(0, size=1)
        # e_factor = e_factor + e_factor ** noise_slope * 10 ** noise_level * np.random.normal(0, size=1)


        return sty[0], e_factor[0]


def test_benchmark():
    exp = ReactionSimulator(noise_level=0.05)
    conditions = pd.DataFrame(
        [{"res_time": 1, "equiv": 1, "conc_dfnb": 0.1, "temp": 30}]
    )
    conditions = DataSet.from_df(conditions)
    results = exp.run_experiments(conditions)
    print(results["sty"],results["e_factor"])

if __name__ == "__main__":
    test_benchmark()





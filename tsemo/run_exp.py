# This is for suggesting next experiment

import numpy as np
import pandas as pd
import openpyxl
from utils import calculate_hypervolume
import matplotlib.pyplot as plt

from summit import *
from optimize import MOBO


# Create experimental domain
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
    name="temp", description="temperature", bounds=[60, 140]
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


data_df = pd.read_excel("initial_design/initial_design_8.xlsx",
                        header=None, usecols="A:F", skiprows=[0], nrows=20,
                        names=["res_time","equiv","conc_dfnb","temp","sty","e_factor"])

data = DataSet.from_df(data_df)
strategy = TSEMO(domain)
result = strategy.suggest_experiments(1, prev_res=data)
print(result)

# entry = 'TrainingSet_MOBO_4var_sty'
# initial_exp = 38
# Data_DF = pd.read_excel(entry+'.xlsx', sheet_name='data', header=None, usecols="A:F", skiprows=[0], nrows=initial_exp,
#                         names=["equiv","flowrate","solv","elec","sty","e_factor"])
#
# Data = DataSet.from_df(Data_DF)
# # print(Data_DF)
# # print(Data)
# strategy = MOBO(domain)
# result = strategy.suggest_experiments(1, prev_res=Data)
#
#
# hvs = calculate_hypervolume(strategy.all_experiments,domain)
# print(hvs)
# print(strategy.all_experiments)
# # print(strategy.pareto_data)
#
# fig, ax = plt.subplots(1)
# ax.plot(np.linspace(1,initial_exp,initial_exp),hvs)
# axis_fontsize= 14
# ax.set_ylabel("Hypervolume", fontsize=axis_fontsize)
# ax.set_xlabel("Iteration", fontsize=axis_fontsize)
# # ax.set_ylim(0,3000)
# fig.show()
# ax.tick_params(direction="in", labelsize=axis_fontsize)
# fig.savefig("results/hypervolume.png", dpi=300)


# equiv_new = result["equiv"][0].round(1)
# flowrate_new = result["flowrate"][0].round(2)
# solvent_new = result["solv"][0]
# elec_new = result["elec"][0]
#
# next_exp = [equiv_new,flowrate_new,solvent_new,elec_new]
# print(next_exp)
# # #



# def get_maximum_rows(*, sheet_object):
#     rows = 0
#     for max_row, row in enumerate(sheet_object, 1):
#         if not all(col.value is None for col in row):
#             rows += 1
#     return rows
#
# excel = openpyxl.load_workbook(entry+'.xlsx')
# worksheet = excel['data']
# max_rows = get_maximum_rows(sheet_object=worksheet)
#
# for index, value in enumerate(next_exp):
#     worksheet.cell(row=max_rows+1, column=1+index).value = value
#
# excel.save(entry+'.xlsx')
# strategy.save(entry+'.json')

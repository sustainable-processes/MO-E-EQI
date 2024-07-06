import os
import torch


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

import time
import warnings
from pyDOE2 import lhs
from reaction_simulator import *

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models import HeteroskedasticSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize, standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
from transform import *

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


domain = [(0.5, 2), (1, 5), (0.1, 0.5), (30,120)]
problem_bounds = torch.tensor(domain).t()
n_samples = 20
ref_point = torch.tensor([13, -4])
num_objectives = 2
dim = problem_bounds.size(1)


def initial_design_lhs(problem_bounds, n_samples):

    dim = problem_bounds.size(1)
    samples = lhs(dim, samples=n_samples)
    # samples = lhs(dim, samples=n_samples, criterion='maximin')

    # Scale the samples to the bounds using PyTorch operations
    samples_tensor = torch.tensor(samples)
    for i in range(dim):
        lower_bound, upper_bound = problem_bounds[:,i]
        samples_tensor[:, i] = samples_tensor[:, i] * (upper_bound - lower_bound) + lower_bound

    return samples_tensor




def transform_y(y):
    """ Transform train_y using mean and std"""
    mean_y = torch.mean(y, dim=0)
    std_y = torch.std(y, dim=0)
    return (y - mean_y) / std_y

def transform_var(y, yvar):
    std_y = torch.std(y, dim=0)
    std_y_sq = std_y.pow(2)
    return yvar / std_y_sq




def initialize_model(train_x, train_obj, NOISE_LEVEL, NOISE_STRUCTURE):
    # define models for objective and constraint

    ########################################
    # ## This is for FixedNoiseGP
    # train_x = normalize(train_x, problem_bounds)
    # models = []
    # for i in range(train_obj.shape[-1]):
    #     train_y = train_obj[..., i : i + 1]
    #     train_yvar = torch.full_like(train_y, NOISE_LEVEL ** 2)
    #     models.append(
    #         FixedNoiseGP(
    #             train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
    #         )
    #     )


    ########################################
    # ## This is for HeteroskedasticSingleTaskGP
    import warnings
    warnings.filterwarnings("ignore")
    models = []
    train_x = normalize(train_x, problem_bounds)
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]

        if NOISE_STRUCTURE == "LINEAR":
            train_Yvar = (train_y * NOISE_LEVEL / torch.std(train_y)) ** 2

            models.append(
                HeteroskedasticSingleTaskGP(
                    train_x, train_y, train_Yvar.detach(),
                    outcome_transform=Standardize_self(m=1))
                )

        elif NOISE_STRUCTURE == "LOGLINEAR_1":
            noise_slope = 0.8495
            noise_level = -1.699
            train_y_pos = abs(train_y)  # change values in train_y to positive
            train_Yvar = ((train_y_pos ** noise_slope * 10 ** noise_level) / torch.std(train_y_pos)) ** 2

            models.append(
                HeteroskedasticSingleTaskGP(
                    train_x, train_y, train_Yvar.detach(),
                    outcome_transform=Standardize_self(m=1))
            )


        elif NOISE_STRUCTURE == "LOGLINEAR_2":
            noise_slope = 1.20
            noise_level = -1.30
            train_y_pos = abs(train_y)  # change values in train_y to positive
            train_Yvar = ((train_y_pos ** noise_slope * 10 ** noise_level) / torch.std(train_y_pos)) ** 2

            models.append(
                HeteroskedasticSingleTaskGP(
                    train_x, train_y, train_Yvar.detach(),
                    outcome_transform=Standardize_self(m=1))
            )


        elif NOISE_STRUCTURE == "NA":
            train_Yvar = torch.full_like(train_y, 0)
            models.append(
                FixedNoiseGP(
                    train_x, train_y, train_Yvar,
                    outcome_transform=Standardize(m=1))
            )

    # ########################################

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model



def optimize_qehvi_and_get_observation(model, train_x, sampler, bounds,
                                       BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES,
                                       NOISE_LEVEL, NOISE_STRUCTURE):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem_bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    # new_x = candidates
    new_obj_true = reaction_simulator(new_x, NOISE_LEVEL=0, NOISE_STRUCTURE="NA")
    new_obj = reaction_simulator(new_x, NOISE_LEVEL, NOISE_STRUCTURE=NOISE_STRUCTURE)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true



def optimize_qnehvi_and_get_observation(model, train_x, sampler, bounds,
                                        BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES,
                                        NOISE_LEVEL, NOISE_STRUCTURE):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, problem_bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    # new_x = candidates
    new_obj_true = reaction_simulator(new_x, NOISE_LEVEL=0, NOISE_STRUCTURE="NA")
    new_obj = reaction_simulator(new_x, NOISE_LEVEL, NOISE_STRUCTURE=NOISE_STRUCTURE)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true




def optimize_qnparego_and_get_observation(model, train_x, sampler, bounds,
                                          BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES,
                                          NOISE_LEVEL, NOISE_STRUCTURE):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, problem_bounds)
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(num_objectives, **tkwargs).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=pred)
        )
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    # new_x = candidates
    new_obj_true = reaction_simulator(new_x, NOISE_LEVEL=0, NOISE_STRUCTURE="NA")
    new_obj = reaction_simulator(new_x, NOISE_LEVEL, NOISE_STRUCTURE=NOISE_STRUCTURE)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true)* NOISE_SE

    return new_x, new_obj, new_obj_true
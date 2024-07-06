## This is for algorithm comparisons qNEHVI, qEHVI, qParEGO
## Reference: https://botorch.org/tutorials/multi_objective_bo





import pandas as pd
import matplotlib.pyplot as plt
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from reaction_simulator import *
from helper import *

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPEAT_START = 1
REPEAT_END = 15

NOISE_LEVEL = 0.05
NOISE_STRUCTURE = "LOGLINEAR_1"

verbose = True
N_BATCH = 20  # how many iterations
MC_SAMPLES = 128
BATCH_SIZE = 1  # how many exp suggest each time
NUM_RESTARTS = 10
RAW_SAMPLES = 512



for repeat in range(REPEAT_START, REPEAT_END):

    # Initialization
    hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_random = [], [], [], []
    standard_bounds = torch.zeros(2, 4, **tkwargs)
    standard_bounds[1] = 1

    ##########################################################################################

    # Import MaxPro initial design from excel
    samples = pd.read_csv("/Users/zhang/PycharmProjects/bo_mixed/snar/initial_design/TrainingSet1.csv")
    samples = samples.rename(columns={"conc": "conc_dfnb"})

    train_x_qparego = samples.iloc[:,0:4]
    train_x_qparego = torch.tensor(train_x_qparego.values)

    # Calculate train_obj based on x
    train_obj_true_qparego_res = []
    train_obj_qparego_res = []

    for i in range(0,20):
        train_obj_true_qparego_res.append(reaction_simulator(train_x_qparego[i].unsqueeze(dim=0),
                                                               NOISE_LEVEL=0,
                                                               NOISE_STRUCTURE="NA"))

        train_obj_qparego_res.append(reaction_simulator(train_x_qparego[i].unsqueeze(dim=0),
                                                          NOISE_LEVEL=NOISE_LEVEL,
                                                          NOISE_STRUCTURE=NOISE_STRUCTURE))

    train_obj_true_qparego = torch.cat(train_obj_true_qparego_res, dim=0)
    train_obj_qparego = torch.cat(train_obj_qparego_res, dim=0)

    train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )
    train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )

    train_x_random, train_obj_random, train_obj_true_random = (
        train_x_qparego,
        train_obj_qparego,
        train_obj_true_qparego,
    )


    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, NOISE_LEVEL, NOISE_STRUCTURE)
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, NOISE_LEVEL, NOISE_STRUCTURE)
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, NOISE_LEVEL, NOISE_STRUCTURE)



    # #########################################################################################
    # unused here
    # # generate initial training data and initialize model
    #
    # train_x_qparego = initial_design_lhs(problem_bounds, n_samples)
    # train_obj_qparego = []
    # train_obj_true_qparego = []
    #
    # for row in train_x_qparego:
    #     res= reaction_simulator_2(torch.unsqueeze(row, dim=0))
    #     train_obj_qparego.append(res)
    #
    #
    # train_obj_qparego  = torch.cat(train_obj_qparego, dim=0)
    # train_obj_true_qparego  =  train_obj_qparego + torch.randn_like(train_obj_qparego) * NOISE_SE
    #
    #
    # train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = (
    #     train_x_qparego,
    #     train_obj_qparego,
    #     train_obj_true_qparego,
    # )
    # train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (
    #     train_x_qparego,
    #     train_obj_qparego,
    #     train_obj_true_qparego,
    # )
    # train_x_random, train_obj_random, train_obj_true_random = (
    #     train_x_qparego,
    #     train_obj_qparego,
    #     train_obj_true_qparego,
    # )
    #
    # mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
    # mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    # mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

    ##########################################################################################

    # compute hypervolume
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_true_qparego)
    volume = bd.compute_hypervolume().item()

    hvs_qparego.append(volume)
    hvs_qehvi.append(volume)
    hvs_qnehvi.append(volume)
    hvs_random.append(volume)

    ##########################################################################################

    print(f"\n\nThis is repeat_{repeat}")

    time_all = []

    for iteration in range(1, N_BATCH + 1):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_qparego)
        fit_gpytorch_mll(mll_qehvi)
        fit_gpytorch_mll(mll_qnehvi)


        # define the qEI and qNEI acquisition modules using a QMC sampler
        qparego_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition functions and get new observations
        (new_x_qparego,new_obj_qparego, new_obj_true_qparego,) = optimize_qnparego_and_get_observation(
            model_qparego, train_x_qparego, qparego_sampler, standard_bounds,
            BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_LEVEL, NOISE_STRUCTURE)

        new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
            model_qehvi, train_x_qehvi, qehvi_sampler, standard_bounds,
            BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_LEVEL, NOISE_STRUCTURE)

        (new_x_qnehvi,new_obj_qnehvi,new_obj_true_qnehvi,) = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_qnehvi, qnehvi_sampler, standard_bounds,
            BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, NOISE_LEVEL, NOISE_STRUCTURE)

        new_x_random = initial_design_lhs(problem_bounds, 1)
        new_obj_true_random = reaction_simulator(new_x_random,
                                                   NOISE_LEVEL=0, NOISE_STRUCTURE="NA")
        new_obj_random = reaction_simulator(new_x_random,
                                              NOISE_LEVEL, NOISE_STRUCTURE=NOISE_STRUCTURE)
    #
        #######################################

         # update training points
        train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
        train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
        train_obj_true_qparego = torch.cat([train_obj_true_qparego, new_obj_true_qparego])

        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
        train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])

        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])
        train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random])


        #######################################
        # # update progress
        for hvs_list, train_obj in zip(
            (hvs_random, hvs_qparego, hvs_qehvi, hvs_qnehvi),
            (
                train_obj_true_random,
                train_obj_true_qparego,
                train_obj_true_qehvi,
                train_obj_true_qnehvi,
            ),
        ):
            # compute hypervolume
            bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
            volume = bd.compute_hypervolume().item()
            hvs_list.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration


        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego, NOISE_LEVEL, NOISE_STRUCTURE)
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi, NOISE_LEVEL, NOISE_STRUCTURE)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, NOISE_LEVEL, NOISE_STRUCTURE)


        #######################################

        t1 = time.monotonic()
        if verbose:
            print(
                # f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI) = "
                # f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}, {hvs_qnehvi[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )

            time_all.append(t1-t0)
        else:
            print(".", end="")

    print("average_time", np.sum(time_all))

    ##########################################################################################
    # save train_x, train_obj to csvs
    #
    # column_names = ["res_time",	"equiv", "conc_dfnb", "temp", "sty", "e_factor"]
    #
    # res_qparego = torch.cat((train_x_qparego, train_obj_qparego), dim=1)
    # df_res_qparego = pd.DataFrame(res_qparego.numpy(), columns=column_names)
    # df_res_qparego.iloc[:,-1] = -df_res_qparego.iloc[:,-1]
    # df_res_qparego.to_csv(f'results/qparego/repeat_{repeat}.csv', index=False)
    # # df_res_qparego.to_csv(f'results/repeat1_{repeat}.csv', index=False)
    #
    # res_qehvi = torch.cat((train_x_qehvi, train_obj_qehvi), dim=1)
    # df_res_qehvi = pd.DataFrame(res_qehvi.numpy(), columns=column_names)
    # df_res_qehvi.iloc[:, -1] = -df_res_qehvi.iloc[:, -1]
    # df_res_qehvi.to_csv(f'results/qehvi/repeat_{repeat}.csv', index=False)
    # # df_res_qehvi.to_csv(f'results/repeat2_{repeat}.csv', index=False)
    #
    # res_qnehvi = torch.cat((train_x_qnehvi, train_obj_qnehvi), dim=1)
    # df_res_qnehvi = pd.DataFrame(res_qnehvi.numpy(), columns=column_names)
    # df_res_qnehvi.iloc[:, -1] = -df_res_qnehvi.iloc[:, -1]
    # df_res_qnehvi.to_csv(f'results/qnehvi/repeat_{repeat}.csv', index=False)
    # # df_res_qnehvi.to_csv(f'results/repeat3_{repeat}.csv', index=False)
    #
    # res_random = torch.cat((train_x_random, train_obj_random), dim=1)
    # df_res_random = pd.DataFrame(res_random.numpy(), columns=column_names)
    # df_res_random.iloc[:, -1] = -df_res_random.iloc[:, -1]
    # df_res_random.to_csv(f'results/random/repeat_{repeat}.csv', index=False)
    # # df_res_random.to_csv(f'results/repeat4_{repeat}.csv', index=False)
    #
    # ##########################################################################################
    # ## plot results
    #
    fig, axes = plt.subplots(1, 4, figsize=(23, 5), sharex=True, sharey=True)
    algos = ["Random", "qNParEGO", "qEHVI", "qNEHVI"]


    batch_number = torch.cat(
        [
            torch.zeros(2 * (dim + 1)),
            torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
        ]
    ).numpy()
    for i, train_obj in enumerate(
        (
            train_obj_random,
            train_obj_qparego,
            train_obj_qehvi,
            train_obj_qnehvi,
        )
    ):
        sc = axes[i].scatter(
            train_obj[:20, 0].cpu().numpy(),
            -train_obj[:20, 1].cpu().numpy(),
            c='k')

        sc = axes[i].scatter(
            train_obj[20:, 0].cpu().numpy(),
            -train_obj[20:, 1].cpu().numpy(),
             c='r')


        axes[i].set_title(algos[i])
        axes[i].set_xlabel("Objective 1")
    axes[0].set_ylabel("Objective 2")
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    fig.subplots_adjust(right=0.9)

    # fig.savefig(f"results/plot_{repeat}.png", dpi=300)
    plt.show()



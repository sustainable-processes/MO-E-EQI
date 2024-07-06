
## This is for running TSEMO

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import typer
from summit import *

import wandb
from benchmark_1 import ReactionSimulator  # for SnAr reaction
from optimize import *
from utils import WandbRunner


class StrategyType(Enum):
    TSEMO = "TSEMO"

def main(
    repeats: int = 20,
    max_iterations: int = 20,
    num_initial_experiments: int = 20,
    strategy: StrategyType = "TSEMO",
    noise_level: float = 0.01,
    save_dir: str = "results",
    show_plot: bool = True,
    wandb_tracking: bool = True,
    wandb_project: str = "bo_mixed",
    wandb_entity: str = "ceb-sre",
    wandb_artifact_name: str = "mixed_benchmark",
    intialization_data_path: Optional[str] = "initial_design/",

):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Setup experiment
    exp = ReactionSimulator(noise_level=noise_level)



    # Runner class
    if wandb_tracking:
        runner_cls = WandbRunner
    else:
        runner_cls = Runner

    for i in range(1,21):

        if intialization_data_path is not None:

            df = pd.read_csv(intialization_data_path + f"TrainingSet1.csv")
            df = df.rename(columns={"conc": "conc_dfnb"})

            ds = DataSet.from_df(df)
            prev_res = exp.run_experiments(ds)

        else:
            prev_res = None


        # Reset experiment
        exp.reset()

        # Setup optimization
        if strategy == StrategyType.TSEMO:
            strategy_cls = TSEMO(exp.domain)

        else:
            raise ValueError(f"Unknown strategy {strategy}")
        r = runner_cls(
            strategy=strategy_cls,
            experiment=exp,
            max_iterations=max_iterations,
            num_initial_experiments=num_initial_experiments,
        )

        # Setup wandb
        wandb_run = None
        if wandb_tracking:
            config = r.to_dict()
            del config["experiment"]["data"]
            wandb_run = wandb.init(
                project=wandb_project, entity=wandb_entity, config=config
            )

        # Run optimization
        r.run(skip_wandb_initialization=True, prev_res=prev_res)


        # Plot results
        fig, ax = exp.pareto_plot(colorbar=True)
        if show_plot:
            plt.show()
        # fig.savefig(save_dir / f"pareto_plot_repeat_{i}.png",dpi=300)
        if wandb_tracking:
            wandb.log({"pareto_plot": wandb.Image(fig)})
#
        # Save results
        r.save(save_dir / f"repeat_{i}.json")


        # save all results to json file, both initial and optimisation data
        data_dict = r.to_dict()
        data = data_dict['experiment']['data']['data']
        columns = [col[0] for col in data_dict['experiment']['data']['columns']]
        df_opt = pd.DataFrame(data, columns=columns)
        df_opt = DataSet.from_df(df_opt)
        df_all = pd.concat([pd.DataFrame(prev_res), df_opt])
        df_all = pd.DataFrame(df_all.iloc[:, 0:6].values,
                              columns=["res_time", "equiv", "conc_dfnb", "temp", "sty", "e_factor"])
        df_all.to_csv(save_dir / f"repeat_{i}.csv", index=False)



        if wandb_tracking:
            artifact = wandb.Artifact(wandb_artifact_name, type="optimization_result")
            # artifact.add_file(save_dir / f"repeat_{i}.json")
            # artifact.add_file(output_path / f"repeat_{i}_model.pth")
            wandb_run.log_artifact(artifact)
            wandb.finish()
#
#
if __name__ == "__main__":
    typer.run(main)





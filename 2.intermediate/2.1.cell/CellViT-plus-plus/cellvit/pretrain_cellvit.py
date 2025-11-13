# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network for pretraining a model
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import os

import sys
os.environ["WANDB_MODE"] = "online"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.insert(0,project_root)

import wandb
import logging
os.environ["WANDB__SERVICE_WAIT"] = "300"

from cellvit.training.base_ml.base_cli import ExperimentBaseParser
from cellvit.training.experiments.experiment_cellvit_pretraining import (
    ExperimentCellVitPretraining,
)

if __name__ == "__main__":
    # Parse arguments
    configuration_parser = ExperimentBaseParser()
    configuration = configuration_parser.parse_arguments()
    experiment_class = ExperimentCellVitPretraining

    # Setup experiment
    try:
        if "checkpoint" in configuration:
            # continue checkpoint
            experiment = experiment_class(
                default_conf=configuration, checkpoint=configuration["checkpoint"]
            )
            outdir = experiment.run_experiment()
        else:
            experiment = experiment_class(default_conf=configuration)
            if configuration["run_sweep"] is True:
                # run new sweep
                sweep_configuration = experiment_class.extract_sweep_arguments(
                    configuration
                )
                os.environ["WANDB_DIR"] = os.path.abspath(
                    configuration["logging"]["wandb_dir"]
                )
                sweep_id = wandb.sweep(
                    sweep=sweep_configuration, project=configuration["logging"]["project"]
                )
                wandb.agent(sweep_id=sweep_id, function=experiment.run_experiment)
            elif "agent" in configuration and configuration["agent"] is not None:
                # add agent to already existing sweep, not run sweep must be set to true
                configuration["run_sweep"] = True
                os.environ["WANDB_DIR"] = os.path.abspath(
                    configuration["logging"]["wandb_dir"]
                )
                wandb.agent(
                    sweep_id=configuration["agent"], function=experiment.run_experiment
                )
            else:
                # casual run
                outdir = experiment.run_experiment()
    
        wandb.finish()
    
    except Exception as e:
        print("An error occurred:")
        print(e)
        traceback.print_exc()

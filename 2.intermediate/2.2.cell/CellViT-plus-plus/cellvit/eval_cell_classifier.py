import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
os.environ["WANDB_API_KEY"] = "da70cff767bdc711b75b2645db443e9a289db810"
os.environ["WANDB_MODE"] = "online"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

from cellvit.training.base_ml.base_cli import ExperimentBaseParser




configuration_parser = ExperimentBaseParser()
configuration = configuration_parser.parse_arguments()
experiment_class = XXX


experiment = experiment_class(
    default_conf=configuration, checkpoint=configuration["checkpoint"]
)
outdir = experiment.run_experiment()




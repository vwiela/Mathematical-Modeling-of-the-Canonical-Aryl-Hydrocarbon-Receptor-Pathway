from functools import partial
import logging
import os
from pathlib import Path
import pandas as pd

import fides
import amici
import petab
import petab_select
import pypesto.engine
import pypesto.logging
import pypesto.optimize
import pypesto.select
import pypesto.logging
from pypesto.select.postprocessors import (
    multi_postprocessor,
    save_postprocessor,
    model_id_binary_postprocessor,
    report_postprocessor,
)
from petab_select.constants import Method

# copy method to add hierarchical
from petab_select import Model, parameter_string_to_value
from petab_select.constants import PETAB_PROBLEM
from pypesto.optimize.ess import SacessFidesFactory
from pypesto.select import SacessMinimizeMethod
from pypesto.objective import Objective
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from petab.C import ESTIMATE, NOMINAL_VALUE

from typing import Dict, Iterable

# set logging to reduce log-file size
import amici.swig_wrappers
import logging
amici.swig_wrappers.logger.setLevel(logging.CRITICAL)

import argparse
from path_config import PETAB_DIR, RESULTS_DIR, MODEL_SELECTION_RESULTS_DIR

MODEL_SELECTION_RUN = "all_ligands_sacess_brute_force_mixed_noise"
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input" / "all_ligands_brute_force_mixed_noise"
OUTPUT_PATH = Path(
    os.environ.get(
        "AHR_MODEL_SELECTION_OUTPUT_DIR",
        MODEL_SELECTION_RESULTS_DIR / MODEL_SELECTION_RUN,
    )
)
TMPDIR = Path(
    os.environ.get(
        "AHR_SACESS_TMPDIR",
        RESULTS_DIR / "tmp" / "sacess" / MODEL_SELECTION_RUN,
    )
)

parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int, default=-1)
parser.add_argument("--script-index", type=int, default=-1)
args = parser.parse_args()

# Postprocessor to save yaml file in intermediate steps of model selection.
def model_yaml_postprocessor(problem, output_path: str):
    model_id = problem.model.model_id
    problem.model.to_yaml(Path(output_path) / (model_id + ".yaml"))

pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)
script_index = args.script_index
if script_index < 0:
    script_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

# do the one failed start by hand
# job_array_id = 1

if __name__ == '__main__':

    run_tmpdir = TMPDIR / str(script_index)
    run_tmpdir.mkdir(parents=True, exist_ok=True)

    minimize_method = SacessMinimizeMethod(
        num_workers=8,
        local_optimizer=SacessFidesFactory(fides_kwargs={
            'verbose': 0,
            'hessian_update': fides.BFGS()
        }),
        tmpdir=str(run_tmpdir),
        max_walltime_s=24*60*60,
        sacess_loglevel=logging.ERROR,
        ess_loglevel=logging.ERROR,
    )

    minimize_options = {
        "startpoint_method": pypesto.startpoint.UniformStartpoints(),
    }

    # Output
    output_path = OUTPUT_PATH
    models_history_output_path = output_path / "history"
    models_best_output_path = output_path / "best"
    models_history_output_path.mkdir(parents=True, exist_ok=True)
    models_best_output_path.mkdir(parents=True, exist_ok=True)
    models_history_output_path.mkdir(parents=True, exist_ok=True)

    # Postprocessor
    save_output_path = output_path / "hdf5" / str(script_index)
    save_output_path.mkdir(parents=True, exist_ok=True)
    report_output_filepath = output_path / "report" / "brute_force_report.tsv"
    report_output_filepath.parent.mkdir(parents=True, exist_ok=True)
    summary_output_filepath = output_path / "summary" / (str(script_index) + ".tsv")
    summary_output_filepath.parent.mkdir(parents=True, exist_ok=True)

    prepared_save_postprocessor = partial(save_postprocessor, output_path=save_output_path)
    prepared_report_postprocessor = partial(report_postprocessor, output_filepath=report_output_filepath)
    prepared_yaml_postprocessor = partial(model_yaml_postprocessor, output_path=models_history_output_path)

    model_postprocessor = partial(
        multi_postprocessor,
        postprocessors=[
            model_id_binary_postprocessor,
            prepared_save_postprocessor, # saves hdf5 files
            prepared_yaml_postprocessor,
            prepared_report_postprocessor,
        ],
    )

    # Setup problem and run
    petab_select_yaml = (
        PETAB_DIR
        / "model_selection"
        / "petab_select_all_ligands_brute_force_mixed_noise"
        / "petab_select_problem_brute_force.yaml"
    )
    petab_select_problem = petab_select.Problem.from_yaml(petab_select_yaml)


    predecessor_model_yaml = INPUT_DIR / "predecessor_models" / f"{script_index}.yaml"
    if not predecessor_model_yaml.exists():
        raise FileNotFoundError(
            f"Predecessor model file not found: {predecessor_model_yaml}. "
            "Set --script-index or SLURM_ARRAY_TASK_ID to an existing predecessor model index."
        )
    predecessor_model = petab_select.Model.from_yaml(predecessor_model_yaml)


    def objective_customizer(objective):
        objective.amici_solver.setRelativeTolerance(1e-12)
        objective.amici_solver.setAbsoluteTolerance(1e-8)
        objective.amici_solver.setMaxSteps(10**6)
        objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
        pass

    calibrated_model = pypesto.select.model_problem.ModelProblem(
        model=predecessor_model,
        criterion=petab_select.constants.Criterion.AICC,
        valid=True,
        autorun=True,
        minimize_method = minimize_method,
        minimize_options=minimize_options,
        objective_customizer=objective_customizer,
        postprocessor = model_postprocessor
    )

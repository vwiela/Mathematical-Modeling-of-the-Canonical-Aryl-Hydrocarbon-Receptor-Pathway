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
from pypesto.objective import Objective
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from petab.C import ESTIMATE, NOMINAL_VALUE

from typing import Dict, Iterable

# Postprocessor to save yaml file in intermediate steps of model selection.
def model_yaml_postprocessor(problem, output_path: str):
    model_id = problem.model.model_id
    problem.model.to_yaml(Path(output_path) / (model_id + ".yaml"))


# now to the main part

pypesto.logging.log(level=logging.WARNING, name="pypesto.petab", console=True)
job_array_id = os.environ.get('SLURM_ARRAY_TASK_ID', 0)

engine = pypesto.engine.MultiProcessEngine()
optimizer = pypesto.optimize.FidesOptimizer(verbose=0, hessian_update=fides.BFGS())
minimize_options = {
    "engine": engine,
    "n_starts": 100,
    "progress_bar": False,
    "filename": None,
    "optimizer": optimizer
}

# Output
output_path = Path(__file__).resolve().parent / "output" / "famos_mixed_noise"
models_history_output_path = output_path / "history"
models_intermediate_output_path = output_path / "intermediate"
models_best_output_path = output_path / "best"
models_history_output_path.mkdir(parents=True, exist_ok=True)
models_best_output_path.mkdir(parents=True, exist_ok=True)
models_intermediate_output_path.mkdir(parents=True, exist_ok=True)

# Postprocessor
save_output_path = output_path / "hdf5" / str(job_array_id)
save_output_path.mkdir(parents=True, exist_ok=True)
report_output_filepath = output_path / "report" / (str(job_array_id) + ".tsv")
report_output_filepath.parent.mkdir(parents=True, exist_ok=True)
summary_output_filepath = output_path / "summary" / (str(job_array_id) + ".tsv")
summary_output_filepath.parent.mkdir(parents=True, exist_ok=True)

prepared_save_postprocessor = partial(save_postprocessor, output_path=save_output_path)
prepared_report_postprocessor = partial(report_postprocessor, output_filepath=report_output_filepath)
prepared_intermediate_postprocessor = partial(model_yaml_postprocessor, output_path=models_intermediate_output_path)

model_postprocessor = partial(
    multi_postprocessor,
    postprocessors=[
        model_id_binary_postprocessor,
        prepared_save_postprocessor, # saves hdf5 files
        prepared_report_postprocessor,
        prepared_intermediate_postprocessor,
    ],
)

# Setup problem and run
petab_select_yaml = Path(__file__).resolve().parent / 'petab' / 'petab_select_mixed_noise' / 'petab_select_problem_famos.yaml'
petab_select_problem = petab_select.Problem.from_yaml(petab_select_yaml)


# Setup problem and run
pypesto_select_problem = pypesto.select.Problem(
    petab_select_problem=petab_select_problem,
    model_postprocessor=model_postprocessor,
)


predecessor_model = petab_select.Model.from_yaml(f"input/famos_mixed_noise/predecessor_models/{job_array_id}.yaml")
candidate_space_kwargs = {
    'summary_tsv': summary_output_filepath,
    'predecessor_model': predecessor_model,
}
candidate_space = petab_select_problem.new_candidate_space(
    **candidate_space_kwargs,
)


def objective_customizer(objective):
    objective.amici_solver.setRelativeTolerance(1e-11)
    objective.amici_solver.setAbsoluteTolerance(1e-16)
    objective.amici_solver.setMaxSteps(10**5)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    pass

_ = pypesto.select.model_problem.ModelProblem(
    model=predecessor_model,
    criterion=petab_select.constants.Criterion.AICC,
    valid=True,
    autorun=True,
    minimize_options=minimize_options,
    objective_customizer=objective_customizer,
    postprocessor = model_postprocessor
)

best_models = pypesto_select_problem.select_to_completion(
    minimize_options=minimize_options,
    candidate_space=candidate_space,
    objective_customizer=objective_customizer,
)

# Save models
petab_select.model.models_to_yaml_list(
    models=pypesto_select_problem.calibrated_models.values(),
    output_yaml=models_history_output_path / f"{job_array_id}.yaml",
)
petab_select.model.models_to_yaml_list(
    models=best_models,
    output_yaml=models_best_output_path / f"{job_array_id}.yaml",
)

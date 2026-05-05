import warnings
warnings.filterwarnings('ignore')

import copy
import pypesto
import petab
import fides
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import pypesto.visualize.model_fit
import amici
import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
import pandas as pd
import libsbml
import sys
from pypesto.store import save_to_hdf5
import dill
import pypesto.hierarchical
from pypesto.store import read_from_hdf5
import pypesto.sample as sample
import os
from pathlib import Path
import csv

import functionalities as func
from optim_functionalities import hierarchical_optimization
from pypesto.result import OptimizerResult, Result

import model_selection_functionalities as msfunc
import petab_select
from path_config import PETAB_DIR, MODEL_SELECTION_RESULTS_DIR

engine = pypesto.engine.MultiProcessEngine()

def objective_customizer(objective):
    objective.amici_solver.setRelativeTolerance(1e-12)
    objective.amici_solver.setAbsoluteTolerance(1e-8)
    objective.amici_solver.setMaxSteps(10**6)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    pass


models_to_sample = ['M100',
 'M11',
 'M114',
 'M117',
 'M318',
 'M328',
 'M331',
 'M335',
 'M360',
 'M535',
 'M64',
 'M66',
 'M77',
 'M78',
 'M8',
 'M83',
 'M87',
 'M88',
 'M97',
 'M99']

model = "bmm_model_reduced"
data = "BpA_3MC_ICZ_mixed_noise"
model_name = f"{model}_{data}"
yaml_file = PETAB_DIR / data / "full_petab_problem.yaml"

petab_problem = petab.Problem.from_yaml(yaml_file)

output_path = MODEL_SELECTION_RESULTS_DIR / "all_ligands_sacess_brute_force_mixed_noise"

models_dict = {}
for model_id in models_to_sample:
    for file in os.listdir(os.path.join(output_path, "history/")):
        if f"{model_id}" == file.split("-")[0]:
            models_dict[model_id] = petab_select.model.Model.from_yaml(os.path.join(output_path, f"history/{file}"))

pypesto_problems_dict = {}
best_vector_dict = {}
rdatas_dict = {}
for model_id, model in models_dict.items():
    pypesto_problem = pypesto.select.model_to_pypesto_problem(model)
    # set the solver
    objective_customizer(pypesto_problem.objective)
    # store teh pypesto problem
    pypesto_problems_dict[model_id] = pypesto_problem
    # get the result
    result = pypesto.store.read_result(os.path.join(output_path, "hdf5", model_id.split("M")[1], f"{model.model_id}.hdf5"), problem=False, sample=False)
    best_vector = result.optimize_result[0].x
    best_vector_dict[model_id] = best_vector
    # simulate one trajectory with amici and store it.
    parameter_mapping = pypesto_problem.objective.parameter_mapping
    edatas = copy.deepcopy(pypesto_problem.objective.edatas)
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    
    conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
    ncond = len(conditions)

    observable_ids = petab_problem.get_observable_ids()
    meas_df = petab_problem.measurement_df
    meas_timepoints = meas_df['time'].unique()
    ntimepoints = len(meas_timepoints)
    sim_timepoints = np.linspace(0, meas_timepoints[-1], 100)

    x = best_vector[
        pypesto_problem.x_free_indices
    ]
    objective_result = pypesto_problem.objective(x, return_dict=True)

    x_dct = dict(zip(pypesto_problem.x_names, best_vector))


    for j in range(ncond):
        edatas[j].setTimepoints(sim_timepoints)

    amici.parameter_mapping.fill_in_parameters(
        edatas=edatas,
        problem_parameters=x_dct,
        scaled_parameters=True,
        parameter_mapping=parameter_mapping,
        amici_model=amici_model,
    )

    rdatas = amici.runAmiciSimulations(
                amici_model,
                amici_solver,
                edatas,
            )

    # store the rdatas
    rdatas_dict[model_id] = rdatas

# save the dictionaries
with open(output_path / "pypesto_problems_dict_best_models.dill", "wb") as f:
    dill.dump(pypesto_problems_dict, f)
with open(output_path / "best_vector_dict_best_models.dill", "wb") as f:
    dill.dump(best_vector_dict, f)
with open(output_path / "rdatas_dict_best_models.dill", "wb") as f:
    dill.dump(rdatas_dict, f)




#------------------------Evaluate sampling of the best models--------------------
# job_id = os.environ.get('SLURM_ARRAY_TASK_ID', 0)
# model_id = models_to_sample[int(job_id)]

# output_path = Path("results/model_selection/all_ligands_sacess_brute_force_mixed_noise/")
# for file in os.listdir(os.path.join(output_path, "history/")):
#     if f"{model_id}" == file.split("-")[0]:
#         yaml_file = file

# model = petab_select.models.Model.from_yaml(os.path.join(output_path, f"history/{yaml_file}"))
# pypesto_problem = pypesto.select.model_to_pypesto_problem(model)

# # set the solver
# objective_customizer(pypesto_problem.objective)

# result = pypesto.store.read_result(os.path.join(output_path, "hdf5", model_id.split("M")[1], f"{model.model_id}.hdf5"), problem=False)
# result.problem = pypesto_problem


# visualize.sampling_parameter_traces(result, use_problem_bounds=True, size=(15,15))
# plt.savefig(f"results/figures/{model_id}_parameter_traces.png")

# visualize.sampling_fval_traces(result, size=(15,15))
# plt.savefig(f"results/figures/{model_id}_fval_trace.png")

# pypesto.sample.geweke_test(result=result)
# pypesto.sample.geweke_test(result=result)

# alpha = [99, 95, 90]
# visualize.sampling_parameter_cis(result, alpha=alpha, size=(15,15))
# plt.savefig(f"results/figures/{model_id}_parameter_cis.png")

# visualize.sampling_1d_marginals(result, size=(15,15))
# plt.savefig(f"results/figures/{model_id}_marginals.png")

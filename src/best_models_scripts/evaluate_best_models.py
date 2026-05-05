import warnings
warnings.filterwarnings('ignore')

import copy
import math
import pypesto
import petab
import fides
import pypesto.petab
import pypesto.select
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

from pypesto.result import OptimizerResult, Result

# import model_selection_functionalities as msfunc
import petab_select

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PETAB_DIR = PROJECT_ROOT / "petab"
MODEL_SELECTION_DIR = PROJECT_ROOT / "results" / "model_selection" / "all_ligands_sacess_brute_force_mixed_noise"
BEST_MODEL_OPTIM_DIR = MODEL_SELECTION_DIR / "best_models_optim"
BEST_MODEL_RESULTS_DIR = MODEL_SELECTION_DIR / "best"

engine = pypesto.engine.MultiProcessEngine()

def objective_customizer(objective):
    objective.amici_solver.setRelativeTolerance(1e-12)
    objective.amici_solver.setAbsoluteTolerance(1e-8)
    objective.amici_solver.setMaxSteps(10**6)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    pass

def vector_to_result(free_par_vector, pypesto_problem):
    fake_result = Result()
    optimizer_result = OptimizerResult(
    id="fake_result_for_problem_with_no_estimated_parameters",
    x=pypesto_problem.objective.pre_post_processor.preprocess(free_par_vector),
    fval=pypesto_problem.objective(free_par_vector),
    grad=None,
    hess=None,
    res=None,
    sres=None,
    n_fval=1,
    n_grad=0,
    n_hess=0,
    n_res=0,
    n_sres=0,
    x0=[],
    fval0=pypesto_problem.objective(free_par_vector),
    history=None,
    exitflag=0,
    time=10,
    message="Fake result for problem with no estimated parameters.",
)
    fake_result.optimize_result.append(optimizer_result)
    fake_result.problem = pypesto_problem
    return fake_result

models_to_evaluate = ['M100',
 'M11',
 'M114',
 'M117',
 'M318',
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

output_path = BEST_MODEL_OPTIM_DIR
root_path = SCRIPT_DIR

models_dict = {}
for model_id in models_to_evaluate:
    model_no = model_id.split("M")[1] 
    for file in os.listdir(os.path.join(root_path, "input/")):
        if f"{model_no}" == file.split(".")[0]:
            models_dict[model_id] = petab_select.model.Model.from_yaml(os.path.join(root_path, f"input/{file}"))

row_list = []
column_names = ["model", "fval", "x"]
pypesto_problems_dict = {}
best_vector_dict = {}
rdatas_dict = {}
for model_id, model in models_dict.items():
    pypesto_problem = pypesto.select.model_to_pypesto_problem(model)
    # set the solver
    objective_customizer(pypesto_problem.objective)
    # store the pypesto problem
    pypesto_problems_dict[model_id] = pypesto_problem
    fig, axs = plt.subplots(3,4,figsize=(12,16))
    for model_run in range(12):
        unique_tag = f"{model_id}_{model_run}"
        # get the result
        result = pypesto.store.read_result(os.path.join(output_path, f"{unique_tag}.hdf5"), problem=False, sample=False)
        pypesto.visualize.waterfall(
            ax=axs[math.floor(model_run/4),model_run%4],
            results=result
        )
    plt.tight_lyaout()
    BEST_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BEST_MODEL_RESULTS_DIR / f"waterfalls_{model_id}.pdf", bbox_inches="tight")
        
          
        #best_vector = result.optimize_result[0].x
        #fval = result.optimize_result[0].fval
        #model_specs = [unique_tag, fval, best_vector]
        #row_list.append(dict( (column_names[i], model_specs[i]) for i in range(3)))        
        #best_vector_dict[unique_tag] = best_vector
        # simulate one trajectory with amici and store it.
        #parameter_mapping = pypesto_problem.objective.parameter_mapping
        #edatas = copy.deepcopy(pypesto_problem.objective.edatas)
        #amici_model = pypesto_problem.objective.amici_model
        #amici_solver = pypesto_problem.objective.amici_solver
    
        #conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
        #ncond = len(conditions)

        #observable_ids = petab_problem.get_observable_ids()
        #meas_df = petab_problem.measurement_df
        #meas_timepoints = meas_df['time'].unique()
        #ntimepoints = len(meas_timepoints)
        #sim_timepoints = np.linspace(0, meas_timepoints[-1], 100)

        #x = best_vector[
        #pypesto_problem.x_free_indices
        #]
        #objective_result = pypesto_problem.objective(x, return_dict=True)

        #x_dct = dict(zip(pypesto_problem.x_names, best_vector))
        
        #for j in range(ncond):
        #    edatas[j].setTimepoints(sim_timepoints)

        #amici.petab.conditions.fill_in_parameters(
        #    edatas=edatas,
        #    problem_parameters=x_dct,
        #    scaled_parameters=True,
        #    parameter_mapping=parameter_mapping,
        #    amici_model=amici_model,
        #)

        #rdatas = amici.runAmiciSimulations(
        #        amici_model,
        #        amici_solver,
        #        edatas,
        #    )

        # store the rdatas
        #rdatas_dict[unique_tag] = rdatas     
        
#model_eval_df = pd.DataFrame(row_list, columns=column_names)
#model_eval_df.to_csv("results/model_selection/all_ligands_sacess_brute_force_mixed_noise/best_models_eval_df.csv")
          

# save the dictionaries
#with open(f"results/model_selection/all_ligands_sacess_brute_force_mixed_noise/pypesto_problems_dict_best_models.dill", "wb") as f:
#    dill.dump(pypesto_problems_dict, f)
#with open(f"results/model_selection/all_ligands_sacess_brute_force_mixed_noise/all_best_vector_dict_best_models.dill", "wb") as f:
#    dill.dump(best_vector_dict, f)
#with open(f"results/model_selection/all_ligands_sacess_brute_force_mixed_noise/rdatas_dict_best_models.dill", "wb") as f:
#    dill.dump(rdatas_dict, f)




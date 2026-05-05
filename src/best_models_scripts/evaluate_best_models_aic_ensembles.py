'''
Assumes that the script evalaute_best_models_ensembles.py has ran 
successfully and that the full-results for each model has been stored.
'''

import warnings
warnings.filterwarnings('ignore')

import copy
import pypesto
import petab
import fides
import pypesto.petab
import pypesto.select
import pypesto.optimize as optimize
from pypesto.ensemble import Ensemble
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
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import euclidean_distances

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

def aic_from_nllh(select_model, nllh):
    n_par = Counter(select_model.parameters.values())["estimate"]
    aic = 2*(n_par+nllh)
    return aic


models_to_consider = ['M100',
 'M11',
 'M114',
 'M117',
 'M360',
 'M64',
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
BEST_MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

models_dict = {}
full_results_dict = {}
best_nllhs_dict = {}
best_aics_dict = {}
for model_id in models_to_consider:
    model_no = model_id.split("M")[1] 
    for file in os.listdir(os.path.join(root_path, "input/")):
        if f"{model_no}" == file.split(".")[0]:
            select_model = petab_select.model.Model.from_yaml(os.path.join(root_path, f"input/{file}"))
            models_dict[model_id] = select_model
            
    pypesto_problem = pypesto.select.model_to_pypesto_problem(select_model)
    # set the solver
    objective_customizer(pypesto_problem.objective)
    full_result = pypesto.Result()
    full_result.problem = pypesto_problem
    for model_run in range(12):
        unique_tag = f"{model_id}_{model_run}"
        # get the result
        result = pypesto.store.read_result(os.path.join(output_path, f"{unique_tag}.hdf5"), problem=False, sample=False)
        full_result.optimize_result.list.extend(result.optimize_result.list)

    pypesto.store.write_result(full_result, BEST_MODEL_RESULTS_DIR / f"full_result_{model_id}.hdf5", overwrite=True)
    nllh = full_result.optimize_result.fval[0]
    full_results_dict[model_id] = full_result
    best_nllhs_dict[model_id] = nllh
    best_aics_dict[model_id] = petab_select.criteria.calculate_aic(nllh=nllh, n_estimated=len(select_model.to_petab()['petab_problem'].x_free_indices))

with open(BEST_MODEL_RESULTS_DIR / "best_aics_dict.dill", "wb") as f:
    dill.dump(best_aics_dict, f)
with open(BEST_MODEL_RESULTS_DIR / "best_nllhs_dict.dill", "wb") as f:
    dill.dump(best_nllhs_dict, f)


best_aic = np.min(list(best_aics_dict.values()))
AIC_MAX = best_aic + 10

ensemble_trajectory_dict = {}
ensemble_rdatas_dict = {}

for model_id, full_result in full_results_dict.items():
    select_model = models_dict[model_id]
    opt_result_subset = pypesto.result.OptimizeResult()
    for i_start, start in enumerate(full_result.optimize_result.list):
        start_nllh = start.fval
        start_aic = petab_select.criteria.calculate_aic(nllh=start_nllh, n_estimated=len(select_model.to_petab()['petab_problem'].x_free_indices))
        if start_aic < AIC_MAX:
            opt_result_subset.append(start, prefix=f"{i_start}")
    aic_sel_result = pypesto.result.Result()
    aic_sel_result.optimize_result = opt_result_subset

    pypesto_problem = pypesto.select.model_to_pypesto_problem(select_model)
    # set the solver
    objective_customizer(pypesto_problem.objective)
    # make one result objects out of the single optimization runs
    aic_sel_result.problem = pypesto_problem

    if len(aic_sel_result.optimize_result.list) > 500:
        considered_starts = np.arange(start=0, stop=len(aic_sel_result.optimize_result.list), step=np.round(len(aic_sel_result.optimize_result.list)/500), dtype=int)
    else:
        considered_starts = range(len(aic_sel_result.optimize_result.list))
    for index in considered_starts:
        x = aic_sel_result.optimize_result.list[index].x
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
        sim_timepoints = np.linspace(0, meas_timepoints[-1], 1000)

        # testing only, check that objective is callable on the vector
        #objective_result = pypesto_problem.objective(x, return_dict=True)
        
        x_full = pypesto_problem.get_full_vector(x)
        x_dct = dict(zip(pypesto_problem.x_names, x_full))

        for j in range(ncond):
            edatas[j].setTimepoints(sim_timepoints)

        amici.petab.conditions.fill_in_parameters(
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

        # store the rdatas if successful
        if np.all([rdatas[i].status == 0 for i in range(len(rdatas))]):
            model_traj_dict = {}
            model_traj_dict["t"] = rdatas[0].t
            for k in range(3):
                model_traj_dict[rdatas[k].id] = rdatas[k].x
                model_traj_dict[f"obs_{rdatas[k].id}"] = rdatas[k].y
            ensemble_trajectory_dict[f"{model_id}_{index}"] = model_traj_dict
            ensemble_rdatas_dict[f"{model_id}_{index}"] = rdatas
        else:
            print(f"Simulation failed for model: {model_id}_{index}")
            
    
# save the ensemble
with open(MODEL_SELECTION_DIR / "aic_trajectories_ensemble.dill", "wb") as f:
    dill.dump(ensemble_trajectory_dict, f)        
        
    
time_series = np.stack([np.concatenate([rdata[i].y for i in range(3)], axis=1) for _, rdata in ensemble_rdatas_dict.items()])

# Standardizing the data
scaler = StandardScaler()
normalized_time_series = np.copy(time_series)
for i in range(time_series.shape[-1]):
    max = np.max(time_series[:,:,i])
    min = np.min(time_series[:,:,i])
    normalized_time_series[:,:,i] = max - time_series[:,:,i]/(max-min)

def overall_distance_matrix_euclid(time_series, log=False):
    """
    Compute the overall distance matrix averaged over all conditions using eucliedean distance pointwise.
    ----------------------------------------------------------------
    Parameters
    ----------
    time_series : np.ndarray
        A 3D array of shape (n_samples, n_timepoints, n_conditions) containing the time series data.
    ----------------------------------------------------------------
    Returns
    -------
    dist_matrix : np.ndarray
        A 2D array representing the average distance matrix across all conditions.
    ----------------------------------------------------------------
    This function computes the average distance matrix across all conditions in the provided time series data.
    """
    dist_matrices = []
    for i in range(time_series.shape[-1]):
        dist_matrices.append(euclidean_distances(time_series[:, :, i]))
    dist_matrix = np.mean(dist_matrices, axis=0)
    return dist_matrix

dist_matrix = overall_distance_matrix_euclid(normalized_time_series)
all_traj_model = AgglomerativeClustering(n_clusters=5, metric='precomputed', linkage="complete", compute_distances=True, compute_full_tree=True)
all_traj_model.fit(dist_matrix)


# save the time series
with open(MODEL_SELECTION_DIR / "aic_trajectories_timeseries.dill", "wb") as f:
    dill.dump(normalized_time_series, f)   

# save the distance matrix
with open(MODEL_SELECTION_DIR / "aic_trajectories_dist_matrix.dill", "wb") as f:
    dill.dump(dist_matrix, f)   

# save the cluster model
with open(MODEL_SELECTION_DIR / "aic_trajectories_5_clusters.dill", "wb") as f:
    dill.dump(all_traj_model, f)   

# more clusters
dist_matrix = overall_distance_matrix_euclid(normalized_time_series)
all_traj_model = AgglomerativeClustering(n_clusters=10, metric='precomputed', linkage="complete", compute_distances=True, compute_full_tree=True)
all_traj_model.fit(dist_matrix)

# save the cluster model
with open(MODEL_SELECTION_DIR / "aic_trajectories_10_clusters.dill", "wb") as f:
    dill.dump(all_traj_model, f)   

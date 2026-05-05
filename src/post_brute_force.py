from pathlib import Path

import numpy as np
import pandas as pd
import petab_select
from petab_select.constants import PETAB_PROBLEM
from pypesto.select import model_to_pypesto_problem
from pypesto.result import OptimizerResult, Result
# include functionalities
import model_selection_functionalities as msfunc
import functionalities as func
import itertools
import os

import amici
import petab
import pypesto
import pypesto.petab
from path_config import PETAB_DIR, MODEL_SELECTION_RESULTS_DIR

# get path to output folders
os.getcwd()
# output_path = Path(os.path.join(os.getcwd(), "results/model_selection/brute_force_mixed_noise/"))
output_path = MODEL_SELECTION_RESULTS_DIR / "all_ligands_sacess_brute_force_mixed_noise"

petab_select_yaml = PETAB_DIR / "model_selection" / "petab_select_all_ligands_brute_force_mixed_noise" / "petab_select_problem_brute_force.yaml"
petab_select_problem = petab_select.Problem.from_yaml(petab_select_yaml)

hdf5_folder = output_path / "hdf5"

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

subfolder_path = os.path.join(hdf5_folder, str(job_id))

# Check if the item is a folder and if its name is numeric
if os.path.isdir(subfolder_path):
    # List all files in the subfolder
    files = os.listdir(subfolder_path)
    files = [ f for f in files if ".hdf5" in f]
if len(files) > 0:
    model_id = files[0].removesuffix(".hdf5")

model_idx = model_id.split('-')[1]
model_idx = [int(x) for x in list(model_idx)]
model = petab_select_problem.model_space.model_subspaces["M"].indices_to_model(model_idx)

# model = petab_select_problem.model_hash_to_model(model_id)

pypesto_problem = model_to_pypesto_problem(model)
petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]

pypesto_result = pypesto.store.read_result(f"{subfolder_path}/{model_id}.hdf5", problem=False)

# if the model was overfitting just use a different start idx
overfitting_models = [46, 52, 54, 58, 64, 72, 75, 79, 96, 101, 105, 106, 107, 115, 121, 122]
overffiting_new_starts = [27, 8, 17, 4, 25, 2, 2, 8, 29, 7, 1, 6, 14, 13, 6, 50]
if job_id in overfitting_models:
    start_idx = overffiting_new_starts[overfitting_models.index(job_id)]
else:
    start_idx = 0

pypesto_result.problem = pypesto_problem

# set criterion in the petab_select model
model.set_criterion(criterion = petab_select.constants.Criterion.NLLH, value = pypesto_result.optimize_result.get_for_key('fval')[start_idx])
model.compute_criterion(criterion = petab_select.constants.Criterion.AIC)
model.compute_criterion(criterion = petab_select.constants.Criterion.AICC)
model.compute_criterion(criterion = petab_select.constants.Criterion.BIC)
model.set_estimated_parameters(dict(zip(pypesto_problem.x_names, pypesto_result.optimize_result.list[start_idx].x)), scaled=True)

# write petab_select model to yaml
model.to_yaml(f"{output_path}/history/{model_id}.yaml")

# get current report file
report_file = f"{output_path}/report/brute_force_report.tsv"
report_df = pd.read_csv(report_file, sep="\t")

_, clustsize = pypesto.util.assign_clusters(pypesto.util.delete_nan_inf(pypesto_result.optimize_result.get_for_key('fval'))[1])
n_converged = clustsize[0]

# write report file
report_df = pd.concat([report_df, pd.DataFrame(
    {
        "model_id": model_id,
        "total_time": np.sum(pypesto_result.optimize_result.get_for_key('time')),
        "NLLH": model.get_criterion(petab_select.constants.Criterion.NLLH),
        "AIC": model.get_criterion(petab_select.constants.Criterion.AIC),
        "AICc": model.get_criterion(petab_select.constants.Criterion.AICC),
        "BIC": model.get_criterion(petab_select.constants.Criterion.BIC),
        "n_converged": n_converged,
        "used_start": start_idx,
    }, index=[int(job_id)])],
    ignore_index=True
)
report_df.to_csv(report_file, sep="\t", index=False)


# store result with problem
# pypesto.store.write_result(pypesto_result, f"{subfolder_path}/{model_id}.hdf5", overwrite=True, problem=True)
